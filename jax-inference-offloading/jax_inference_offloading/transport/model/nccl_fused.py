#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.buffer_callback import buffer_callback
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec

from jax_inference_offloading.api.param_mapping_pb2 import TpModelMappingSpecs
from jax_inference_offloading.api.utils import DataclassFor
from jax_inference_offloading.controller.trainer_client import TrainerClient
from jax_inference_offloading.sharding import PolymorphicMesh
from jax_inference_offloading.transport.tensor.nccl_base import nccl_type
from jax_inference_offloading.transport.tensor.nccl_star import NcclStarTransport
from jax_inference_offloading.timer import Timer

logger = logging.getLogger(__name__)


def _getattr_or_none(obj, attr):
  return getattr(obj, attr) if hasattr(obj, attr) else None


class NcclFusedModelTransport:

  @dataclass(frozen=True)
  class Transform:
    slice: Tuple
    transpose: Tuple
    reshape: Tuple

  def __init__(
    self,
    main_mesh: Union[Mesh, PolymorphicMesh],
    mapping_specs: DataclassFor[TpModelMappingSpecs],
    gateway: TrainerClient,
    transports: List[NcclStarTransport],
    transport_config: Dict[str, Any],
    timer: Timer | None = None,
  ):
    self._main_mesh = main_mesh
    self._mapping_specs = mapping_specs
    self._gateway = gateway
    self._transports = transports
    self._transport_config = transport_config
    self._timer = timer or Timer()

    self._jitted = {}
    self._transport_lookup = {t._comm.device_id():t for t in transports}

  def __call__(self, named_parameters: Dict[str, jax.Array]):
    mapping_specs = self._mapping_specs
    gateway = self._gateway
    transports = self._transports
    transport_config = self._transport_config

    gateway.start_weight_transfer('fused')

    if transport_config['MODE'] == 'fan-in':

      if isinstance(self._main_mesh, PolymorphicMesh):
        coupling_mesh = self._main_mesh.view(
          (transport_config['ROLLOUT_RANKS'],
          transport_config['TRAINER_RANKS'] // transport_config['ROLLOUT_RANKS']),
          ("dst","src")
        )
        axis_dst = coupling_mesh.axis('dst')
        axis_src = coupling_mesh.axis('src')
      else:
        coupling_mesh = jax.make_mesh(
          (transport_config['ROLLOUT_RANKS'],
          transport_config['TRAINER_RANKS'] // transport_config['ROLLOUT_RANKS']),
          ("dst","src")
        )
        axis_dst = 'dst'
        axis_src = 'src'

      # transfer all tensors in one jax.jit
      with coupling_mesh as mesh:

        if '!fused' not in self._jitted:

          def transport_callback(ctx, outputs, inputs):
            _, payload = inputs
            tensor = jnp.from_dlpack(payload)
            self._transport_lookup[tensor.device.local_hardware_id]._comm.send(
              sendbuf=tensor.unsafe_buffer_pointer(),
              count=tensor.size,
              datatype=nccl_type(tensor.dtype.name),
              peer=0,
              stream=ctx.stream,
            )

          @partial(jax.jit, static_argnums=(1,))
          def _fused_transfer(state_dict, mappings):
            barrier = jnp.zeros((), dtype=jnp.uint8)  # for ordering the buffer callbacks and preventing XLA DCE

            for i, param in enumerate(mappings):
              x = state_dict[param.jax_param.name]

              # reshape-transpose-slice
              if transform := _getattr_or_none(param.jax_param, 'transform'):
                if slice_ := _getattr_or_none(transform, 'slice'):
                  x = x[slice_]
                if transpose := _getattr_or_none(transform, 'transpose'):
                  x = x.transpose(transpose)
                if reshape := _getattr_or_none(transform, 'reshape'):
                  x = x.reshape(reshape)

              # reshard
              partition_spec = [None] * len(param.vllm_param.shape)
              sharding_specs = param.vllm_param.tp_sharding
              if sharding_specs.parallelism == transport_config['ROLLOUT_RANKS']:
                partition_spec[sharding_specs.dim] = axis_dst
                partition_spec[sharding_specs.aux_dim] = axis_src
              elif sharding_specs.parallelism in (0, 1):
                partition_spec[sharding_specs.aux_dim] = axis_src
              else:
                raise RuntimeError(
                  f"Unsupported sharding parallelism {sharding_specs.parallelism} for {param.vllm_param.name}. "
                  "Expected 0, 1, or equal to JAX TP size."
                )
              x = lax.with_sharding_constraint(x, PartitionSpec(*partition_spec))

              # send out via external NCCL comms
              per_device_transport = shard_map(
                lambda barrier, x: buffer_callback(transport_callback, barrier)((barrier, x)),
                mesh=mesh,
                in_specs=(PartitionSpec(), PartitionSpec(*partition_spec)),
                out_specs=PartitionSpec(),
              )
              barrier = per_device_transport(barrier, x)

            return barrier

          with self._timer.section("jit"):
            self._jitted['!fused'] = _fused_transfer.lower(named_parameters, mapping_specs.mappings).compile()

        with self._timer.section("reshard+send"):
          self._jitted['!fused'](named_parameters).block_until_ready()

    elif transport_config['MODE'] == 'fan-out':

      k = transport_config['ROLLOUT_RANKS'] // transport_config['TRAINER_RANKS']
      if isinstance(self._main_mesh, PolymorphicMesh):
        coupling_mesh = self._main_mesh.view((transport_config['TRAINER_RANKS'],), ("src",))
        axis_src = coupling_mesh.axis('src')
      else:
        coupling_mesh = jax.make_mesh((transport_config['TRAINER_RANKS'],), ("src",))
        axis_src = 'src'

      # fused fan-out postponed and will be solved together with vLLM in-place recv
      raise NotImplementedError("Fused fan-out transfer is not yet implemented.")

      with coupling_mesh as mesh:

        if '!fused' not in self._jitted:

          def transport_callback(ctx, outputs, inputs):
            _, payloads = inputs
            for i, payload in enumerate(payloads):
              tensor = jnp.from_dlpack(payload)

              # // wip
              nccl.groupStart()
              for peer, buffer in zip(range(1, self._comm.size()), buffers):  # rank 0 is the current GPU
                assert self._comm.device_id() == buffer.device.local_hardware_id
                with cuda.Device(buffer.device.local_hardware_id):
                  stream = cuda.get_current_stream().ptr
                  # print(f'  FAN-OUT to {peer} shape {buffer.shape}')
                  self._comm.send(
                    sendbuf=buffer.unsafe_buffer_pointer(),
                    count=buffer.size,
                    datatype=nccl_type(buffer.dtype.name),
                    peer=peer,
                    stream=stream,
                  )
              nccl.groupEnd()

              # self._transport_lookup[tensor.device.local_hardware_id]._comm.send(
              #   sendbuf=tensor.unsafe_buffer_pointer(),
              #   count=tensor.size,
              #   datatype=nccl_type(tensor.dtype.name),
              #   peer=0,
              #   stream=ctx.stream,
              # )
              # t = self._transport_lookup[shard.device.local_hardware_id]
              # t.scatter(buffers)

          @partial(jax.jit, static_argnums=(1,))
          def _fused_transfer(state_dict, mappings):
            barrier = jnp.zeros((), dtype=jnp.uint8)  # for ordering the buffer callbacks and preventing XLA DCE

            for i, param in enumerate(mappings):
              x = state_dict[param.jax_param.name]

              # reshape-transpose-slice
              if transform := _getattr_or_none(param.jax_param, 'transform'):
                if slice_ := _getattr_or_none(transform, 'slice'):
                  x = x[slice_]
                if transpose := _getattr_or_none(transform, 'transpose'):
                  x = x.transpose(transpose)
                if reshape := _getattr_or_none(transform, 'reshape'):
                  x = x.reshape(reshape)

              # reshard
              partition_spec = [None] * len(param.vllm_param.shape)
              sharding_specs = param.vllm_param.tp_sharding
              if sharding_specs.parallelism == transport_config['ROLLOUT_RANKS']:
                partition_spec[sharding_specs.dim] = axis_src
              elif sharding_specs.parallelism in (0, 1):
                pass
              else:
                raise RuntimeError(
                  f"Unsupported sharding parallelism {sharding_specs.parallelism} for {param.vllm_param.name}. "
                  "Expected 0, 1, or equal to JAX TP size."
                )

              x = lax.with_sharding_constraint(x, PartitionSpec(*partition_spec))

              # split for vLLM ranks
              def split(x):
                if sharding_specs.parallelism == transport_config['ROLLOUT_RANKS']:
                  buffers = jnp.split(shard.data, k, axis=sharding_specs.dim)
                elif sharding_specs.parallelism in (0, 1):
                  buffers = [shard.data] * k
                else:
                  raise RuntimeError(
                    f"Unsupported sharding parallelism {sharding_specs.parallelism} for {param.vllm_param.name}. "
                    "Expected 0, 1, or equal to JAX TP size."
                  )

              shard_map(
                # lambda x:
                mesh=mesh,
                in_specs=(PartitionSpec(*partition_spec)),
                out_specs=PartitionSpec(),
              )

              # send out via external NCCL comms
              per_device_transport = shard_map(
                lambda barrier, x: buffer_callback(transport_callback, barrier)((barrier, x)),
                mesh=mesh,
                in_specs=(PartitionSpec(), PartitionSpec(*partition_spec)),
                out_specs=PartitionSpec(),
              )
              barrier = per_device_transport(barrier, x)

            return barrier

          with self._timer.section("jit"):
            self._jitted['!fused'] = _fused_transfer.lower(named_parameters, mapping_specs.mappings).compile()

        with self._timer.section("reshard+send"):
          self._jitted['!fused'](named_parameters).block_until_ready()

    logger.info("done sending")
