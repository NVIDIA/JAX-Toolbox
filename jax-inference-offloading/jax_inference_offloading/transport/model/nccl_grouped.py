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

from typing import Protocol, runtime_checkable

from jax_inference_offloading.api.param_mapping_pb2 import TpModelMappingSpecs
from jax_inference_offloading.api.utils import DataclassFor
from jax_inference_offloading.sharding import PolymorphicMesh


@runtime_checkable
class WeightTransferGateway(Protocol):
  """Protocol for objects that can trigger weight transfer."""
  def start_weight_transfer(self, mode: str) -> None:
    """Signal the rollout side to start receiving weights."""
    ...
from jax_inference_offloading.transport.tensor.nccl_base import nccl_type
from jax_inference_offloading.transport.tensor.nccl_star import NcclStarTransport
from jax_inference_offloading.timer import Timer

logger = logging.getLogger(__name__)


def _getattr_or_none(obj, attr):
  return getattr(obj, attr) if hasattr(obj, attr) else None


class NcclGroupedModelTransport:

  @dataclass(frozen=True)
  class Transform:
    slice: Tuple
    transpose: Tuple
    reshape: Tuple

  def __init__(
    self,
    main_mesh: Union[Mesh, PolymorphicMesh],
    mapping_specs: DataclassFor[TpModelMappingSpecs],
    gateway: WeightTransferGateway,
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
    transport_config = self._transport_config

    gateway.start_weight_transfer('grouped')

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
            """Send all tensors in a grouped NCCL operation."""
            from cupy.cuda import nccl
            _, tensor_tuple = inputs

            # Get the device from the first tensor
            first_tensor = jnp.from_dlpack(tensor_tuple[0])
            comm = self._transport_lookup[first_tensor.device.local_hardware_id]._comm

            # Group all sends together for better performance
            nccl.groupStart()
            for i, payload in enumerate(tensor_tuple):
              tensor = jnp.from_dlpack(payload)
              comm.send(
                sendbuf=tensor.unsafe_buffer_pointer(),
                count=tensor.size,
                datatype=nccl_type(tensor.dtype.name),
                peer=0,
                stream=ctx.stream,
              )
            nccl.groupEnd()

          @partial(jax.jit, static_argnums=(1,))
          def _fused_transfer(state_dict, mappings):
            # Collect all transformed and resharded tensors
            transformed_tensors = []
            partition_specs = []

            for i, param in enumerate(mappings):
              x = state_dict[param.jax_param.name]

              # slice -> replicate -> transpose -> reshape
              if transform := _getattr_or_none(param.jax_param, 'transform'):
                if slice_ := _getattr_or_none(transform, 'slice'):
                  x = x[slice_]
                if transform.replication_count > 1:
                  x = jnp.concatenate([x] * transform.replication_count, axis=transform.replication_axis)
                if transpose := _getattr_or_none(transform, 'transpose'):
                  x = x.transpose(transpose)
                if reshape := _getattr_or_none(transform, 'reshape'):
                  x = x.reshape(reshape)

              # reshard
              partition_spec = [None] * len(param.vllm_param.shape)
              sharding_specs = param.vllm_param.tp_sharding
              rep_count = getattr(transform, 'replication_count', 1) if transform else 1
              if sharding_specs.parallelism * rep_count > 1:
                partition_spec[sharding_specs.dim] = axis_dst
                partition_spec[sharding_specs.aux_dim] = axis_src
              elif sharding_specs.parallelism == -1:
                partition_spec[sharding_specs.aux_dim] = axis_src
              else:
                raise RuntimeError(
                  f"Unsupported sharding parallelism {sharding_specs.parallelism} for {param.vllm_param.name}. "
                  "Expected -1, or a value > 1 when combined with replication_count."
                )
              x = lax.with_sharding_constraint(x, PartitionSpec(*partition_spec))

              transformed_tensors.append(x)
              partition_specs.append(tuple(partition_spec))

            # Convert to tuple for pytree compatibility
            tensor_tuple = tuple(transformed_tensors)

            # Single barrier for all transfers
            barrier = jnp.zeros((), dtype=jnp.uint8)

            # Send all tensors in one grouped operation
            per_device_transport = shard_map(
              lambda barrier, tensors: buffer_callback(transport_callback, barrier)((barrier, tensors)),
              mesh=mesh,
              in_specs=(PartitionSpec(), tuple(PartitionSpec(*spec) for spec in partition_specs)),
              out_specs=PartitionSpec(),
            )
            barrier = per_device_transport(barrier, tensor_tuple)

            return barrier

          self._jitted['!fused'] = _fused_transfer.lower(named_parameters, mapping_specs.mappings).compile()

        self._jitted['!fused'](named_parameters).block_until_ready()

    elif transport_config['MODE'] == 'fan-out':

      k = transport_config['ROLLOUT_RANKS'] // transport_config['TRAINER_RANKS']
      if isinstance(self._main_mesh, PolymorphicMesh):
        coupling_mesh = self._main_mesh.view((transport_config['TRAINER_RANKS'],), ("src",))
        axis_src = coupling_mesh.axis('src')
      else:
        coupling_mesh = jax.make_mesh((transport_config['TRAINER_RANKS'],), ("src",))
        axis_src = 'src'

      with coupling_mesh as mesh:

        if '!fused' not in self._jitted:

          def transport_callback(ctx, outputs, inputs):
            """Scatter all tensors in a grouped NCCL operation."""
            from cupy.cuda import nccl
            _, all_buffers_list = inputs

            # Get the device and comm from the first buffer
            first_buffer_list = all_buffers_list[0]
            first_buffer = jnp.from_dlpack(first_buffer_list[0])
            comm = self._transport_lookup[first_buffer.device.local_hardware_id]._comm

            # Group all scatters together for better performance
            nccl.groupStart()
            for param_idx, buffers_for_param in enumerate(all_buffers_list):
              # Each buffers_for_param contains k buffers, one for each vLLM rank
              for peer, buffer_dlpack in zip(range(1, comm.size()), buffers_for_param):
                buffer = jnp.from_dlpack(buffer_dlpack)
                comm.send(
                  sendbuf=buffer.unsafe_buffer_pointer(),
                  count=buffer.size,
                  datatype=nccl_type(buffer.dtype.name),
                  peer=peer,
                  stream=ctx.stream,
                )
            nccl.groupEnd()

          @partial(jax.jit, static_argnums=(1, 2))
          def _fused_transfer(state_dict, mappings, k):
            # Collect all transformed and resharded tensors
            transformed_tensors = []
            partition_specs = []
            parallelisms = []
            replication_counts = []
            dims = []

            for i, param in enumerate(mappings):
              x = state_dict[param.jax_param.name]

              # slice -> replicate -> transpose -> reshape
              if transform := _getattr_or_none(param.jax_param, 'transform'):
                if slice_ := _getattr_or_none(transform, 'slice'):
                  x = x[slice_]
                if getattr(transform, 'replication_count', 1) > 1:
                  x = jnp.concatenate([x] * transform.replication_count, axis=transform.replication_axis)
                if transpose := _getattr_or_none(transform, 'transpose'):
                  x = x.transpose(transpose)
                if reshape := _getattr_or_none(transform, 'reshape'):
                  x = x.reshape(reshape)

              # reshard
              partition_spec = [None] * len(param.vllm_param.shape)
              sharding_specs = param.vllm_param.tp_sharding
              rep_count = getattr(transform, 'replication_count', 1) if transform else 1
              if sharding_specs.parallelism * rep_count > 1:
                partition_spec[sharding_specs.dim] = axis_src
              elif sharding_specs.parallelism in (-1, 1):
                pass
              else:
                raise RuntimeError(
                  f"Unsupported sharding parallelism {sharding_specs.parallelism} for {param.vllm_param.name}. "
                  "Expected -1, 1, or a value > 1 when combined with replication_count."
                )

              x = lax.with_sharding_constraint(x, PartitionSpec(*partition_spec))

              transformed_tensors.append(x)
              partition_specs.append(tuple(partition_spec))
              parallelisms.append(sharding_specs.parallelism)
              replication_counts.append(rep_count)
              dims.append(sharding_specs.dim)

            # Single barrier for all transfers
            barrier = jnp.zeros((), dtype=jnp.uint8)

            # Define per-device operation that splits/replicates and sends
            def per_device_op(barrier, *tensors):
              all_buffers_list = []
              for i, tensor in enumerate(tensors):
                product = parallelisms[i] * replication_counts[i]
                if product > 1:
                  # Split along the sharding dimension into k pieces
                  buffers = jnp.split(tensor, k, axis=dims[i])
                elif parallelisms[i] in (-1, 1):
                  # Replicate to all vLLM ranks
                  buffers = [tensor] * k
                else:
                  raise RuntimeError(
                    f"Unsupported sharding parallelism {parallelisms[i]}. "
                    "Expected -1, 1, or a value > 1 when combined with replication_count."
                  )

                all_buffers_list.append(buffers)

              return buffer_callback(transport_callback, barrier)((barrier, all_buffers_list))

            # Send all tensors in one grouped operation
            per_device_transport = shard_map(
              per_device_op,
              mesh=mesh,
              in_specs=(PartitionSpec(), *[PartitionSpec(*spec) for spec in partition_specs]),
              out_specs=PartitionSpec(),
            )
            barrier = per_device_transport(barrier, *transformed_tensors)

            return barrier

          self._jitted['!fused'] = _fused_transfer.lower(named_parameters, mapping_specs.mappings, k).compile()

        self._jitted['!fused'](named_parameters).block_until_ready()

    logger.info("done sending")
