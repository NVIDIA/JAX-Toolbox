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
import itertools as it
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec

from jax_inference_offloading.api.param_mapping_pb2 import TpModelMappingSpecs
from jax_inference_offloading.api.utils import DataclassFor
from jax_inference_offloading.controller.trainer_client import TrainerClient
from jax_inference_offloading.sharding import PolymorphicMesh
from jax_inference_offloading.transport.tensor.nccl_star import NcclStarTransport
from jax_inference_offloading.timer import Timer

logger = logging.getLogger(__name__)


def _getattr_or_none(obj, attr):
  return getattr(obj, attr) if hasattr(obj, attr) else None

@partial(jax.jit, static_argnums=(1, 2))
def _reshard(x, transform, partition_spec):
  if transform:
    if slice_ := _getattr_or_none(transform, 'slice'):
      x = x[slice_]
    if transform.replication_count > 1:
      x = jnp.concatenate([x] * transform.replication_count, axis=transform.replication_axis)
    if transpose := _getattr_or_none(transform, 'transpose'):
      x = x.transpose(transpose)
    if reshape := _getattr_or_none(transform, 'reshape'):
      x = x.reshape(reshape)
  x = lax.with_sharding_constraint(x, PartitionSpec(*partition_spec))
  return x


class NcclUnfusedModelTransport:

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

    gateway.start_weight_transfer('unfused')

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

      for i, param in enumerate(mapping_specs.mappings):

        payload = named_parameters[param.jax_param.name]

        with coupling_mesh as mesh:

          if param.jax_param.name not in self._jitted:  # compile resharding function and cache it
            # plan resharding
            partition_spec = [None] * len(param.vllm_param.shape)
            sharding_specs = param.vllm_param.tp_sharding
            transform = param.jax_param.transform
            if sharding_specs.parallelism * transform.replication_count > 1:
              partition_spec[sharding_specs.dim] = axis_dst
              partition_spec[sharding_specs.aux_dim] = axis_src
            elif sharding_specs.parallelism == -1:
              # partition_spec[sharding_specs.dim] = None  # already None
              partition_spec[sharding_specs.aux_dim] = axis_src
            else:
              raise RuntimeError(
                f"Unsupported sharding parallelism {sharding_specs.parallelism} for {param.vllm_param.name}. "
                "Expected -1 (unsharded) or an integer > 1."
              )
            partition_spec = tuple(partition_spec)

            # compile resharding function
            with self._timer.section("jit"):
              self._jitted[param.jax_param.name] = _reshard.lower(
                payload,
                transform,
                partition_spec,
              ).compile()

          # reshard
          with self._timer.section("reshard"):
            payload = self._jitted[param.jax_param.name](payload).block_until_ready()

        # send out
        with self._timer.section("send"):
          for shard in payload.addressable_shards:
            t = self._transport_lookup[shard.device.local_hardware_id]
            t.send(shard.data)

          for t in transports:
            t.wait_for_completion()

    elif transport_config['MODE'] == 'fan-out':

      k = transport_config['ROLLOUT_RANKS'] // transport_config['TRAINER_RANKS']
      if isinstance(self._main_mesh, PolymorphicMesh):
        coupling_mesh = self._main_mesh.view((transport_config['TRAINER_RANKS'],), ("src",))
        axis_src = coupling_mesh.axis('src')
      else:
        coupling_mesh = jax.make_mesh((transport_config['TRAINER_RANKS'],), ("src",))
        axis_src = 'src'

      for i, param in enumerate(mapping_specs.mappings):

        payload = named_parameters[param.jax_param.name]

        sharding_specs = param.vllm_param.tp_sharding
        transform = param.jax_param.transform

        with coupling_mesh as mesh:

          if param.jax_param.name not in self._jitted:

            partition_spec = [None] * len(param.vllm_param.shape)
            if sharding_specs.parallelism * transform.replication_count > 1:
              partition_spec[sharding_specs.dim] = axis_src
            elif sharding_specs.parallelism in (-1, 1):
              pass
            else:
              raise RuntimeError(
                f"Unsupported sharding parallelism {sharding_specs.parallelism} for {param.vllm_param.name}. "
                "Expected -1, 1, or a value > 1."
              )
            partition_spec = tuple(partition_spec)

            with self._timer.section("jit"):
              self._jitted[param.jax_param.name] = _reshard.lower(
                payload,
                transform,
                partition_spec,
              ).compile()

        with self._timer.section("reshard"):
          payload = self._jitted[param.jax_param.name](payload).block_until_ready()

        # send out
        with self._timer.section("send"):
          for shard in payload.addressable_shards:
            if sharding_specs.parallelism * transform.replication_count > 1:
              buffers = jnp.split(shard.data, k, axis=sharding_specs.dim)
            elif sharding_specs.parallelism in (0, 1, -1):
              buffers = it.repeat(shard.data, k)
            else:
              raise RuntimeError(
                f"Unsupported sharding parallelism {sharding_specs.parallelism} for {param.vllm_param.name}. "
                "Expected -1, 0, 1, or a value > 1."
              )
            t = self._transport_lookup[shard.device.local_hardware_id]
            t.scatter(buffers)

          for t in transports:
            t.wait_for_completion()

    logger.info("done sending")
