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
from logging import getLogger

import numpy as np
from cupy.cuda import runtime as cudart

from jax_inference_offloading.api.param_mapping_pb2 import TpModelMappingSpecs
from jax_inference_offloading.transport.tensor import get_transport_class
from vllm.distributed.parallel_state import (
  get_tensor_model_parallel_rank,
  get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.linear import (
  ColumnParallelLinear,
  MergedColumnParallelLinear,
  QKVParallelLinear,
  RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

logger = getLogger(__name__)


class VLLMWorkerExtension:
  def device_info(self):
    i = cudart.getDevice()
    props = cudart.getDeviceProperties(i)
    pci_bus_id = cudart.deviceGetPCIBusId(i)

    return (
      f"GPU {i}: {props['name'].decode('utf-8')}, "
      f"Total Memory: {props['totalGlobalMem'] / 1024**3:.2f} GB, "
      f"Compute Capability: {props['major']}.{props['minor']}, "
      f"Multiprocessors: {props['multiProcessorCount']}, "
      f"PCI Bus ID: {pci_bus_id}"
    )

  def set_sharding(self):
    for _, module in self.model_runner.model.named_modules():
      if type(module) in [
        RowParallelLinear,
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
      ]:
        module.weight.is_sharded_weight = True

  def get_tp_sharding_specs(self):
    sharding_specs = {}
    tp_rank = get_tensor_model_parallel_rank()
    tp_world_size = get_tensor_model_parallel_world_size()

    # vLLM's parallel linear modules use `is_sharded_weight` to denote
    # if the incoming weight is already sharded on disk. This may be confusing
    # because the flag does not specify whether the in-memory weight tensor is sharded
    # across TP ranks or not (which is implied and always true for the *ParallelLinear modules).
    # We rewrite the flag to True to force the weight loader to use the path for loading sharded incoming weights

    for name, module in self.model_runner.model.named_modules():
      if isinstance(module, VocabParallelEmbedding):
        # VocabParallelEmbedding sharding is very complicated
        # we will skip assigning it a sharding spec here,
        # so that it will be treated as unsharded by subsequent code
        # and send the whole weight tensor to every rank
        # vLLM's weight loader will handle the cropping upon receiving
        continue
      elif type(module) is RowParallelLinear:
        param = module.weight
        param_name = name + ".weight"
        sharding_specs[param_name] = dict(dim=param.input_dim, parallelism=tp_world_size)
      elif type(module) is ColumnParallelLinear:
        param = module.weight
        param_name = name + ".weight"
        sharding_specs[param_name] = dict(dim=param.output_dim, parallelism=tp_world_size)
      elif type(module) is MergedColumnParallelLinear:
        param = module.weight
        output_dim = param.output_dim
        packed_modules_mapping = self.model_runner.model.packed_modules_mapping
        for packed_name in packed_modules_mapping:
          if packed_name in name:
            for individual_name in packed_modules_mapping[packed_name]:
              unpacked_param_name = name.replace(packed_name, individual_name) + ".weight"
              sharding_specs[unpacked_param_name] = dict(dim=output_dim, parallelism=tp_world_size)
            break
      elif type(module) is QKVParallelLinear:
        """
                QKV sharding:
                - Q: evenly divided across TP ranks
                    e.g. [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7] ->
                        [Q0, Q1] on rank 0
                        [Q2, Q3] on rank 1
                        [Q4, Q5] on rank 2
                        [Q6, Q7] on rank 3
                - K/V: evenly divided across TP ranks, but if number of K/V heads is less than TP world size,
                    then K/V shards are replicated across the TP ranks
                    e.g. [K0, K1] ->
                        [K0] on rank 0
                        [K0] on rank 1
                        [K1] on rank 2
                        [K1] on rank 3
                """
        param = module.weight
        output_dim = param.output_dim
        packed_modules_mapping = self.model_runner.model.packed_modules_mapping
        for packed_name in packed_modules_mapping:
          if packed_name in name:
            for individual_name in packed_modules_mapping[packed_name]:
              unpacked_param_name = name.replace(packed_name, individual_name) + ".weight"
              if "q" in individual_name:
                sharding_specs[unpacked_param_name] = dict(
                  dim=output_dim,
                  parallelism=tp_world_size,
                )
              elif "k" in individual_name or "v" in individual_name:
                sharding_specs[unpacked_param_name] = dict(
                  dim=output_dim,
                  parallelism=module.total_num_kv_heads // module.num_kv_heads,  # parallelism may be less than TP world size due to replication
                  replication_count=module.num_kv_head_replicas,
                )
            break

    return (tp_rank, sharding_specs)

  def create_transport(self, config):
    self._transport_config = config
    transport_cls = get_transport_class(config['BACKEND'])
    self.transport = transport_cls.create_rollout_transport(
      config,
      tp_rank=get_tensor_model_parallel_rank(),
    )
    return repr(self.transport)

  def reset_stage(self):
    """Use a staging container to hold weights before committing them to the model.
    May not be necessary if we choose to call `load_weights` tensor-by-tensor,
    but will complicate NCCL streaming.
    """
    self._staged_weights = []

  def update_weights(self, mapping_specs: TpModelMappingSpecs):
    self.reset_stage()

    tp_rank = get_tensor_model_parallel_rank()

    if self._transport_config['MODE'] == 'fan-in':
      for i, param in enumerate(mapping_specs.mappings):
        sharding_specs = param.vllm_param.tp_sharding
        shape = np.array(param.vllm_param.shape, dtype=np.int32)
        if sharding_specs.parallelism > 0:
          shape[sharding_specs.dim] //= sharding_specs.parallelism

        # logger.warning(f'vLLM TP rank {tp_rank} receiving {param.vllm_param.name} ...')
        weight = self.transport.gather(
          shape, param.vllm_param.dtype or 'bfloat16',
          sharding_specs.aux_dim, sharding_specs.aux_parallelism
        )
        # logger.warning(f'vLLM TP rank {tp_rank} received {param.vllm_param.name} shape {weight.shape}')
        self._staged_weights.append((param.vllm_param.name, weight))

        # TODO: make it optional
        self.sync()
        self.commit_staged_weights(reset=True)

    elif self._transport_config['MODE'] == 'fan-out':
      for i, param in enumerate(mapping_specs.mappings):
        sharding_specs = param.vllm_param.tp_sharding
        shape = np.array(param.vllm_param.shape, dtype=np.int32)
        if sharding_specs.parallelism > 0:
          shape[sharding_specs.dim] //= sharding_specs.parallelism

        raw_specs_str = ' '.join(str(sharding_specs).split('\n'))
        logger.info(f"vLLM expecting: {param.vllm_param.name} shape {shape.tolist()} raw specs {param}")

        weight = self.transport.recv(shape, param.vllm_param.dtype or 'bfloat16')
        self._staged_weights.append((param.vllm_param.name, weight))

        # TODO: make it optional
        self.sync()
        self.commit_staged_weights(reset=True)

    self.sync()
    logger.warning("done receiving")
    self.commit_staged_weights(reset=True)

  def update_weights_grouped(self, mapping_specs: TpModelMappingSpecs):
    self.reset_stage()

    tp_rank = get_tensor_model_parallel_rank()

    if self._transport_config['MODE'] == 'fan-in':
      # Prepare parameter specifications for grouped gathering
      param_specs = []
      param_names = []

      for i, param in enumerate(mapping_specs.mappings):
        sharding_specs = param.vllm_param.tp_sharding
        shape = np.array(param.vllm_param.shape, dtype=np.int32)
        if sharding_specs.parallelism > 0:
          shape[sharding_specs.dim] //= sharding_specs.parallelism

        param_specs.append((
          shape,
          param.vllm_param.dtype or 'bfloat16',
          sharding_specs.aux_dim,
          sharding_specs.aux_parallelism
        ))
        param_names.append(param.vllm_param.name)

      # Receive all weights in one grouped operation
      logger.warning(f'vLLM TP rank {tp_rank} receiving {len(param_specs)} weights via grouped gather...')
      weights = self.transport.gather_grouped(param_specs)
      logger.warning(f'vLLM TP rank {tp_rank} received all weights')

      # Stage all received weights
      for name, weight in zip(param_names, weights):
        self._staged_weights.append((name, weight))

      # Commit all weights at once
      self.commit_staged_weights(reset=True)

    elif self._transport_config['MODE'] == 'fan-out':
      # Prepare parameter specifications for grouped receiving
      param_specs = []
      param_names = []

      for i, param in enumerate(mapping_specs.mappings):
        sharding_specs = param.vllm_param.tp_sharding
        shape = np.array(param.vllm_param.shape, dtype=np.int32)
        if sharding_specs.parallelism > 0:
          shape[sharding_specs.dim] //= sharding_specs.parallelism

        param_specs.append((shape, param.vllm_param.dtype or 'bfloat16'))
        param_names.append(param.vllm_param.name)

      # Receive all weights in one grouped operation
      logger.warning(f'vLLM TP rank {tp_rank} receiving {len(param_specs)} weights via grouped recv...')
      weights = self.transport.recv_grouped(param_specs)
      logger.warning(f'vLLM TP rank {tp_rank} received all weights')

      # Stage all received weights
      for name, weight in zip(param_names, weights):
        self._staged_weights.append((name, weight))

      # Commit all weights at once
      self.commit_staged_weights(reset=True)

    self.sync()
    logger.warning("done receiving")
    self.commit_staged_weights(reset=True)

  def commit_staged_weights(self, reset=True):
    loaded_weights = self.model_runner.model.load_weights(weights=self._staged_weights)
    if reset:
      self.reset_stage()
    return loaded_weights

  def sync(self):
    self.transport.wait_for_completion()
