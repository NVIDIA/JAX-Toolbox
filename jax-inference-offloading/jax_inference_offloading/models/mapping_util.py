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
from copy import deepcopy

import google.protobuf.text_format as text_format
import numpy as np
from vllm import LLM

import jax_inference_offloading.api.param_mapping_pb2 as mapping


def slice_to_proto(slicing) -> mapping.TensorSlice:
  result = mapping.TensorSlice()
  for dim in slicing:
    slice_dim = mapping.TensorSlice.Dim()
    if dim is Ellipsis:
      slice_dim.ellipsis.SetInParent()
    elif isinstance(dim, slice):
      slice_dim.slice.SetInParent()
      if dim.start is not None:
        slice_dim.slice.start = dim.start
      if dim.stop is not None:
        slice_dim.slice.stop = dim.stop
      assert dim.step is None
    else:
      assert isinstance(dim, int), dim
      slice_dim.index.index = dim
    result.dims.append(slice_dim)
  return result


def _proto_to_slice(proto: mapping.TensorSlice):
  result = []
  for dim in proto.dims:
    which = dim.WhichOneof("elem")
    if which == "ellipsis":
      result.append(Ellipsis)
    elif which == "slice":
      args = [x if x > -(2**63) else None for x in [dim.slice.start, dim.slice.stop]]
      result.append(slice(*args))
    else:
      assert which == "index"
      result.append(dim.index.index)
  return tuple(result)


def apply_transform(tensor, transform: mapping.JaxParam.Transform):
  if transform.slice.dims:
    tensor = tensor[_proto_to_slice(transform.slice)]
  if transform.transpose:
    tensor = tensor.transpose(transform.transpose)
  if transform.reshape:
    tensor = tensor.reshape(transform.reshape)
  return tensor


def load_mapping_spec(filename: str) -> mapping.TpModelMappingSpecs:
  with open(filename, "r") as file:
    return text_format.Parse(file.read(), mapping.TpModelMappingSpecs())


def add_sharding_specs(model_mapping: mapping.TpModelMappingSpecs, llm: LLM, jax_tp_size: int):
  per_rank_sharding_specs = llm.collective_rpc("get_tp_sharding_specs")
  vllm_tp_size = len(per_rank_sharding_specs)
  if jax_tp_size >= vllm_tp_size:
    aux_parallelism = jax_tp_size // vllm_tp_size
  else:
    aux_parallelism = -1

  # transpose the sharding specs and verify consistency across TP ranks
  per_tensor_sharding_specs = {}
  for _, per_rank_specs in per_rank_sharding_specs:
    for name, specs in per_rank_specs.items():
      try:
        assert per_tensor_sharding_specs[name] == specs, (
          f"Sharding specs for {name} differ across ranks: {per_tensor_sharding_specs[name]} vs {specs}"
        )
      except KeyError:
        per_tensor_sharding_specs[name] = specs

  # convert to VllmTPShardingSpecs
  augmented_mapping = deepcopy(model_mapping)
  for param in augmented_mapping.mappings:
    if spec := per_tensor_sharding_specs.get(param.vllm_param.name):
      dim, parallelism = spec["dim"], spec["parallelism"]
      assert parallelism > 1, (
        f'Unsharded or singleton sharding parallelism {parallelism} does not '
        f'need explicit specification for {param.vllm_param.name}.'
      )
      # find auxiliary dim for sharding, if JAX TP size is larger than VLLM TP size
      masked_shape = np.array(param.vllm_param.shape)
      masked_shape[dim] = 0
      aux_dim = np.argmax(masked_shape)
      replication_count = spec.get('replication_count', 1)
    else:
      dim, parallelism = 999999, -1  # 999999 is an ugly sentinel value but likely never used in reality
      aux_dim = np.argmax(param.vllm_param.shape)
      replication_count = 1

    param.vllm_param.tp_sharding.dim = dim
    param.vllm_param.tp_sharding.parallelism = parallelism
    param.vllm_param.tp_sharding.aux_dim = aux_dim
    param.vllm_param.tp_sharding.aux_parallelism = aux_parallelism
    param.jax_param.transform.replication_count = replication_count

  return augmented_mapping, vllm_tp_size
