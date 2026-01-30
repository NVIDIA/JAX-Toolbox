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
import json
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
  """Apply a JAX -> vLLM transform to a tensor."""
  if transform.slice.dims:
    tensor = tensor[_proto_to_slice(transform.slice)]
  if transform.transpose:
    tensor = tensor.transpose(transform.transpose)
  if transform.reshape:
    tensor = tensor.reshape(transform.reshape)
  return tensor


def apply_vllm_transform(tensor, transform: mapping.VllmParam.Transform):
  """Apply a vLLM -> JAX transform to a tensor (for checkpoint loading)."""
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


def _parse_slice_from_json(slice_list: list) -> mapping.TensorSlice:
  """Parse a JSON slice specification into a TensorSlice proto.

  Args:
    slice_list: A list of slice specifications. Each element can be:
      - "..." for ellipsis
      - An integer for index
      - A list [start, stop] for slice (use null for None)

  Returns:
    A TensorSlice protobuf message.
  """
  result = mapping.TensorSlice()
  for dim in slice_list:
    slice_dim = mapping.TensorSlice.Dim()
    if dim == "...":
      slice_dim.ellipsis.SetInParent()
    elif isinstance(dim, int):
      slice_dim.index.index = dim
    elif isinstance(dim, list):
      slice_dim.slice.SetInParent()
      if len(dim) >= 1 and dim[0] is not None:
        slice_dim.slice.start = dim[0]
      if len(dim) >= 2 and dim[1] is not None:
        slice_dim.slice.stop = dim[1]
    else:
      raise ValueError(f"Invalid slice specification: {dim}")
    result.dims.append(slice_dim)
  return result


def _parse_jax_transform_from_json(transform_dict: dict) -> mapping.JaxParam.Transform:
  """Parse a JSON transform specification into a JaxParam.Transform proto.

  Args:
    transform_dict: A dictionary with optional keys:
      - "transpose": list of ints (axis permutation)
      - "reshape": list of ints (new shape, -1 for inferred)
      - "slice": list of slice specs
      - "replication_axis": int
      - "replication_count": int

  Returns:
    A JaxParam.Transform protobuf message.
  """
  result = mapping.JaxParam.Transform()

  if "slice" in transform_dict:
    result.slice.CopyFrom(_parse_slice_from_json(transform_dict["slice"]))

  if "transpose" in transform_dict:
    result.transpose.extend(transform_dict["transpose"])

  if "reshape" in transform_dict:
    result.reshape.extend(transform_dict["reshape"])

  if "replication_axis" in transform_dict:
    result.replication_axis = int(transform_dict["replication_axis"])

  if "replication_count" in transform_dict:
    result.replication_count = int(transform_dict["replication_count"])

  return result


def _parse_vllm_transform_from_json(transform_dict: dict) -> mapping.VllmParam.Transform:
  """Parse a JSON transform specification into a VllmParam.Transform proto.

  Used for vLLM -> JAX transforms when loading checkpoints.

  Args:
    transform_dict: A dictionary with optional keys:
      - "transpose": list of ints (axis permutation)
      - "reshape": list of ints (new shape, -1 for inferred)
      - "slice": list of slice specs

  Returns:
    A VllmParam.Transform protobuf message.
  """
  result = mapping.VllmParam.Transform()

  if "slice" in transform_dict:
    result.slice.CopyFrom(_parse_slice_from_json(transform_dict["slice"]))

  if "transpose" in transform_dict:
    result.transpose.extend(transform_dict["transpose"])

  if "reshape" in transform_dict:
    result.reshape.extend(transform_dict["reshape"])

  return result


def _parse_partition_spec_from_json(partition_spec_list: list) -> mapping.JaxParam.PartitionSpecs:
  """Parse a JSON partition spec into a PartitionSpecs proto.

  Args:
    partition_spec_list: A list of axis names or null for unsharded dimensions.
      e.g., ["fsdp", null, "tp"] means shard dim 0 on "fsdp", dim 2 on "tp".

  Returns:
    A JaxParam.PartitionSpecs protobuf message.
  """
  result = mapping.JaxParam.PartitionSpecs()
  for axis in partition_spec_list:
    # Use empty string for null/None (unsharded dimensions)
    result.axes.append(axis if axis is not None else "")
  return result


def load_mapping_from_json(json_path: str) -> mapping.TpModelMappingSpecs:
  """Load parameter mapping from a JSON configuration file.

  The JSON schema is:
  {
    "mesh_axes": ["fsdp", "tp"],
    "num_layers": 32,
    "mappings": [
      {
        "jax_param": {
          "name": "layers.{layer}.attn.q_proj.w",
          "partition_spec": ["fsdp", null, "tp"],
          "transform": { "transpose": [1, 2, 0], "reshape": [-1, 4096] }
        },
        "vllm_param": {
          "name": "model.layers.{layer}.self_attn.q_proj.weight",
          "shape": [4096, 4096],
          "transform": { "transpose": [1, 0] }
        }
      },
      ...
    ]
  }

  - mesh_axes: Axis names for JAX mesh creation (e.g., ["fsdp", "tp"])
  - JAX parameter names should match the NNX module structure (no prefix)
  - vLLM parameter names typically have a "model." prefix (matching vLLM's model structure)
  - Mappings with `{layer}` placeholder are expanded into `num_layers` copies
  - Mappings without `{layer}` are kept as singletons
  - jax_param.transform: Applied when sending JAX -> vLLM (optional)
  - jax_param.partition_spec: Sharding spec for checkpoint loading (optional)
  - vllm_param.transform: Applied when loading vLLM checkpoint -> JAX (optional)

  Args:
    json_path: Path to the JSON configuration file.

  Returns:
    A TpModelMappingSpecs protobuf message with all mappings expanded.
  """
  with open(json_path, "r") as f:
    config = json.load(f)

  num_layers = config.get("num_layers", 0)
  mesh_axes = config.get("mesh_axes", [])
  json_mappings = config.get("mappings", [])

  model_mapping = mapping.TpModelMappingSpecs()

  # Set mesh axes
  model_mapping.mesh_axes.extend(mesh_axes)

  for json_mapping in json_mappings:
    jax_param_spec = json_mapping["jax_param"]
    vllm_param_spec = json_mapping["vllm_param"]

    jax_name = jax_param_spec["name"]
    vllm_name = vllm_param_spec["name"]
    vllm_shape = vllm_param_spec["shape"]

    # Check if this is a templated per-layer mapping
    if "{layer}" in jax_name or "{layer}" in vllm_name:
      # Expand for all layers
      layer_indices = range(num_layers)
    else:
      # Singleton mapping - use None as sentinel
      layer_indices = [None]

    for layer_idx in layer_indices:
      param_mapping = mapping.ParamMapping()

      # Expand layer placeholder if present
      if layer_idx is not None:
        expanded_jax_name = jax_name.replace("{layer}", str(layer_idx))
        expanded_vllm_name = vllm_name.replace("{layer}", str(layer_idx))
      else:
        expanded_jax_name = jax_name
        expanded_vllm_name = vllm_name

      # Set JAX param
      param_mapping.jax_param.name = expanded_jax_name

      # Parse and set JAX transform if present (JAX -> vLLM)
      if "transform" in jax_param_spec:
        transform = _parse_jax_transform_from_json(jax_param_spec["transform"])
        param_mapping.jax_param.transform.CopyFrom(transform)

      # Parse and set partition spec if present (for checkpoint loading)
      if "partition_spec" in jax_param_spec:
        partition_specs = _parse_partition_spec_from_json(jax_param_spec["partition_spec"])
        param_mapping.jax_param.partition_specs.CopyFrom(partition_specs)

      # Set vLLM param
      param_mapping.vllm_param.name = expanded_vllm_name
      param_mapping.vllm_param.shape.extend(vllm_shape)

      # Parse and set vLLM transform if present (vLLM -> JAX for checkpoint loading)
      if "transform" in vllm_param_spec:
        vllm_transform = _parse_vllm_transform_from_json(vllm_param_spec["transform"])
        param_mapping.vllm_param.transform.CopyFrom(vllm_transform)

      model_mapping.mappings.append(param_mapping)

  return model_mapping


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
  # Use protobuf's CopyFrom instead of Python's deepcopy for proper message copying
  augmented_mapping = mapping.TpModelMappingSpecs()
  augmented_mapping.CopyFrom(model_mapping)
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
