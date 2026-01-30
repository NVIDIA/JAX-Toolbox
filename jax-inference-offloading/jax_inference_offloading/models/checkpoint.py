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
"""Checkpoint loading utilities for JAX models.

This module provides functions to load HuggingFace/vLLM checkpoints directly
into JAX arrays, using the parameter mapping JSON to handle shape transformations
and sharding.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

import jax_inference_offloading.api.param_mapping_pb2 as mapping
from jax_inference_offloading.models.mapping_util import (
    apply_vllm_transform,
    load_mapping_from_json,
)


@dataclass
class MappingConfig:
  """Configuration loaded from a parameter mapping JSON file."""
  mesh_axes: List[str]
  mapping_specs: mapping.TpModelMappingSpecs


def load_mapping_config(json_path: str) -> MappingConfig:
  """Load a parameter mapping JSON file and return mesh_axes + mapping specs.

  Args:
    json_path: Path to the JSON configuration file.

  Returns:
    MappingConfig with mesh_axes and mapping_specs.
  """
  mapping_specs = load_mapping_from_json(json_path)
  return MappingConfig(
    mesh_axes=list(mapping_specs.mesh_axes),
    mapping_specs=mapping_specs,
  )


def _partition_spec_from_proto(
    partition_specs: mapping.JaxParam.PartitionSpecs,
    mesh_axes: List[str],
) -> PartitionSpec:
  """Convert a PartitionSpecs proto to a JAX PartitionSpec.

  Args:
    partition_specs: Proto with axis names (empty string = unsharded).
    mesh_axes: List of valid axis names from mesh_axes config.

  Returns:
    A JAX PartitionSpec.
  """
  axes = []
  for axis in partition_specs.axes:
    if axis == "":
      axes.append(None)
    else:
      if axis not in mesh_axes:
        raise ValueError(
          f"Partition spec axis '{axis}' not found in mesh_axes {mesh_axes}"
        )
      axes.append(axis)
  return PartitionSpec(*axes)


def load_checkpoint_to_jax(
    checkpoint_path: str,
    mapping_specs: mapping.TpModelMappingSpecs,
    mesh: Optional[jax.sharding.Mesh] = None,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Dict[str, jax.Array]:
  """Load vLLM/HuggingFace checkpoint weights into JAX arrays.

  This function reads safetensors files from a checkpoint directory and
  transforms them to match JAX model parameter shapes using the vllm_param.transform
  specifications. If a mesh is provided, parameters are sharded according to
  jax_param.partition_specs.

  Args:
    checkpoint_path: Path to checkpoint directory containing safetensors files.
    mapping_specs: TpModelMappingSpecs proto with parameter mappings.
    mesh: Optional JAX mesh for sharding. If None, arrays are not sharded.
    dtype: Target dtype for loaded arrays (default: bfloat16).

  Returns:
    Dictionary mapping JAX parameter names to JAX arrays.
  """
  try:
    from safetensors import safe_open
  except ImportError:
    raise ImportError(
      "safetensors package is required for checkpoint loading. "
      "Install with: pip install safetensors"
    )

  checkpoint_path = Path(checkpoint_path)
  mesh_axes = list(mapping_specs.mesh_axes)

  # Find all safetensors files in the checkpoint directory
  safetensor_files = list(checkpoint_path.glob("*.safetensors"))
  if not safetensor_files:
    raise FileNotFoundError(
      f"No safetensors files found in {checkpoint_path}"
    )

  # Build a mapping from vLLM param names to file paths
  vllm_name_to_file = {}
  for st_file in safetensor_files:
    with safe_open(st_file, framework="numpy") as f:
      for key in f.keys():
        vllm_name_to_file[key] = st_file

  # Load and transform each parameter
  jax_params = {}

  for param_mapping in mapping_specs.mappings:
    jax_name = param_mapping.jax_param.name
    vllm_name = param_mapping.vllm_param.name

    # Find the safetensors file containing this parameter
    if vllm_name not in vllm_name_to_file:
      raise KeyError(
        f"Parameter '{vllm_name}' not found in checkpoint. "
        f"Available parameters: {list(vllm_name_to_file.keys())[:10]}..."
      )

    st_file = vllm_name_to_file[vllm_name]

    # Load the tensor
    with safe_open(st_file, framework="numpy") as f:
      tensor = f.get_tensor(vllm_name)

    # Apply vLLM -> JAX transform if specified
    if param_mapping.vllm_param.HasField("transform"):
      tensor = apply_vllm_transform(tensor, param_mapping.vllm_param.transform)

    # Convert to JAX array with target dtype
    tensor = jnp.asarray(tensor, dtype=dtype)

    # Apply sharding if mesh is provided and partition_specs are defined
    if mesh is not None and param_mapping.jax_param.HasField("partition_specs"):
      partition_spec = _partition_spec_from_proto(
        param_mapping.jax_param.partition_specs,
        mesh_axes,
      )
      sharding = NamedSharding(mesh, partition_spec)
      tensor = jax.device_put(tensor, sharding)

    jax_params[jax_name] = tensor

  return jax_params
