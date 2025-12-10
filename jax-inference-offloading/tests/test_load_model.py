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
import jax
import pytest

from jax_inference_offloading.sharding import PolymorphicMesh
from jax_inference_offloading.tunix.load_model import load_model


@pytest.fixture(params=["vanilla_mesh", "polymorphic_mesh"])
def mesh(request):
  """Create a JAX mesh for testing - either vanilla or polymorphic."""
  num_devices = len(jax.devices())

  if request.param == "vanilla_mesh":
    # Create a vanilla JAX mesh
    mesh_shape = (1, num_devices)
    axis_names = ("fsdp", "tp")
    return jax.make_mesh(
      mesh_shape,
      axis_names,
      axis_types=(jax.sharding.AxisType.Auto,) * len(axis_names)
    )
  elif request.param == "polymorphic_mesh":
    # Create a PolymorphicMesh and return a view
    devices = jax.devices()
    primary_shape = (1, num_devices)
    polymorphic_mesh = PolymorphicMesh(devices, primary_shape)
    # Return a view with standard axis names
    return polymorphic_mesh.view(shape=(1, num_devices), axis_names=("fsdp", "tp"))
  else:
    raise ValueError(f"Unknown mesh type: {request.param}")


@pytest.mark.parametrize("model_name", [
  "meta-llama/Llama-3.2-1B-Instruct",
  "meta-llama/Llama-3.2-3B-Instruct",
  "meta-llama/Llama-3.1-8B-Instruct",
  "meta-llama/Llama-3.1-70B-Instruct",
])
def test_load_model_smoke(model_name, mesh):
  """Test that load_model successfully creates dummy models for various Llama configurations."""
  # Load the model with dummy weights (no checkpoint_path)
  model = load_model(
    name=model_name,
    mesh=mesh,
    checkpoint_path=None,
  )

  # Basic smoke test assertions
  assert model is not None, f"Model should not be None for {model_name}"

  # Verify the model is a valid object with expected structure
  # The model should be a Llama3 instance from tunix
  assert callable(model), f"Model should be callable for {model_name}"
