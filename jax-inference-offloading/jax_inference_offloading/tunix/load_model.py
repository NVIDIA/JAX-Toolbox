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
import re
from dataclasses import asdict, replace

import jax

from jax_inference_offloading.sharding import PolymorphicMesh
from tunix.models.dummy_model_creator import create_dummy_model


def load_model(name, mesh: jax.sharding.Mesh = None, checkpoint_path: str = None, dtype: jax.dtypes = None, random_seed: int = 42):

  # Llama 3 Base/Instruct
  if m := re.match(r'(?i)^(?:[^/]+/)?(?:Meta-)?Llama-3(?:\.\d+)?-(?P<size>\d+B)(?:-Instruct)?$', name):
    from tunix.models.llama3.model import Llama3, ModelConfig
    from tunix.models.llama3.params import create_model_from_safe_tensors
    config_factory = {
      '1B': ModelConfig.llama3_2_1b,
      '3B': ModelConfig.llama3_2_3b,
      '8B': ModelConfig.llama3_1_8b,
      '70B': ModelConfig.llama3_70b,
      '405B': ModelConfig.llama3_405b,
    }
    try:
      config = config_factory[m.group('size')]()
    except KeyError:
      raise ValueError(f"Unsupported model size {m.group('size')} for Llama 3")

    if isinstance(mesh, PolymorphicMesh.PolymorphicMeshView):
      config = replace(
        config,
        shd_config=replace(
          config.shd_config,
          **{
            cls: tuple(map(mesh.axis, sharding_specs))
            for cls, sharding_specs in asdict(config.shd_config).items()
          }
        )
      )

    if checkpoint_path:
      model = create_model_from_safe_tensors(checkpoint_path, config, mesh=mesh, dtype=dtype)
    else:
      model = create_dummy_model(
        Llama3,
        config,
        mesh=mesh,
        dtype=dtype,
        random_seed=random_seed,
      )
    return model
  # Gemma 3 Base/IT/PT
  elif m := re.match(r'(?i)^(?:google/)?gemma-?3(?:\.[0-9]+)?-(?P<size>270m|1b|4b|12b|27b)(?:-(?:it|pt))?$', name):
    from tunix.models.gemma3.model import Gemma3, ModelConfig
    from tunix.models.gemma3.params import create_model_from_checkpoint
    config_factory = {
      '270M': ModelConfig.gemma3_270m,
      '1B': ModelConfig.gemma3_1b,
      '4B': ModelConfig.gemma3_4b,
      '12B': ModelConfig.gemma3_12b,
      '27B': ModelConfig.gemma3_27b,
    }
    try:
      config = config_factory[m.group('size')]()
    except KeyError:
      raise ValueError(f"Unsupported model size {m.group('size')} for Gemma 3")

    if isinstance(mesh, PolymorphicMesh.PolymorphicMeshView):
      config = replace(
        config,
        shd_config=replace(
          config.shd_config,
          **{
            cls: tuple(map(mesh.axis, sharding_specs))
            for cls, sharding_specs in asdict(config.shd_config).items()
          }
        )
      )

    if checkpoint_path:
      model = create_model_from_checkpoint(checkpoint_path, config, mesh=mesh, dtype=dtype)
    else:
      model = create_dummy_model(
        Gemma3,
        config,
        mesh=mesh,
        dtype=dtype,
        random_seed=random_seed,
      )
    return model
  else:
    raise ValueError(f"Unsupported model name: {name}")
