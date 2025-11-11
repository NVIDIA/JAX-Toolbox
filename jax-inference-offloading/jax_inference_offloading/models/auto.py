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
import jax_inference_offloading.api.param_mapping_pb2 as mapping

from .gemma import get_gemma_2b_mapping, get_gemma_7b_mapping
from .gemma3 import get_gemma3_1b_mapping
from .llama3 import get_llama3_8b_mapping, get_llama3_70b_mapping, get_llama3_405b_mapping


def get_tp_model_mapping(model_name, jax_prefix="model", vllm_prefix="model") -> mapping.TpModelMappingSpecs:
  if model_name in ("google/gemma-2b", "google/gemma-2b-it"):
    return get_gemma_2b_mapping(jax_prefix, vllm_prefix)
  elif model_name in ("google/gemma-7b", "google/gemma-7b-it"):
    return get_gemma_7b_mapping(jax_prefix, vllm_prefix)
  elif model_name in ("google/gemma-3-1b", "google/gemma-3-1b-it"):
    return get_gemma3_1b_mapping(jax_prefix, vllm_prefix)
  elif model_name in (
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
  ):
    return get_llama3_8b_mapping(jax_prefix, vllm_prefix)
  elif model_name in (
    # Llama 3.2 covers only 1B/3B
    # Llama 3.3 is instruct-only at 70B
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
  ):
    return get_llama3_70b_mapping(jax_prefix, vllm_prefix)
  elif model_name in (
    "meta-llama/Meta-Llama-3.1-405B",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
    "meta-llama/Llama-3.1-405B",
    "meta-llama/Llama-3.1-405B-Instruct",
    "meta-llama/Llama-3.1-405B-Instruct-FP8",
  ):
    return get_llama3_405b_mapping(jax_prefix, vllm_prefix)
  raise Exception(f"Unknown model {model_name}.")
