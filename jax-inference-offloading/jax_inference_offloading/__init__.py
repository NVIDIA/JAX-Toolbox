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
"""JAX-vLLM Inference Offloading Bridge.

This package provides infrastructure for offloading inference/rollout
generation from JAX training to vLLM, enabling efficient RL post-training.

Quick Start:
    >>> from jax_inference_offloading import VLLMRolloutEngine, InferenceConfig
    >>>
    >>> engine = VLLMRolloutEngine(
    ...     gateway_url="localhost:50051",
    ...     model_name="meta-llama/Llama-3.1-8B-Instruct",
    ...     mesh=jax.make_mesh((8,), ("tp",)),
    ... )
    >>>
    >>> engine.update_weights(my_params)
    >>> output = engine.generate(prompts, InferenceConfig(max_tokens=128))
"""

# Core API types
from jax_inference_offloading.api import (
    CompletionOutput,
    InferenceConfig,
    InferenceOutput,
)

# Engine implementations
from jax_inference_offloading.engines import VLLMRolloutEngine

# Low-level access (for advanced users)
from jax_inference_offloading.jax import OffloadingBridge

__all__ = [
    # Core API types
    "CompletionOutput",
    "InferenceConfig",
    "InferenceOutput",
    # Engines
    "VLLMRolloutEngine",
    # Advanced
    "OffloadingBridge",
]
