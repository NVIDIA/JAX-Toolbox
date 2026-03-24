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

Split Architecture:
    - JAX Controller: Uses VLLMTransferEngine to transfer weights, receives results via gRPC
    - Prompt Dispatcher: Uses VLLMRolloutRequester to send inference requests
    - vLLM Worker: Runs vLLM, receives weights and inference requests
    - Gateway: gRPC message broker coordinating all processes
"""

# Core API types
from jax_inference_offloading.api import (
    CompletionOutput,
    InferenceConfig,
    InferenceOutput,
)

# Session
from jax_inference_offloading.session import OffloadingSession

# Engine implementations
from jax_inference_offloading.engines import (
    VLLMRolloutRequester,
    VLLMTransferEngine,
)

__all__ = [
    # Core API types
    "CompletionOutput",
    "InferenceConfig",
    "InferenceOutput",
    # Session
    "OffloadingSession",
    # Engines
    "VLLMRolloutRequester",
    "VLLMTransferEngine",
]
