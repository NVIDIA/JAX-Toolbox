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
"""Async vLLM Worker: Runs vLLM inference with streaming results and bounded staleness.

This process is part of an asynchronous split architecture where:
- JAX Controller: Accumulates rollout results, computes gradients, pushes weights
- Prompt Dispatcher: Continuously sends prompts without waiting for sync signals
- vLLM Worker (this file): Generates rollouts, receives weight updates via WEIGHT_UPDATES

Key async behavior:
- Receives weight updates via WEIGHT_UPDATES topic (same as sync version)
- Streams individual RolloutResult messages back as each rollout completes
- **Bounded staleness**: vLLM limits itself to MAX_STALENESS prompts before waiting
  for a weight update. This ensures controllable staleness in the weights used.

Staleness control:
- vLLM tracks how many prompts it has processed since the last weight update
- When the count reaches MAX_STALENESS, subsequent inference requests are queued
- Queued requests are processed after the next weight update arrives
- This ensures vLLM never runs more than MAX_STALENESS prompts ahead of weight updates

Assumes:
- JAX and vLLM are running in different processes on the same physical node.
- JAX and vLLM occupy different GPUs.
- MAX_STALENESS >= JAX's UPDATE_INTERVAL to avoid deadlock.
"""

import logging
import os

from jax_inference_offloading.controller.rollout_client import make_async_rollout_client

logger = logging.getLogger(__name__)


def validate_staleness_config(max_staleness, batch_size, update_interval):
    """Validate staleness configuration to prevent deadlocks.
    
    With partial batch processing support, the constraint is simple:
    MAX_STALENESS >= UPDATE_INTERVAL (vLLM can process exactly up to the limit).
    
    Args:
        max_staleness: Maximum prompts before requiring weight update (0 = unlimited).
        batch_size: Number of prompts per batch (informational only).
        update_interval: JAX controller's weight update interval.
    
    Raises:
        ValueError: If configuration would cause deadlock.
    """
    if max_staleness == 0:
        # Unlimited staleness, no constraints
        return
    
    # With partial batch support, vLLM processes exactly up to max_staleness prompts
    # JAX needs update_interval prompts to trigger an update
    if max_staleness < update_interval:
        raise ValueError(
            f"Deadlock: MAX_STALENESS ({max_staleness}) < UPDATE_INTERVAL ({update_interval}). "
            f"vLLM will block after {max_staleness} prompts, but JAX needs {update_interval} "
            f"prompts to trigger weight update. "
            f"Set MAX_STALENESS >= {update_interval}."
        )
    
    print(f"Staleness config validated: vLLM will process up to {max_staleness} prompts "
          f"before waiting for weight update (UPDATE_INTERVAL={update_interval})")


def main():
    from jax_inference_offloading.vllm import LLM

    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    # Parse environment variables
    enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1"
    gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
    tensor_parallel_size = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", len(cuda_visible_devices)))
    distributed_backend = os.environ.get("VLLM_DISTRIBUTED_BACKEND", "mp")
    load_format = os.environ.get("VLLM_LOAD_FORMAT", "dummy")
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    model_path = os.environ.get("MODEL_PATH", None)
    model = model_path or model_name
    param_mapping_path = os.environ.get("PARAM_MAPPING_PATH", None)
    
    # Staleness control: 0 = no limit (unlimited staleness)
    max_staleness = int(os.environ.get("MAX_STALENESS", "0"))
    
    # Read batch size and update interval for validation
    batch_size = int(os.environ.get("BATCH_SIZE", "3"))
    update_interval = int(os.environ.get("UPDATE_INTERVAL", "5"))

    logging.basicConfig(level=logging.INFO)

    gateway_url = os.environ.get("GATEWAY_URL")

    print("=" * 80)
    print("Async vLLM Worker starting...")
    print(f"Model: {model}")
    print(f"Gateway URL: {gateway_url}")
    print(f"Max staleness: {max_staleness if max_staleness > 0 else 'unlimited'}")
    print(f"Batch size: {batch_size}")
    print(f"Update interval: {update_interval}")
    print("=" * 80)
    
    # Validate staleness configuration
    validate_staleness_config(max_staleness, batch_size, update_interval)

    # Create offline inference server
    llm = LLM(
        model=model,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_backend,
        load_format=load_format,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=1024,
    )

    # Subscribe to control messages using async client with staleness control
    rollout_client = make_async_rollout_client(gateway_url, max_staleness=max_staleness)
    rollout_client.subscribe_to_control_messages(llm, mapping_json_path=param_mapping_path)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
