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
"""Async Prompt Dispatcher: Autonomously sends prompts/inference requests to vLLM.

This process is part of an asynchronous split architecture where:
- JAX Controller: Accumulates rollout results, computes gradients, pushes weights
- Prompt Dispatcher (this file): Continuously sends prompts without waiting for sync signals
- vLLM Worker: Generates rollouts, checks for weight updates between batches

Unlike the synchronous version, this dispatcher does NOT wait for signals from the
JAX controller. It continuously dispatches prompts and blocks only when the
inference queue is full (backpressure).

This is a lightweight process with no JAX/GPU dependency.

Prerequisites:
- Gateway server running
- vLLM worker running
- JAX controller running (for initial handshake/setup)

Environment variables:
- GATEWAY_URL: URL of the gateway server (e.g., "localhost:50051")
- NUM_BATCHES: Number of batches to dispatch (default: 10, 0 for infinite)
- BATCH_SIZE: Number of prompts per batch (default: 3)
- NUM_ROLLOUTS: Number of rollouts per prompt (default: 4)
- DISPATCH_DELAY: Delay between batches in seconds (default: 0.0)
"""

import os
import time
import traceback

from jax_inference_offloading import InferenceConfig, VLLMRolloutRequester

# --- Configuration ---
gateway_url = os.environ.get("GATEWAY_URL", "localhost:50051")
num_batches = int(os.environ.get("NUM_BATCHES", "10"))
batch_size = int(os.environ.get("BATCH_SIZE", "3"))
num_rollouts = int(os.environ.get("NUM_ROLLOUTS", "4"))
dispatch_delay = float(os.environ.get("DISPATCH_DELAY", "0.0"))


def get_prompts(batch_idx: int):
    """Load prompts for inference.
    
    In a real application, this would load prompts from a dataset,
    generate them dynamically, or receive them from another source.
    
    Args:
        batch_idx: The current batch index (can be used for deterministic selection).
    
    Returns:
        List of prompt strings.
    """
    # Example prompts - in production, load from dataset
    all_prompts = [
        "Explain the theory of relativity in simple terms:",
        "Write a short poem about coding:",
        "What are the benefits of exercise?",
        "Describe the process of photosynthesis:",
        "What is machine learning?",
        "Explain how neural networks work:",
        "What are the causes of climate change?",
        "Describe the water cycle:",
        "What is quantum computing?",
        "Explain the concept of recursion in programming:",
    ]
    # Rotate through prompts based on batch index
    start_idx = (batch_idx * batch_size) % len(all_prompts)
    prompts = []
    for i in range(batch_size):
        prompts.append(all_prompts[(start_idx + i) % len(all_prompts)])
    return prompts


def main():
    print("Async Prompt Dispatcher starting...")
    print(f"Gateway URL: {gateway_url}")
    print(f"Batch size: {batch_size}")
    print(f"Rollouts per prompt: {num_rollouts}")
    print(f"Number of batches: {num_batches if num_batches > 0 else 'infinite'}")

    # --- Create VLLMRolloutRequester ---
    # This will automatically wait for the JAX controller to complete setup
    # by polling the gateway KV store for the response topic.
    print("Creating VLLMRolloutRequester (waiting for JAX controller setup)...")
    requester = VLLMRolloutRequester(gateway_url=gateway_url)
    print("VLLMRolloutRequester ready.")

    # --- Create inference config ---
    config = InferenceConfig(
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
        n=num_rollouts,
    )

    # --- Main Loop (Autonomous) ---
    print("\n" + "=" * 80)
    print("Starting autonomous dispatch loop...")
    print("(Will block if inference queue is full)")
    print("=" * 80)

    batch_idx = 0
    try:
        while True:
            # Check termination condition
            if num_batches > 0 and batch_idx >= num_batches:
                print(f"\nCompleted {num_batches} batches, exiting.")
                break

            # Load prompts for this batch
            prompts = get_prompts(batch_idx)
            
            # Send inference request with streaming enabled
            # This may block if the inference queue is full (backpressure)
            print(f"Dispatching batch {batch_idx + 1} with {len(prompts)} prompts...")
            batch_id = requester.request(
                prompts=prompts,
                config=config,
                streaming=True,  # Enable per-rollout streaming
            )
            print(f"  Batch dispatched: batch_id={batch_id}")

            batch_idx += 1

            # Optional delay between batches
            if dispatch_delay > 0:
                time.sleep(dispatch_delay)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error in dispatch loop: {e}")
        traceback.print_exc()
        raise

    # --- Shutdown ---
    print("\nShutting down...")
    requester.shutdown()
    print("Async Prompt Dispatcher complete.")


if __name__ == "__main__":
    main()
