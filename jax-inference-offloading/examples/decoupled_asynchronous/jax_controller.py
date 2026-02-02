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
"""Async JAX Controller: Accumulates rollout results and pushes weight updates.

This process is part of an asynchronous split architecture where:
- JAX Controller (this file): Accumulates rollout results, pushes weight updates
- Prompt Dispatcher: Continuously sends prompts without waiting for sync signals
- vLLM Worker: Generates rollouts, receives weight updates via WEIGHT_UPDATES topic

Key async behavior:
- A separate consumer thread reads RolloutResult messages from the gRPC stream
- Results are accumulated in a thread-safe manner
- Main thread polls for completed groups and triggers weight updates
- Weight updates block on NCCL, but consumer thread keeps reading (no message loss)
- Does NOT send sync signals to prompt dispatcher

Prerequisites:
- Gateway server running
- vLLM worker running
- Prompt dispatcher running

Environment variables:
- GATEWAY_URL: URL of the gateway server (e.g., "localhost:50051")
- MODEL_PATH: Path to HuggingFace model checkpoint
- PARAM_MAPPING_PATH: Path to JSON parameter mapping file
- TRANSFER_MODE: Weight transfer mode (grouped/fused/unfused)
- NUM_ROLLOUTS: Number of rollouts per prompt (default: 4)
- UPDATE_INTERVAL: Push weight update after this many completed prompts (default: 10)
- MAX_COMPLETED_PROMPTS: Stop after this many prompts (default: 100, 0 for infinite)
"""

import os
import threading
import traceback

import jax
import jax.numpy as jnp
from google.protobuf.wrappers_pb2 import StringValue

from jax_inference_offloading import (
    OffloadingSession,
    VLLMTransferEngine,
)
from jax_inference_offloading.engines import RolloutAccumulator, result_consumer_thread
from jax_inference_offloading.timer import Timer
from jax_inference_offloading.models.checkpoint import (
    load_mapping_config,
    load_checkpoint_to_jax,
)
import jax_inference_offloading.api.controller_pb2 as ctrl
from jax_inference_offloading.api.message_broker_pb2 import SubscribeRequest
from jax_inference_offloading.controller.utils import create_topic

timer = Timer()

# --- Configuration ---
model_path = os.environ.get("MODEL_PATH", None)
param_mapping_path = os.environ.get("PARAM_MAPPING_PATH", None)
gateway_url = os.environ.get("GATEWAY_URL", "localhost:50051")
transfer_mode = os.environ.get("TRANSFER_MODE", "grouped")
num_rollouts = int(os.environ.get("NUM_ROLLOUTS", "4"))
update_interval = int(os.environ.get("UPDATE_INTERVAL", "5"))
max_completed_prompts = int(os.environ.get("MAX_COMPLETED_PROMPTS", "100"))

# Shared topic for results
RESULTS_TOPIC = "inference/results/shared"

# Validate required environment variables
if model_path is None:
    raise ValueError("MODEL_PATH environment variable is required")
if param_mapping_path is None:
    raise ValueError("PARAM_MAPPING_PATH environment variable is required")


def main():
    # --- Load Mapping Config ---
    if jax.process_index() == 0:
        print(f"Loading mapping config from {param_mapping_path}")

    with timer.section("load_mapping_config"):
        mapping_config = load_mapping_config(param_mapping_path)

    # Create mesh using axis names from the mapping config
    mesh_shape = (jax.process_count(), jax.local_device_count())
    mesh_axes = tuple(mapping_config.mesh_axes)
    if len(mesh_shape) != len(mesh_axes):
        raise ValueError(
            f"Mesh shape {mesh_shape} does not match mesh_axes {mesh_axes}. "
            f"Expected {len(mesh_shape)} axes, got {len(mesh_axes)}."
        )
    mesh = jax.make_mesh(mesh_shape, mesh_axes)

    if jax.process_index() == 0:
        print(f"Created mesh with shape {mesh_shape} and axes {mesh_axes}")

    # --- Load Checkpoint ---
    if jax.process_index() == 0:
        print(f"Loading checkpoint from {model_path}")

    with timer.section("load_checkpoint"):
        params = load_checkpoint_to_jax(
            checkpoint_path=model_path,
            mapping_specs=mapping_config.mapping_specs,
            mesh=mesh,
            dtype=jnp.bfloat16,
        )

    if jax.process_index() == 0:
        print(f"Loaded {len(params)} parameters")

    # --- Create OffloadingSession ---
    if jax.process_index() == 0:
        print(f"Creating OffloadingSession with gateway_url={gateway_url}")

    with timer.section("create_session"):
        session = OffloadingSession(
            gateway_url=gateway_url,
            mesh=mesh,
            model_path=model_path,
            param_mapping_path=param_mapping_path,
        )

    # --- Create VLLMTransferEngine ---
    if jax.process_index() == 0:
        print("Creating VLLMTransferEngine...")

    with timer.section("create_transfer_engine"):
        transfer_engine = VLLMTransferEngine(
            session=session,
            transfer_mode=transfer_mode,
            timer=timer,
        )

    # --- Store results topic in KV for prompt dispatcher ---
    if jax.process_index() == 0:
        print(f"Storing results topic in KV: {RESULTS_TOPIC}")
        topic_value = StringValue(value=RESULTS_TOPIC)
        request = ctrl.KVPutRequest(key="inference_response_topic")
        request.value.Pack(topic_value)
        session.controller_stub.KVPut(request)

    # --- Subscribe to streamed rollout results ---
    if jax.process_index() == 0:
        print(f"Subscribing to results topic: {RESULTS_TOPIC}")

    results_stream = session.broker_stub.SubscriptionStream(
        SubscribeRequest(topics=[create_topic(RESULTS_TOPIC)])
    )

    # --- Create accumulator and start consumer thread ---
    accumulator = RolloutAccumulator(num_rollouts=num_rollouts)
    
    # Use verbose=True to see each result as it arrives (helpful for debugging)
    verbose_consumer = os.environ.get("VERBOSE_CONSUMER", "0") == "1"
    
    consumer_thread = threading.Thread(
        target=result_consumer_thread,
        args=(results_stream, accumulator, verbose_consumer),
        daemon=True,
        name="ResultConsumer",
    )
    consumer_thread.start()
    
    if jax.process_index() == 0:
        print("Consumer thread started.")

    # --- Tracking variables ---
    completed_groups = []  # For potential use in training
    prompts_since_last_update = 0
    total_prompts_completed = 0
    weight_updates_pushed = 0

    # --- Initial weight transfer ---
    if jax.process_index() == 0:
        print("Performing initial weight transfer...")

    with timer.section("initial_transfer"):
        transfer_engine.update_weights(params)

    if jax.process_index() == 0:
        print("Initial weights transferred.")

    # --- Main Loop ---
    if jax.process_index() == 0:
        print("\n" + "=" * 80)
        print("Starting async accumulation loop (with consumer thread)...")
        print(f"  Rollouts per prompt: {num_rollouts}")
        print(f"  Weight update interval: {update_interval} prompts")
        print(f"  Max prompts: {max_completed_prompts if max_completed_prompts > 0 else 'infinite'}")
        print("=" * 80)

    try:
        while True:
            # Check for consumer thread errors
            if accumulator.error:
                raise accumulator.error

            # --- 1. Get completed groups from accumulator (non-blocking) ---
            groups = accumulator.get_completed_groups(timeout=0.1)
            
            for batch_id, prompt_idx, group in groups:
                completed_groups.append((batch_id, prompt_idx, group))
                prompts_since_last_update += 1
                total_prompts_completed += 1

                if jax.process_index() == 0:
                    print(
                        f"Group complete: batch={batch_id[:8]}..., "
                        f"prompt={prompt_idx} "
                        f"(total completed: {total_prompts_completed})"
                    )

                # --- 2. Push weight update after processing N prompts ---
                if prompts_since_last_update >= update_interval:
                    if jax.process_index() == 0:
                        print(f"\n--- Pushing weight update (after {prompts_since_last_update} prompts) ---")
                        print(f"    (Consumer thread continues reading during NCCL transfer)")

                    # In a real training loop, you would:
                    # 1. Compute rewards for completed_groups
                    # 2. Compute GRPO advantages
                    # 3. Update params with gradient descent
                    # For now, we just transfer the same weights

                    with timer.section(f"weight_update.{weight_updates_pushed}"):
                        # Transfer weights via NCCL (blocks here, but consumer thread keeps reading)
                        transfer_engine.update_weights(params)

                    weight_updates_pushed += 1
                    prompts_since_last_update = 0

                    if jax.process_index() == 0:
                        print(f"Weight update {weight_updates_pushed} complete.")
                        print(f"    (Queued during transfer: {accumulator.completed_queue_size} groups)\n")

                    # Clear completed groups (in real training, use them for gradient computation)
                    # This is where you would implement the GRPO logic
                    completed_groups.clear()

                # --- 3. Check termination condition ---
                if max_completed_prompts > 0 and total_prompts_completed >= max_completed_prompts:
                    if jax.process_index() == 0:
                        print(f"\nReached {max_completed_prompts} completed prompts, stopping.")
                    break
            
            # Break outer loop if termination condition met
            if max_completed_prompts > 0 and total_prompts_completed >= max_completed_prompts:
                break

    except Exception as e:
        if jax.process_index() == 0:
            print(f"Error in main loop: {e}")
            traceback.print_exc()
        raise
    finally:
        # Stop consumer thread
        accumulator.stop()
        consumer_thread.join(timeout=5)

    # --- Print summary ---
    if jax.process_index() == 0:
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Total prompts completed: {total_prompts_completed}")
        print(f"Weight updates pushed: {weight_updates_pushed}")
        print(f"Pending groups remaining: {accumulator.pending_count}")
        print(f"Completed groups in queue: {accumulator.completed_queue_size}")

    # --- Print timing summary ---
    if jax.process_index() == 0:
        print("\n" + "=" * 80)
        print("Timing Summary")
        print("=" * 80)
        timer.summary(sort_by="name", precision=3)

    # --- Shutdown ---
    if jax.process_index() == 0:
        print("\nShutting down...")

    session.shutdown()

    if jax.process_index() == 0:
        print("Async JAX Controller complete.")


if __name__ == "__main__":
    main()
