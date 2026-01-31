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
"""JAX Controller: Handles weight transfer and receives inference results.

This process is part of a split architecture where:
- JAX Controller (this file): Transfers weights to vLLM, receives inference results
- Prompt Dispatcher: Sends prompts/inference requests to vLLM

The two processes coordinate via pub/sub synchronization through the gateway.

Prerequisites:
- Gateway server running
- vLLM worker running
- Prompt dispatcher running (or will start soon)

Environment variables:
- GATEWAY_URL: URL of the gateway server (e.g., "localhost:50051")
- MODEL_PATH: Path to HuggingFace model checkpoint
- PARAM_MAPPING_PATH: Path to JSON parameter mapping file
- TRANSFER_MODE: Weight transfer mode (grouped/fused/unfused)
- NUM_ITERATIONS: Number of training iterations (default: 3)
"""

import os
import traceback

import jax
import jax.numpy as jnp
from google.protobuf.wrappers_pb2 import StringValue

from jax_inference_offloading import (
    OffloadingSession,
    VLLMTransferEngine,
)
from jax_inference_offloading.timer import Timer
from jax_inference_offloading.models.checkpoint import (
    load_mapping_config,
    load_checkpoint_to_jax,
)
import jax_inference_offloading.api.controller_pb2 as ctrl
from jax_inference_offloading.api.message_broker_pb2 import (
    PublishRequest,
    SubscribeRequest,
)
from jax_inference_offloading.controller.utils import create_topic

timer = Timer()

# --- Configuration ---
model_path = os.environ.get("MODEL_PATH", None)
param_mapping_path = os.environ.get("PARAM_MAPPING_PATH", None)
gateway_url = os.environ.get("GATEWAY_URL", "localhost:50051")
transfer_mode = os.environ.get("TRANSFER_MODE", "grouped")
num_iterations = int(os.environ.get("NUM_ITERATIONS", "3"))

# Shared topic IDs for cross-process coordination
RESULTS_TOPIC = "inference/results/shared"
SYNC_TOPIC = "sync/weights_ready"

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

    # --- Store results topic in KV for rollout process to discover ---
    if jax.process_index() == 0:
        print(f"Storing results topic in KV: {RESULTS_TOPIC}")
        topic_value = StringValue(value=RESULTS_TOPIC)
        request = ctrl.KVPutRequest(key="inference_response_topic")
        request.value.Pack(topic_value)
        session.controller_stub.KVPut(request)

    # --- Subscribe to inference results ---
    if jax.process_index() == 0:
        print(f"Subscribing to results topic: {RESULTS_TOPIC}")

    results_stream = session.broker_stub.SubscriptionStream(
        SubscribeRequest(topics=[create_topic(RESULTS_TOPIC)])
    )

    # --- Main Loop ---
    if jax.process_index() == 0:
        print("\n" + "=" * 80)
        print(f"Starting main loop ({num_iterations} iterations)")
        print("=" * 80)

    try:
        for iteration in range(num_iterations):
            if jax.process_index() == 0:
                print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

            # 1. Transfer weights to vLLM
            if jax.process_index() == 0:
                print("Transferring weights to vLLM...")

            with timer.section(f"transfer.iter{iteration}"):
                transfer_engine.update_weights(params)

            if jax.process_index() == 0:
                print("Weights transferred successfully!")

            # 2. Signal rollout process that weights are ready
            if jax.process_index() == 0:
                print(f"Signaling prompt dispatcher (iteration {iteration})...")
                sync_msg = PublishRequest()
                sync_msg.topic.CopyFrom(create_topic(SYNC_TOPIC))
                iteration_value = StringValue(value=str(iteration))
                sync_msg.message.payload.Pack(iteration_value)
                session.broker_stub.Publish(sync_msg)

            # 3. Wait for inference results
            if jax.process_index() == 0:
                print("Waiting for inference results...")

            with timer.section(f"receive_results.iter{iteration}"):
                delivery = next(results_stream)
                result = ctrl.InferenceResponse()
                delivery.message.payload.Unpack(result)

            # 4. Process results
            if jax.process_index() == 0:
                print(f"Received {len(result.outputs)} outputs")
                for i, output in enumerate(result.outputs[:3]):  # Show first 3
                    text_preview = output.generated_text[:100].replace("\n", " ")
                    print(f"  Output {i + 1}: {text_preview}...")

            # In a real training loop, you would:
            # - Compute rewards/losses from the generated text
            # - Update params with gradient descent
            # - params = train_step(params, result)

    except Exception as e:
        if jax.process_index() == 0:
            print(f"Error in main loop: {e}")
            traceback.print_exc()
        raise

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
        print("JAX Controller complete.")


if __name__ == "__main__":
    main()
