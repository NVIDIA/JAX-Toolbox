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
"""Prompt Dispatcher: Sends prompts/inference requests to vLLM.

This process is part of a split architecture where:
- JAX Controller: Transfers weights to vLLM, receives inference results
- Prompt Dispatcher (this file): Sends prompts/inference requests to vLLM

The two processes coordinate via pub/sub synchronization through the gateway.
This process waits for the "weights_ready" signal from the JAX controller
before sending each inference request.

This is a lightweight process with no JAX/GPU dependency.

Prerequisites:
- Gateway server running
- vLLM worker running
- JAX controller running

Environment variables:
- GATEWAY_URL: URL of the gateway server (e.g., "localhost:50051")
- NUM_ITERATIONS: Number of iterations to run (default: 3)
"""

import os
import traceback

import grpc
from google.protobuf.wrappers_pb2 import StringValue

from jax_inference_offloading import InferenceConfig, VLLMRolloutRequester
import jax_inference_offloading.api.message_broker_pb2_grpc as broker_grpc
from jax_inference_offloading.api.message_broker_pb2 import SubscribeRequest
from jax_inference_offloading.controller.utils import create_topic

# --- Configuration ---
gateway_url = os.environ.get("GATEWAY_URL", "localhost:50051")
num_iterations = int(os.environ.get("NUM_ITERATIONS", "3"))

# Shared topic ID for synchronization
SYNC_TOPIC = "sync/weights_ready"


def get_prompts():
    """Load prompts for inference.
    
    In a real application, this would load prompts from a dataset,
    generate them dynamically, or receive them from another source.
    """
    return [
        "Explain the theory of relativity in simple terms:",
        "Write a short poem about coding:",
        "What are the benefits of exercise?",
    ]


def main():
    print(f"Prompt Dispatcher starting...")
    print(f"Gateway URL: {gateway_url}")

    # --- Create VLLMRolloutRequester ---
    # This will automatically wait for the JAX controller to complete setup
    # by polling the gateway KV store for the response topic.
    print("Creating VLLMRolloutRequester (waiting for JAX controller setup)...")
    requester = VLLMRolloutRequester(gateway_url=gateway_url)
    print("VLLMRolloutRequester ready.")

    # --- Setup gRPC connection for sync subscription ---
    print("Connecting to gateway for sync subscription...")
    channel = grpc.insecure_channel(gateway_url)
    grpc.channel_ready_future(channel).result(timeout=60)
    broker_stub = broker_grpc.MessageBrokerStub(channel)

    # --- Subscribe to sync signal ---
    print(f"Subscribing to sync topic: {SYNC_TOPIC}")
    sync_stream = broker_stub.SubscriptionStream(
        SubscribeRequest(topics=[create_topic(SYNC_TOPIC)])
    )

    # --- Main Loop ---
    print("\n" + "=" * 80)
    print(f"Waiting for sync signals ({num_iterations} expected)...")
    print("=" * 80)

    iteration = 0
    try:
        for delivery in sync_stream:
            # Extract iteration number from signal
            signal_value = StringValue()
            delivery.message.payload.Unpack(signal_value)
            signal_iteration = int(signal_value.value)

            print(f"\n--- Received weights_ready signal (iteration {signal_iteration}) ---")

            # Load prompts
            prompts = get_prompts()
            print(f"Loaded {len(prompts)} prompts")

            # Create inference config
            config = InferenceConfig(
                max_tokens=256,
                temperature=0.7,
                top_p=0.95,
            )

            # Send inference request
            print("Sending inference request to vLLM...")
            requester.request(prompts, config)
            print("Inference request sent!")

            iteration += 1
            if iteration >= num_iterations:
                print(f"\nCompleted {num_iterations} iterations, exiting.")
                break

    except grpc.RpcError as e:
        if e.code() in (grpc.StatusCode.CANCELLED, grpc.StatusCode.UNAVAILABLE):
            print("Stream closed, shutting down.")
        else:
            print(f"gRPC error: {e}")
            traceback.print_exc()
            raise
    except Exception as e:
        print(f"Error in main loop: {e}")
        traceback.print_exc()
        raise

    # --- Shutdown ---
    print("\nShutting down...")
    requester.shutdown()
    channel.close()
    print("Prompt Dispatcher complete.")


if __name__ == "__main__":
    main()
