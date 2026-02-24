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
"""Prompt source for single-controller example.

In sync mode this waits for controller sync signals.
In async mode this dispatches autonomously.
"""

import os
import time

import grpc
from google.protobuf.wrappers_pb2 import StringValue

import jax_inference_offloading.api.message_broker_pb2_grpc as broker_grpc
from jax_inference_offloading import InferenceConfig, VLLMRolloutRequester
from jax_inference_offloading.api.message_broker_pb2 import SubscribeRequest
from jax_inference_offloading.controller.utils import create_topic


def get_prompts(batch_idx: int, batch_size: int):
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
    "Explain recursion in programming with a tiny example:",
  ]
  start_idx = (batch_idx * batch_size) % len(all_prompts)
  prompts = []
  for i in range(batch_size):
    prompts.append(all_prompts[(start_idx + i) % len(all_prompts)])
  return prompts


def run_sync_mode(requester, gateway_url: str):
  num_iterations = int(os.environ.get("NUM_ITERATIONS", "3"))
  batch_size = int(os.environ.get("BATCH_SIZE", "3"))
  num_rollouts = int(os.environ.get("NUM_ROLLOUTS", "1"))
  sync_topic = os.environ.get("SC_SYNC_TOPIC", "sync/weights_ready")

  config = InferenceConfig(
    max_tokens=256,
    temperature=0.7,
    top_p=0.95,
    n=num_rollouts,
  )

  channel = grpc.insecure_channel(gateway_url)
  grpc.channel_ready_future(channel).result(timeout=60)
  broker_stub = broker_grpc.MessageBrokerStub(channel)
  stream = broker_stub.SubscriptionStream(
    SubscribeRequest(topics=[create_topic(sync_topic)])
  )

  print(f"[PromptSource:sync] waiting on topic={sync_topic}")
  n_sent = 0
  for delivery in stream:
    signal = StringValue()
    delivery.message.payload.Unpack(signal)
    prompts = get_prompts(n_sent, batch_size)
    requester.request(prompts=prompts, config=config, streaming=False)
    n_sent += 1
    print(f"[PromptSource:sync] dispatched request {n_sent}/{num_iterations}")
    if num_iterations > 0 and n_sent >= num_iterations:
      break

  channel.close()


def run_async_mode(requester):
  num_batches = int(os.environ.get("NUM_BATCHES", "10"))
  batch_size = int(os.environ.get("BATCH_SIZE", "3"))
  num_rollouts = int(os.environ.get("NUM_ROLLOUTS", "4"))
  dispatch_delay = float(os.environ.get("DISPATCH_DELAY", "0.0"))

  config = InferenceConfig(
    max_tokens=256,
    temperature=0.7,
    top_p=0.95,
    n=num_rollouts,
  )

  batch_idx = 0
  while True:
    if num_batches > 0 and batch_idx >= num_batches:
      break
    prompts = get_prompts(batch_idx, batch_size)
    batch_id = requester.request(prompts=prompts, config=config, streaming=True)
    batch_idx += 1
    print(f"[PromptSource:async] dispatched batch {batch_idx} batch_id={batch_id}")
    if dispatch_delay > 0:
      time.sleep(dispatch_delay)


def main():
  gateway_url = os.environ.get("GATEWAY_URL", "localhost:50051")
  mode = os.environ.get("SC_MODE", "sync").strip().lower()
  if mode not in ("sync", "async"):
    raise ValueError("SC_MODE must be either 'sync' or 'async'.")

  print(f"Starting prompt source: mode={mode} gateway={gateway_url}")
  requester = VLLMRolloutRequester(gateway_url=gateway_url)

  try:
    if mode == "sync":
      run_sync_mode(requester, gateway_url)
    else:
      run_async_mode(requester)
  finally:
    requester.shutdown()
    print("Prompt source exited.")


if __name__ == "__main__":
  main()
