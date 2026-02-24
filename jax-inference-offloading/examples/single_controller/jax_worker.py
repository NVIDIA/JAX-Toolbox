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
"""JAX worker for single-controller example.

This process does not schedule inference. It waits for commands from the
single controller and executes weight transfers on demand.
"""

import os
import traceback

import grpc
import jax
import jax.numpy as jnp
from google.protobuf.wrappers_pb2 import StringValue

import jax_inference_offloading.api.controller_pb2 as ctrl
from jax_inference_offloading import (
  OffloadingSession,
  VLLMTransferEngine,
)
from jax_inference_offloading.api.message_broker_pb2 import (
  PublishRequest,
  SubscribeRequest,
)
from jax_inference_offloading.controller.utils import create_topic
from jax_inference_offloading.models.checkpoint import (
  load_checkpoint_to_jax,
  load_mapping_config,
)


def _publish_event(session, topic_id: str, value: str):
  msg = PublishRequest()
  msg.topic.CopyFrom(create_topic(topic_id))
  msg.message.payload.Pack(StringValue(value=value))
  session.broker_stub.Publish(msg)


def main():
  model_path = os.environ.get("MODEL_PATH")
  param_mapping_path = os.environ.get("PARAM_MAPPING_PATH")
  gateway_url = os.environ.get("GATEWAY_URL", "localhost:50051")
  transfer_mode = os.environ.get("TRANSFER_MODE", "grouped")

  jax_command_topic = os.environ.get("SC_JAX_COMMAND_TOPIC", "sc/jax/commands")
  jax_event_topic = os.environ.get("SC_JAX_EVENT_TOPIC", "sc/jax/events")
  jax_results_topic = os.environ.get(
    "SC_JAX_RESULTS_TOPIC",
    "inference/results/sc_forwarded",
  )

  if model_path is None:
    raise ValueError("MODEL_PATH environment variable is required")
  if param_mapping_path is None:
    raise ValueError("PARAM_MAPPING_PATH environment variable is required")

  if jax.process_index() == 0:
    print("=" * 80)
    print("Starting single-controller JAX worker...")
    print(f"Gateway URL: {gateway_url}")
    print(f"Transfer mode: {transfer_mode}")
    print(f"Results topic: {jax_results_topic}")
    print("=" * 80)

  # Mapping + mesh
  mapping_config = load_mapping_config(param_mapping_path)
  mesh_shape = (jax.process_count(), jax.local_device_count())
  mesh_axes = tuple(mapping_config.mesh_axes)
  if len(mesh_shape) != len(mesh_axes):
    raise ValueError(
      f"Mesh shape {mesh_shape} does not match mesh axes {mesh_axes}."
    )
  mesh = jax.make_mesh(mesh_shape, mesh_axes)

  params = load_checkpoint_to_jax(
    checkpoint_path=model_path,
    mapping_specs=mapping_config.mapping_specs,
    mesh=mesh,
    dtype=jnp.bfloat16,
  )

  session = OffloadingSession(
    gateway_url=gateway_url,
    mesh=mesh,
    model_path=model_path,
    param_mapping_path=param_mapping_path,
  )
  transfer_engine = VLLMTransferEngine(
    session=session,
    transfer_mode=transfer_mode,
  )

  # Prompt source discovers this topic via KV.
  if jax.process_index() == 0:
    topic_value = StringValue(value=jax_results_topic)
    request = ctrl.KVPutRequest(key="inference_response_topic")
    request.value.Pack(topic_value)
    session.controller_stub.KVPut(request)

  stream = session.broker_stub.SubscriptionStream(
    SubscribeRequest(topics=[create_topic(jax_command_topic), create_topic(jax_results_topic)])
  )

  if jax.process_index() == 0:
    _publish_event(session, jax_event_topic, "ready")
    print("JAX worker ready. Waiting for commands...")

  try:
    for delivery in stream:
      topic_id = delivery.topic.id
      payload = delivery.message.payload

      if topic_id == jax_command_topic:
        command = StringValue()
        payload.Unpack(command)
        value = command.value.strip().lower()

        if value == "update_weights":
          if jax.process_index() == 0:
            print("Received command: update_weights")
          transfer_engine.update_weights(params)
          if jax.process_index() == 0:
            _publish_event(session, jax_event_topic, "weights_updated")
            print("Weight update complete.")
        elif value == "shutdown":
          if jax.process_index() == 0:
            print("Received command: shutdown")
          break
        else:
          if jax.process_index() == 0:
            print(f"Unknown command received: {command.value}")

      elif topic_id == jax_results_topic and jax.process_index() == 0:
        if payload.Is(ctrl.RolloutResult.DESCRIPTOR):
          result = ctrl.RolloutResult()
          payload.Unpack(result)
          print(
            f"[Result] batch={result.batch_id[:8]}... "
            f"prompt={result.prompt_index} rollout={result.rollout_index}"
          )
        elif payload.Is(ctrl.InferenceResponse.DESCRIPTOR):
          result = ctrl.InferenceResponse()
          payload.Unpack(result)
          print(f"[Result] received non-streaming response with {len(result.outputs)} outputs")
        else:
          print("[Result] unknown payload type")
  except grpc.RpcError as e:
    if e.code() not in (grpc.StatusCode.CANCELLED, grpc.StatusCode.UNAVAILABLE):
      raise
  except Exception as e:
    if jax.process_index() == 0:
      print(f"Error in JAX worker: {e}")
      traceback.print_exc()
    raise
  finally:
    session.shutdown(shutdown_gateway=False)
    if jax.process_index() == 0:
      print("JAX worker exited.")


if __name__ == "__main__":
  main()
