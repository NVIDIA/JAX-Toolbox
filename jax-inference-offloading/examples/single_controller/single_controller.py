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
"""Single controller runtime for JAX-vLLM offloading.

This process is the only control-plane coordinator in this example:
- Bridges default ingress control topics to single-controller vLLM topics.
- Owns sync/async scheduling policy.
- Receives rollout results from vLLM and forwards them to the JAX worker.
"""

from collections import defaultdict, deque
import logging
import os

import grpc
from google.protobuf.wrappers_pb2 import StringValue

import jax_inference_offloading.api.controller_pb2 as ctrl
import jax_inference_offloading.api.controller_pb2_grpc as ctrl_grpc
import jax_inference_offloading.api.message_broker_pb2_grpc as broker_grpc
from jax_inference_offloading.api.message_broker_pb2 import (
  PublishRequest,
  SubscribeRequest,
)
from jax_inference_offloading.controller import utils as ctrl_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SingleController:
  def __init__(self):
    self._gateway_url = os.environ.get("GATEWAY_URL", "localhost:50051")
    self._mode = os.environ.get("SC_MODE", "sync").strip().lower()
    if self._mode not in ("sync", "async"):
      raise ValueError("SC_MODE must be either 'sync' or 'async'.")

    # Sync-mode policy
    self._num_iterations = int(os.environ.get("NUM_ITERATIONS", "3"))

    # Async-mode policy
    self._update_interval = int(os.environ.get("UPDATE_INTERVAL", "5"))
    # 0 means unlimited in-flight prompts.
    self._max_inflight_prompts = int(os.environ.get("MAX_STALENESS", "0"))
    # 0 means run until externally stopped.
    self._max_completed_prompts = int(os.environ.get("MAX_COMPLETED_PROMPTS", "0"))

    # Internal wiring topics
    self._jax_commands_topic = ctrl_utils.create_topic(
      os.environ.get("SC_JAX_COMMAND_TOPIC", "sc/jax/commands")
    )
    self._jax_events_topic = ctrl_utils.create_topic(
      os.environ.get("SC_JAX_EVENT_TOPIC", "sc/jax/events")
    )
    self._jax_results_topic = ctrl_utils.create_topic(
      os.environ.get("SC_JAX_RESULTS_TOPIC", "inference/results/sc_forwarded")
    )
    self._sync_signal_topic = ctrl_utils.create_topic(
      os.environ.get("SC_SYNC_TOPIC", "sync/weights_ready")
    )
    self._vllm_events_topic = ctrl_utils.create_topic(
      os.environ.get("SC_VLLM_EVENT_TOPIC", ctrl_utils.SC_VLLM_EVENTS.id)
    )

    self._channel = grpc.insecure_channel(self._gateway_url)
    grpc.channel_ready_future(self._channel).result(timeout=60)
    self._controller_stub = ctrl_grpc.CouplingControllerStub(self._channel)
    self._broker_stub = broker_grpc.MessageBrokerStub(self._channel)

    # Shared result ingress from SC vLLM workers.
    self._controller_results_topic = ctrl_utils.SC_RESULTS

    # State used for mediation/routing.
    self._pending_handshake_response_topics = deque()
    self._deferred_handshakes = deque()
    self._pending_non_streaming_meta = deque()
    self._streaming_batch_meta = {}
    self._streaming_prompt_counts = defaultdict(lambda: defaultdict(int))
    self._streaming_prompts_completed = defaultdict(int)

    # Scheduling state.
    self._jax_ready = False
    self._vllm_ready = False
    self._update_in_flight = False
    self._weights_ready = False
    self._sync_inference_in_flight = False
    self._sync_completed_iterations = 0
    self._inflight_prompts = 0
    self._prompts_completed_since_update = 0
    self._total_completed_prompts = 0
    self._inference_queue = deque()
    self._stop_requested = False

  def _publish(self, topic, payload):
    msg = PublishRequest()
    msg.topic.CopyFrom(topic)
    msg.message.payload.Pack(payload)
    self._broker_stub.Publish(msg)

  def _send_jax_command(self, command: str):
    logger.info("Sending JAX command: %s", command)
    self._publish(self._jax_commands_topic, StringValue(value=command))

  def _republish_to_sc_topic(self, sc_topic, payload):
    self._publish(sc_topic, payload)

  def _bridge_handshake(self, request: ctrl.HandshakeRequest):
    forwarded = ctrl.HandshakeRequest()
    forwarded.CopyFrom(request)
    forwarded.response_topic = self._controller_results_topic.id
    if not self._vllm_ready:
      self._deferred_handshakes.append((request.response_topic, forwarded))
      logger.info("Deferring handshake until vLLM worker signals readiness.")
      return
    self._pending_handshake_response_topics.append(request.response_topic)
    self._republish_to_sc_topic(ctrl_utils.SC_HANDSHAKE, forwarded)

  def _flush_deferred_handshakes(self):
    while self._deferred_handshakes:
      response_topic, forwarded = self._deferred_handshakes.popleft()
      self._pending_handshake_response_topics.append(response_topic)
      self._republish_to_sc_topic(ctrl_utils.SC_HANDSHAKE, forwarded)

  def _handle_vllm_event(self, event: StringValue):
    value = event.value.strip().lower()
    if value == "ready":
      if not self._vllm_ready:
        self._vllm_ready = True
        logger.info("Received vLLM event: ready")
        self._flush_deferred_handshakes()

  def _bridge_create_transport(self, request: ctrl.CreateTransportRequest):
    self._republish_to_sc_topic(ctrl_utils.SC_CREATE_TRANSPORT, request)

  def _bridge_weight_update(self, request: ctrl.StartWeightUpdateRequest):
    self._republish_to_sc_topic(ctrl_utils.SC_WEIGHT_UPDATES, request)

  def _bridge_start_profiler(self, request: ctrl.StartCUDAProfilerRequest):
    self._republish_to_sc_topic(ctrl_utils.SC_START_CUDA_PROFILER, request)

  def _bridge_stop_profiler(self, request: ctrl.StopCUDAProfilerRequest):
    self._republish_to_sc_topic(ctrl_utils.SC_STOP_CUDA_PROFILER, request)

  def _forward_inference_to_vllm(self, request: ctrl.InferenceRequest):
    forwarded = ctrl.InferenceRequest()
    forwarded.CopyFrom(request)
    forwarded.response_topic = self._controller_results_topic.id

    if forwarded.streaming:
      if not forwarded.batch_id:
        raise ValueError("Streaming requests require a non-empty batch_id.")
      self._streaming_batch_meta[forwarded.batch_id] = dict(
        num_outputs=max(1, int(forwarded.config.num_outputs)),
        num_prompts=len(forwarded.prompts),
      )
    else:
      self._pending_non_streaming_meta.append(
        dict(num_prompts=len(forwarded.prompts))
      )

    self._inflight_prompts += len(forwarded.prompts)
    self._republish_to_sc_topic(ctrl_utils.SC_INFERENCE_REQUESTS, forwarded)

  def _queue_or_forward_async_inference(self, request: ctrl.InferenceRequest):
    n_prompts = len(request.prompts)
    if self._max_inflight_prompts > 0 and (
      self._inflight_prompts + n_prompts > self._max_inflight_prompts
    ):
      self._inference_queue.append(request)
      logger.info(
        "Queued async inference request: prompts=%d, inflight=%d/%d, queue=%d",
        n_prompts,
        self._inflight_prompts,
        self._max_inflight_prompts,
        len(self._inference_queue),
      )
      return

    self._forward_inference_to_vllm(request)

  def _drain_async_queue(self):
    if not self._weights_ready:
      return

    if self._max_inflight_prompts <= 0:
      while self._inference_queue:
        req = self._inference_queue.popleft()
        self._forward_inference_to_vllm(req)
      return

    while self._inference_queue:
      next_req = self._inference_queue[0]
      n_prompts = len(next_req.prompts)
      if self._inflight_prompts + n_prompts > self._max_inflight_prompts:
        return
      self._inference_queue.popleft()
      self._forward_inference_to_vllm(next_req)

  def _maybe_start_update(self):
    if not self._jax_ready or self._update_in_flight:
      return
    self._update_in_flight = True
    self._weights_ready = False
    self._send_jax_command("update_weights")

  def _maybe_run_sync_scheduler(self):
    if self._mode != "sync":
      return

    # Stop issuing new work once the configured iteration count is reached.
    if self._num_iterations > 0 and self._sync_completed_iterations >= self._num_iterations:
      return

    if not self._jax_ready:
      return

    if not self._update_in_flight and not self._weights_ready and not self._sync_inference_in_flight:
      self._maybe_start_update()
      return

    if self._weights_ready and not self._sync_inference_in_flight and self._inference_queue:
      req = self._inference_queue.popleft()
      self._forward_inference_to_vllm(req)
      self._weights_ready = False
      self._sync_inference_in_flight = True

  def _initiate_shutdown(self, reason: str):
    if self._stop_requested:
      return
    self._stop_requested = True
    logger.info("Initiating coordinated shutdown: %s", reason)

    try:
      self._send_jax_command("shutdown")
    except Exception:
      logger.exception("Failed sending JAX shutdown command")

    try:
      self._republish_to_sc_topic(
        ctrl_utils.SC_SHUTDOWN,
        ctrl.ShutdownRequest(grace_period=2),
      )
    except Exception:
      logger.exception("Failed forwarding SC shutdown request")

    try:
      self._controller_stub.Shutdown(ctrl.ShutdownRequest(grace_period=2))
    except Exception:
      logger.exception("Failed requesting gateway shutdown")

  def _maybe_run_async_scheduler(self):
    if self._mode != "async":
      return

    if not self._jax_ready:
      return

    if self._prompts_completed_since_update >= self._update_interval and not self._update_in_flight:
      self._maybe_start_update()

    self._drain_async_queue()

  def _handle_jax_event(self, event: StringValue):
    value = event.value.strip().lower()
    if value == "ready":
      self._jax_ready = True
      logger.info("Received JAX event: ready")
      # Kick off the first update cycle.
      self._maybe_start_update()
      return

    if value == "weights_updated":
      self._update_in_flight = False
      self._weights_ready = True
      self._prompts_completed_since_update = 0
      logger.info("Received JAX event: weights_updated")

      if self._mode == "sync":
        self._publish(
          self._sync_signal_topic,
          StringValue(value=str(self._sync_completed_iterations)),
        )

      self._maybe_run_async_scheduler()
      self._maybe_run_sync_scheduler()

  def _handle_handshake_response(self, response: ctrl.HandshakeResponse):
    if not self._pending_handshake_response_topics:
      logger.warning("Dropping unexpected HandshakeResponse with no pending request.")
      return
    dst_topic = self._pending_handshake_response_topics.popleft()
    self._publish(ctrl_utils.create_topic(dst_topic), response)

  def _mark_prompt_completed(self):
    if self._inflight_prompts > 0:
      self._inflight_prompts -= 1
    self._prompts_completed_since_update += 1
    self._total_completed_prompts += 1

    if (
      self._mode == "async"
      and self._max_completed_prompts > 0
      and self._total_completed_prompts >= self._max_completed_prompts
    ):
      self._initiate_shutdown(
        f"reached max completed prompts: {self._total_completed_prompts}"
      )

  def _handle_rollout_result(self, result: ctrl.RolloutResult):
    self._publish(self._jax_results_topic, result)

    meta = self._streaming_batch_meta.get(result.batch_id)
    if not meta:
      return

    num_outputs = max(1, int(meta["num_outputs"]))
    prompt_idx = int(result.prompt_index)
    self._streaming_prompt_counts[result.batch_id][prompt_idx] += 1

    if self._streaming_prompt_counts[result.batch_id][prompt_idx] == num_outputs:
      self._mark_prompt_completed()
      self._streaming_prompts_completed[result.batch_id] += 1

      if self._streaming_prompts_completed[result.batch_id] >= int(meta["num_prompts"]):
        self._streaming_batch_meta.pop(result.batch_id, None)
        self._streaming_prompt_counts.pop(result.batch_id, None)
        self._streaming_prompts_completed.pop(result.batch_id, None)

    self._maybe_run_async_scheduler()

  def _handle_inference_response(self, response: ctrl.InferenceResponse):
    self._publish(self._jax_results_topic, response)

    if self._pending_non_streaming_meta:
      meta = self._pending_non_streaming_meta.popleft()
      for _ in range(int(meta["num_prompts"])):
        self._mark_prompt_completed()

    if self._mode == "sync":
      self._sync_inference_in_flight = False
      self._sync_completed_iterations += 1
      logger.info(
        "Completed sync iteration %d/%s",
        self._sync_completed_iterations,
        self._num_iterations if self._num_iterations > 0 else "inf",
      )
      if self._num_iterations > 0 and self._sync_completed_iterations >= self._num_iterations:
        self._initiate_shutdown(
          f"reached sync iteration target: {self._sync_completed_iterations}"
        )
      else:
        self._maybe_run_sync_scheduler()
    else:
      self._maybe_run_async_scheduler()

  def run(self):
    logger.info(
      "Single controller started: gateway=%s mode=%s update_interval=%d max_inflight=%d",
      self._gateway_url,
      self._mode,
      self._update_interval,
      self._max_inflight_prompts,
    )

    stream = self._broker_stub.SubscriptionStream(
      SubscribeRequest(
        topics=[
          ctrl_utils.HANDSHAKE,
          ctrl_utils.CREATE_TRANSPORT,
          ctrl_utils.WEIGHT_UPDATES,
          ctrl_utils.INFERENCE_REQUESTS,
          ctrl_utils.START_CUDA_PROFILER,
          ctrl_utils.STOP_CUDA_PROFILER,
          ctrl_utils.SHUTDOWN,
          self._controller_results_topic,
          self._jax_events_topic,
          self._vllm_events_topic,
        ]
      )
    )

    for delivery in stream:
      topic_id = delivery.topic.id
      payload = delivery.message.payload

      if topic_id == ctrl_utils.HANDSHAKE.id:
        request = ctrl.HandshakeRequest()
        payload.Unpack(request)
        self._bridge_handshake(request)
      elif topic_id == ctrl_utils.CREATE_TRANSPORT.id:
        request = ctrl.CreateTransportRequest()
        payload.Unpack(request)
        self._bridge_create_transport(request)
      elif topic_id == ctrl_utils.WEIGHT_UPDATES.id:
        request = ctrl.StartWeightUpdateRequest()
        payload.Unpack(request)
        self._bridge_weight_update(request)
      elif topic_id == ctrl_utils.START_CUDA_PROFILER.id:
        request = ctrl.StartCUDAProfilerRequest()
        payload.Unpack(request)
        self._bridge_start_profiler(request)
      elif topic_id == ctrl_utils.STOP_CUDA_PROFILER.id:
        request = ctrl.StopCUDAProfilerRequest()
        payload.Unpack(request)
        self._bridge_stop_profiler(request)
      elif topic_id == ctrl_utils.INFERENCE_REQUESTS.id:
        request = ctrl.InferenceRequest()
        payload.Unpack(request)
        self._inference_queue.append(request)
        if self._mode == "sync":
          self._maybe_run_sync_scheduler()
        else:
          self._drain_async_queue()
      elif topic_id == self._controller_results_topic.id:
        if payload.Is(ctrl.HandshakeResponse.DESCRIPTOR):
          result = ctrl.HandshakeResponse()
          payload.Unpack(result)
          self._handle_handshake_response(result)
        elif payload.Is(ctrl.RolloutResult.DESCRIPTOR):
          result = ctrl.RolloutResult()
          payload.Unpack(result)
          self._handle_rollout_result(result)
        elif payload.Is(ctrl.InferenceResponse.DESCRIPTOR):
          result = ctrl.InferenceResponse()
          payload.Unpack(result)
          self._handle_inference_response(result)
        else:
          logger.warning("Unknown message type on controller results topic.")
      elif topic_id == self._jax_events_topic.id:
        event = StringValue()
        payload.Unpack(event)
        self._handle_jax_event(event)
      elif topic_id == self._vllm_events_topic.id:
        event = StringValue()
        payload.Unpack(event)
        self._handle_vllm_event(event)
      elif topic_id == ctrl_utils.SHUTDOWN.id:
        request = ctrl.ShutdownRequest()
        payload.Unpack(request)
        self._republish_to_sc_topic(ctrl_utils.SC_SHUTDOWN, request)
        self._send_jax_command("shutdown")
        logger.info("Received shutdown signal. Exiting single controller loop.")
        break
      else:
        logger.warning("Ignoring message on unexpected topic: %s", topic_id)

      if self._stop_requested:
        break

    try:
      self._channel.close()
    except Exception:
      pass


def main():
  SingleController().run()


if __name__ == "__main__":
  main()
