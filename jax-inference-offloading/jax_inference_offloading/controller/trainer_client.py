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
import json
import secrets
import traceback
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

import grpc

import jax_inference_offloading.api.controller_pb2 as ctrl
import jax_inference_offloading.api.controller_pb2_grpc as ctrl_grpc
import jax_inference_offloading.api.message_broker_pb2_grpc as broker_grpc
import jax_inference_offloading.controller.utils as ctrl_utils
from jax_inference_offloading.api.message_broker_pb2 import (
  SubscribeRequest,
)
from jax_inference_offloading.controller.client_base import ClientBase
from jax_inference_offloading.controller.spmd import on_spmd_leader
from jax_inference_offloading.transport.tensor import get_transport_class


def async_reply_stream(broker_stub, response_topic_id, response_schema, blocking=True, executor=None):
  result_queue = Queue()

  stream = broker_stub.SubscriptionStream(
      SubscribeRequest(topics=[ctrl_utils.create_topic(response_topic_id)])
  )

  def handle_delivery():
    try:
      for delivery in stream:
        result = response_schema()
        assert delivery.message.payload.Unpack(result)
        result_queue.put(result)
    except grpc.RpcError as e:
      # Treat intentional cancellations/unavailable server as graceful closure.
      if e.code() in (grpc.StatusCode.CANCELLED, grpc.StatusCode.UNAVAILABLE):
        return
      else:
        print("Failure in subscriber:", e)
        traceback.print_exc()
        raise e
    except Exception as e:
      print("Failure in subscriber:", e)
      traceback.print_exc()
      raise e
    return

  executor = executor or ThreadPoolExecutor(max_workers=1)
  future = executor.submit(handle_delivery)

  if blocking:
    while True:
      yield result_queue.get()
  else:
    while True:
      try:
        result = result_queue.get(timeout=1)
        yield result
      except Empty:
        if future.done():
          return
        yield None


class TrainerClient(ClientBase):
  def __init__(self, executor, controller_stub, broker_stub, channel=None):
    super().__init__(executor, controller_stub, broker_stub, channel)
    self._update_future = None
    self._inference_result_topic_id = f"{ctrl_utils.INFERENCE_REQUESTS}/results/{secrets.token_hex(16)}"
    self._inference_response_stream = async_reply_stream(
      broker_stub,
      self._inference_result_topic_id,
      ctrl.InferenceResponse,
      blocking=True
    )
    self._handshake_topic_id = f"{ctrl_utils.HANDSHAKE.id}/results/{secrets.token_hex(16)}"
    self._handshake_response_stream = async_reply_stream(
      broker_stub,
      self._handshake_topic_id,
      ctrl.HandshakeResponse,
      blocking=True
    )

  @on_spmd_leader(
    serializer=lambda m: m.SerializeToString(),
    deserializer=lambda b: (lambda r: (r.ParseFromString(b), r)[1])(ctrl.HandshakeResponse()),
  )
  def handshake(self, jax_n_devices: int, model_name: str = ""):
    """Returns the mapping between JAX and vLLM models. Executed once and broadcast."""
    self._controller_stub.AsyncHandshake(
      ctrl.HandshakeRequest(
        response_topic=self._handshake_topic_id,
        model_name=model_name,
        jax_parallelism=ctrl.JaxParallelism(tp=jax_n_devices),
      )
    )
    result = next(self._handshake_response_stream)

    # validate TP sizes (runs on all ranks after broadcast)
    is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
    assert is_power_of_2(result.vllm_parallelism.tp), "vLLM TP size must be a power of 2."
    assert is_power_of_2(result.jax_parallelism.tp), "JAX TP size must be a power of 2."

    return result

  def create_transport(self, backend, **backend_args):
    """Creates a transport for JAX and vLLM coupling. Configured once and broadcast."""
    transport_cls = get_transport_class(backend)

    def _call():
      cfg = transport_cls.configure(
        self,
        **backend_args
      )
      self._controller_stub.CreateTransport(
        ctrl.CreateTransportRequest(
          config_json=json.dumps(cfg),
        )
      )
      return cfg

    transport_config = on_spmd_leader(_call)()

    transports = transport_cls.create_trainer_transport(
      transport_config
    )
    return transports, transport_config

  @on_spmd_leader(broadcast_result=False)
  def start_weight_transfer(self, mode: str):
    self._controller_stub.StartWeightUpdate(
      ctrl.StartWeightUpdateRequest(mode=mode)
    )

  @on_spmd_leader(broadcast_result=False)
  def start_cuda_profiler(self):
    self._controller_stub.StartCUDAProfiler(ctrl.StartCUDAProfilerRequest())

  @on_spmd_leader(broadcast_result=False)
  def stop_cuda_profiler(self):
    self._controller_stub.StopCUDAProfiler(ctrl.StopCUDAProfilerRequest())

  def inference(self, prompts, config=None):
    """Prompts can be one of:
    - str: a single text prompt
        e.g. "Quick facts about the moon:"
    - List[str]: a list of text prompts
        e.g. ["Quick facts about the moon:", "What is the capital of France?"]
    - List[dict]: a single chat message thread, where each message is a dict with "role" and "content" keys
        e.g. [{"role": "user", "content": "Quick facts about the moon"}]
    - List[List[dict]]: a list of chat message threads, where each thread is a list of messages
        e.g. [
            [{"role": "user", "content": "Quick facts about the moon"}],
            [{"role": "user", "content": "What is the capital of France?"}]
        ]
    - List[int]: a single prompt encoded in token IDs
        e.g. [101, 1234, 5678, 102]
    - List[List[int]]: a list of tokenized prompts
        e.g. [[101, 1234, 5678, 102], [101, 2345, 6789, 102]]
    """
    # assemble request
    request = ctrl.InferenceRequest()

    if isinstance(prompts, str):
      request.prompts.append(ctrl.Prompt(text_prompt=prompts))
    elif isinstance(prompts, list) and all(isinstance(p, int) for p in prompts):
      tids = ctrl.TokenIds()
      tids.ids.extend(prompts)
      request.prompts.append(ctrl.Prompt(tokenized_prompt=tids))
    elif isinstance(prompts, list) and all(isinstance(p, str) for p in prompts):
      for p in prompts:
        request.prompts.append(ctrl.Prompt(text_prompt=p))
    elif isinstance(prompts, list) and all(isinstance(p, dict) for p in prompts):
      request.prompts.append(ctrl.Prompt(chat_messages_json=json.dumps(prompts)))
    elif isinstance(prompts, list) and all(isinstance(p, list) for p in prompts):
      for p in prompts:
        if all(isinstance(m, dict) for m in p):
          request.prompts.append(ctrl.Prompt(chat_messages_json=json.dumps(p)))
        elif all(isinstance(m, int) for m in p):
          tids = ctrl.TokenIds()
          tids.ids.extend(p)
          request.prompts.append(ctrl.Prompt(tokenized_prompt=tids))
        else:
          raise ValueError(f"Invalid prompt format. Expected a list of dicts or a list of ints.Got {p}.")
    else:
      raise ValueError(f"Invalid prompt format {prompts}")

    if config:
      request.config.CopyFrom(config)
    request.response_topic = self._inference_result_topic_id

    # fire request and wait for response
    self._controller_stub.AsyncInference(request)
    # TODO: Maybe expose the asynchronous API?
    return next(self._inference_response_stream)


def make_trainer_client(url, executor, timeout=60):
  channel = grpc.insecure_channel(url)
  grpc.channel_ready_future(channel).result(timeout=timeout)
  controller_stub = ctrl_grpc.CouplingControllerStub(channel)
  broker_stub = broker_grpc.MessageBrokerStub(channel)
  return TrainerClient(executor, controller_stub, broker_stub, channel)
