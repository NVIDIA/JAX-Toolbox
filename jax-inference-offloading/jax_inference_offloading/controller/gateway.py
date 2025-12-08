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
import concurrent.futures as futures
import logging
import os
import time
from copy import deepcopy
from queue import Empty as queue_Empty
from queue import Queue
from threading import Lock

import grpc
from cupy.cuda import nccl
from google.protobuf.empty_pb2 import Empty as protobuf_Empty

import jax_inference_offloading.api.controller_pb2 as ctrl
import jax_inference_offloading.api.controller_pb2_grpc as ctrl_grpc
import jax_inference_offloading.api.message_broker_pb2 as broker
import jax_inference_offloading.api.message_broker_pb2_grpc as broker_grpc
import jax_inference_offloading.controller.utils as ctrl_utils

logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


class MessageQueues:
  def __init__(self):
    self._subscriptions = {}
    self._lock = Lock()
    # TODO: add TIMEOUT for persistent messages
    self._persistent_messages = []

  def _get_subscriptions(self, topic):
    with self._lock:
      if topic.id not in self._subscriptions:
        self._subscriptions[topic.id] = []
      return self._subscriptions[topic.id]

  def subscribe(self, topics):
    subscription = Queue()
    for topic, message in self._persistent_messages:
      if topic in topics:
        subscription.put((topic, message))
    for topic in topics:
      subscriptions = self._get_subscriptions(topic)
      # TODO: lock might not be necessary in Python?
      with self._lock:
        subscriptions.append(subscription)
    return subscription

  def publish(self, topic, message, persistent=False):
    if persistent:
      self._persistent_messages.append((topic, message))
    subscriptions = self._get_subscriptions(topic)
    # TODO: lock might not be necessary in Python?
    with self._lock:
      for subscription in subscriptions:
        subscription.put((topic, message))


class ControllerServicer(ctrl_grpc.CouplingControllerServicer):
  def __init__(self, server, queues):
    self._server = server
    self._queues = queues
    self._kvstore = {}

  def KVPut(self, request, context):
    session_id = dict(context.invocation_metadata()).get('session-id')
    if session_id:
      key = f'{session_id}/{request.key}'
    else:
      key = request.key
    if key in self._kvstore:
      logger.warning(f"Controller KV store overwriting existing key: {key}")
    self._kvstore[key] = deepcopy(request.value)
    return ctrl.KVPutResponse(success=True)

  def KVGet(self, request, context):
    session_id = dict(context.invocation_metadata()).get('session-id')
    if session_id:
      key = f'{session_id}/{request.key}'
    else:
      key = request.key
    if key not in self._kvstore:
      logger.warning(f"Controller KV store key not found: {key}")
      return ctrl.KVGetResponse(found=False)
    return ctrl.KVGetResponse(found=True, value=self._kvstore[key])

  def GetNcclId(self, request, context):
    del context
    return ctrl.GetNcclIdResponse(ids=nccl.get_unique_id())

  def AsyncHandshake(self, request, context):
    del context
    self._queues.publish(ctrl_utils.HANDSHAKE, request, persistent=True)
    return protobuf_Empty()

  def CreateTransport(self, request, context):
    del context
    self._queues.publish(ctrl_utils.CREATE_TRANSPORT, request)
    return ctrl.CreateTransportResponse()

  def StartWeightUpdate(self, request, context):
    del context
    self._queues.publish(ctrl_utils.WEIGHT_UPDATES, request)
    return ctrl.StartWeightUpdateResponse()

  def AsyncInference(self, request, context):
    del context
    self._queues.publish(ctrl_utils.INFERENCE_REQUESTS, request)
    return protobuf_Empty()

  def Shutdown(self, request, context):
    self._queues.publish(ctrl_utils.SHUTDOWN, request)
    def _callback():
      time.sleep(request.grace_period)
      self._server.stop(0)
    context.add_callback(_callback)
    return ctrl.ShutdownResponse()
  
  def StartCUDAProfiler(self, request, context):
    del context
    self._queues.publish(ctrl_utils.START_CUDA_PROFILER, request)
    return ctrl.StartCUDAProfilerResponse()
  
  def StopCUDAProfiler(self, request, context):
    del context
    self._queues.publish(ctrl_utils.STOP_CUDA_PROFILER, request)
    return ctrl.StopCUDAProfilerResponse()

class MessageBrokerServicer(broker_grpc.MessageBrokerServicer):
  def __init__(self, queues):
    self._queues = queues

  def SubscriptionStream(self, request, context):
    queue = self._queues.subscribe(request.topics)
    while context.is_active():
      delivery = broker.SubscribeDelivery()
      try:
        topic, message = queue.get(timeout=1)
      except queue_Empty:
        continue
      delivery.topic.CopyFrom(topic)
      # TODO: Clean this up.
      if message.DESCRIPTOR == broker.Message.DESCRIPTOR:
        delivery.message.CopyFrom(message)
      else:
        delivery.message.payload.Pack(message)
      yield delivery

  def Publish(self, request, context):
    del context
    self._queues.publish(request.topic, request.message)
    return broker.PublishResponse()


def run_gateway():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  queues = MessageQueues()
  controller_port = os.environ.get("GATEWAY_PORT", "50051")  # TODO: move to args
  ctrl_grpc.add_CouplingControllerServicer_to_server(ControllerServicer(server, queues), server)
  broker_grpc.add_MessageBrokerServicer_to_server(MessageBrokerServicer(queues), server)
  server.add_insecure_port(f"[::]:{controller_port}")
  server.start()
  print("gRPC server running.")
  server.wait_for_termination()
  print("gRPC server stopped, exiting.")


if __name__ == "__main__":
  run_gateway()
