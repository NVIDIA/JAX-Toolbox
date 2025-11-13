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
import jax_inference_offloading.api.controller_pb2 as ctrl
from jax_inference_offloading.api.message_broker_pb2 import (
    PublishRequest,
)


class ClientBase:
  def __init__(self, executor, controller_stub, broker_stub, channel=None):
    self._controller_stub = controller_stub
    self._broker_stub = broker_stub
    self._executor = executor
    self._channel = channel

  def kv_put(self, key, value, session_id=None):
    request = ctrl.KVPutRequest(key=key)
    request.value.Pack(value)
    if session_id:
      metadata = (('session-id', session_id),)
    else:
      metadata = ()
    return self._controller_stub.KVPut(request, metadata=metadata).success

  def kv_get(self, key, session_id=None, schema=None):
    if session_id:
      metadata = (('session-id', session_id),)
    else:
      metadata = ()
    response = self._controller_stub.KVGet(ctrl.KVGetRequest(key=key), metadata=metadata)
    if schema is not None:
      result = schema()
      assert response.value.Unpack(result)
      return result
    else:
      return response.value

  def get_nccl_id(self):
    return tuple(self._controller_stub.GetNcclId(ctrl.GetNcclIdRequest()).ids)

  def _publish_result(self, topic, result):
    msg = PublishRequest()
    msg.topic.id = topic
    msg.message.payload.Pack(result)
    self._broker_stub.Publish(msg)

  def shutdown(self, grace_period=10):
    self._controller_stub.Shutdown(ctrl.ShutdownRequest(grace_period=grace_period))
    try:
      self._channel.close()
    except Exception:
      # Ignore close errors; channel may already be closed.
      pass
