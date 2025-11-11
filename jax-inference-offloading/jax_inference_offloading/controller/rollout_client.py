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
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor

import grpc

import jax_inference_offloading.api.controller_pb2 as ctrl
import jax_inference_offloading.api.controller_pb2_grpc as ctrl_grpc
import jax_inference_offloading.api.message_broker_pb2_grpc as broker_grpc
import jax_inference_offloading.controller.utils as ctrl_utils
from jax_inference_offloading.api.message_broker_pb2 import (
  SubscribeRequest,
)
from jax_inference_offloading.controller.client_base import ClientBase
from jax_inference_offloading.models.auto import get_tp_model_mapping
from jax_inference_offloading.models.mapping_util import add_sharding_specs

logger = logging.getLogger(__name__)


class RolloutServicer:
  def __init__(self, llm):
    llm.collective_rpc("set_sharding")

    self._llm = llm
    self._tok = llm.get_tokenizer()

  @staticmethod
  def as_proto(vllm_response) -> ctrl.InferenceResponse:
    def from_vllm_output(vllm_output) -> ctrl.InferenceResponse.Output:
      output_proto = ctrl.InferenceResponse.Output()
      output_proto.generated_text = vllm_output.text
      output_proto.generated_tokens.ids.extend(vllm_output.token_ids)
      output_proto.generated_token_logps.extend(
        [r[token_id].logprob for (token_id, r) in zip(vllm_output.token_ids, vllm_output.logprobs)]
      )
      return output_proto

    response_proto = ctrl.InferenceResponse()
    for response in vllm_response:
      for output in response.outputs:
        op = from_vllm_output(output)
        op.tokenized_prompt.ids.extend(response.prompt_token_ids)
        response_proto.outputs.append(op)
    return response_proto

  def handshake(self, request):
    mapping_specs = get_tp_model_mapping(request.model_name or self._llm.llm_engine.model_config.model)
    mapping_specs, vllm_tp_size = add_sharding_specs(mapping_specs, self._llm,
                                                     request.jax_parallelism.tp)
    self._mapping_specs = mapping_specs
    return ctrl.HandshakeResponse(
      mapping_specs=mapping_specs,
      jax_parallelism=request.jax_parallelism,
      vllm_parallelism=ctrl.VllmParallelism(tp=vllm_tp_size),
    )

  def create_transport(self, request):
    transport_config = json.loads(request.config_json)
    transport_repr = self._llm.collective_rpc(
      "create_transport",
      timeout=100,
      args=(transport_config,),
    )
    logger.warning(f'vLLM coupling transports created: {transport_repr}')

  def update_weights(self, mode: str):
    if mode in ('fused', 'unfused'):
      self._llm.collective_rpc(
        "update_weights",
        timeout=None,
        args=(self._mapping_specs, ),
      )
    elif mode == 'grouped':
      self._llm.collective_rpc(
        "update_weights_grouped",
        timeout=None,
        args=(self._mapping_specs, ),
      )
    else:
      raise ValueError(f"Unsupported weight update mode: {mode}")
    self._llm.reset_prefix_cache ()  # invalidate cached kvs

  def inference(self, request):
    logger.info("received inference request")
    from vllm import SamplingParams
    config = request.config
    sampling_params = SamplingParams(
      max_tokens=config.max_tokens,
      top_k=config.top_k,
      top_p=config.top_p,
      temperature=config.temperature,
      n=config.num_outputs,
      logprobs=1,
      stop_token_ids=list(config.stop_token_ids) if config.stop_token_ids else None,
    )

    prompts = []
    as_token_prompt = lambda p: dict(prompt_token_ids=p)
    for prompt in request.prompts:
      if prompt.WhichOneof("kind") == "chat_messages_json":
        prompts.append(
          as_token_prompt(
            self._tok.apply_chat_template(
              json.loads(prompt.chat_messages_json),
              tokenize=True,
              add_generation_prompt=True,
              add_special_tokens=True,
            )
          )
        )
      elif prompt.WhichOneof("kind") == "text_prompt":
        prompts.append(as_token_prompt(self._tok.encode(prompt.text_prompt, add_special_tokens=True)))
      elif prompt.WhichOneof("kind") == "tokenized_prompt":
        prompts.append(as_token_prompt(list(prompt.tokenized_prompt.ids)))
      else:
        raise ValueError(f"Unknown prompt type: {prompt.WhichOneof('prompt')}")

    vllm_response = self._llm.generate(prompts, sampling_params)
    return self.as_proto(vllm_response)

  def shutdown(self):
    logger.warning("Shutting down rollout client...")
    self._llm.llm_engine.engine_core.shutdown()
    logger.warning("Rollout client shut down.")

  def shutdown(self):
    logger.warning("Shutting down rollout client...")
    self._llm.llm_engine.engine_core.shutdown()
    logger.warning("Rollout client shut down.")


class RolloutClient(ClientBase):
  def __init__(self, executor, controller_stub, broker_stub, channel=None):
    super().__init__(executor, controller_stub, broker_stub, channel)
    self._update_future = None

  def subscribe_to_control_messages(self, llm):
    assert self._update_future is None

    servicer = RolloutServicer(llm)

    def call():
      try:
        for delivery in self._broker_stub.SubscriptionStream(
          SubscribeRequest(topics=[
            ctrl_utils.HANDSHAKE,
            ctrl_utils.CREATE_TRANSPORT,
            ctrl_utils.WEIGHT_UPDATES,
            ctrl_utils.INFERENCE_REQUESTS,
            ctrl_utils.SHUTDOWN,
          ])
        ):
          if delivery.topic.id == ctrl_utils.HANDSHAKE.id:
            handshake_request = ctrl.HandshakeRequest()
            delivery.message.payload.Unpack(handshake_request)
            self._publish_result(
              handshake_request.response_topic,
              servicer.handshake(handshake_request),
            )
          elif delivery.topic.id == ctrl_utils.CREATE_TRANSPORT.id:
            create_transport_request = ctrl.CreateTransportRequest()
            delivery.message.payload.Unpack(create_transport_request)
            servicer.create_transport(create_transport_request)
          elif delivery.topic.id == ctrl_utils.WEIGHT_UPDATES.id:
            start_weight_update_request = ctrl.StartWeightUpdateRequest()
            delivery.message.payload.Unpack(start_weight_update_request)
            servicer.update_weights(start_weight_update_request.mode)
          elif delivery.topic.id == ctrl_utils.INFERENCE_REQUESTS.id:
            inference_request = ctrl.InferenceRequest()
            delivery.message.payload.Unpack(inference_request)
            self._publish_result(
              inference_request.response_topic,
              servicer.inference(inference_request),
            )
          elif delivery.topic.id == ctrl_utils.SHUTDOWN.id:
            servicer.shutdown()
            break
          else:
            raise Exception(f"Unexpected topic {delivery.topic.id}.")
      except Exception as e:
        # Treat intentional cancellations/unavailable server as graceful closure.
        if isinstance(e, grpc.RpcError) and e.code() in (grpc.StatusCode.CANCELLED, grpc.StatusCode.UNAVAILABLE):
          print('Rollout client stream closed.')
          return
        else:
          print("Failure in rollout subscriber:", e)
          traceback.print_exc()
          raise e

    self._update_future = self._executor.submit(call)


def make_rollout_client(url, executor=None, timeout=60):
  if executor is None:
    executor = ThreadPoolExecutor()
  channel = grpc.insecure_channel(url)
  grpc.channel_ready_future(channel).result(timeout=timeout)
  controller_stub = ctrl_grpc.CouplingControllerStub(channel)
  broker_stub = broker_grpc.MessageBrokerStub(channel)
  return RolloutClient(executor, controller_stub, broker_stub, channel)
