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
from jax_inference_offloading.models.mapping_util import add_sharding_specs, load_mapping_from_json

logger = logging.getLogger(__name__)


class RolloutServicer:
  def __init__(self, llm, mapping_json_path=None):
    llm.collective_rpc("set_sharding")

    self._llm = llm
    self._tok = llm.get_tokenizer()
    self._mapping_json_path = mapping_json_path

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
    # Use param_mapping_path from request if provided, otherwise fall back to local config
    mapping_path = getattr(request, 'param_mapping_path', None) or self._mapping_json_path
    if not mapping_path:
      raise ValueError("param_mapping_path is required but not provided")
    mapping_specs = load_mapping_from_json(mapping_path)
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

  def start_cuda_profiler(self):
    self._llm.collective_rpc(
      "start_cuda_profiler",
      timeout=None,
      args=(),
    )

  def stop_cuda_profiler(self):
    self._llm.collective_rpc(
      "stop_cuda_profiler",
      timeout=None,
      args=(),
    )

  def update_weights(self, mode: str):
    """Receive weight update via NCCL.
    
    Args:
        mode: Transfer mode ('fused', 'unfused', or 'grouped').
    """
    logger.info(f"Receiving weight update via NCCL (mode={mode})...")
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
    self._llm.reset_prefix_cache()  # invalidate cached kvs
    logger.info("Weight update complete.")

  def _parse_prompts(self, request):
    """Parse prompts from request into vLLM format.
    
    Returns:
        List of (prompt_dict, prompt_token_ids) tuples.
    """
    prompts = []
    as_token_prompt = lambda p: dict(prompt_token_ids=p)
    for prompt in request.prompts:
      if prompt.WhichOneof("kind") == "chat_messages_json":
        token_ids = self._tok.apply_chat_template(
          json.loads(prompt.chat_messages_json),
          tokenize=True,
          add_generation_prompt=True,
          add_special_tokens=True,
        )
        prompts.append((as_token_prompt(token_ids), token_ids))
      elif prompt.WhichOneof("kind") == "text_prompt":
        token_ids = self._tok.encode(prompt.text_prompt, add_special_tokens=True)
        prompts.append((as_token_prompt(token_ids), token_ids))
      elif prompt.WhichOneof("kind") == "tokenized_prompt":
        token_ids = list(prompt.tokenized_prompt.ids)
        prompts.append((as_token_prompt(token_ids), token_ids))
      else:
        raise ValueError(f"Unknown prompt type: {prompt.WhichOneof('prompt')}")
    return prompts

  def inference(self, request):
    """Non-streaming inference: returns all results as a single InferenceResponse."""
    logger.info("received inference request (non-streaming)")
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

    parsed_prompts = self._parse_prompts(request)
    prompts = [p[0] for p in parsed_prompts]

    vllm_response = self._llm.generate(prompts, sampling_params)
    return self.as_proto(vllm_response)

  def inference_streaming(self, request, start_idx=0, end_idx=None, prompt_index_offset=0):
    """Streaming inference: yields RolloutResult messages as each rollout completes.
    
    Supports partial batch processing for bounded staleness control.
    
    Args:
        request: InferenceRequest protobuf message.
        start_idx: Start index into request.prompts (inclusive). Default 0.
        end_idx: End index into request.prompts (exclusive). Default None (all).
        prompt_index_offset: Offset to add to prompt_index in results. Used when
            processing a partial batch to maintain correct indices for JAX controller.
    
    Yields:
        ctrl.RolloutResult messages, one per completed rollout.
    """
    from vllm import SamplingParams
    
    # Determine which prompts to process
    all_prompts = list(request.prompts)
    if end_idx is None:
      end_idx = len(all_prompts)
    prompts_to_process = all_prompts[start_idx:end_idx]
    num_prompts = len(prompts_to_process)
    
    if num_prompts == 0:
      return
    
    logger.info(
      f"received inference request (streaming), batch_id={request.batch_id}, "
      f"prompts[{start_idx}:{end_idx}] (offset={prompt_index_offset})"
    )
    
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

    # Parse only the prompts we're processing
    parsed_prompts = []
    as_token_prompt = lambda p: dict(prompt_token_ids=p)
    for prompt in prompts_to_process:
      if prompt.WhichOneof("kind") == "chat_messages_json":
        token_ids = self._tok.apply_chat_template(
          json.loads(prompt.chat_messages_json),
          tokenize=True,
          add_generation_prompt=True,
          add_special_tokens=True,
        )
        parsed_prompts.append((as_token_prompt(token_ids), token_ids))
      elif prompt.WhichOneof("kind") == "text_prompt":
        token_ids = self._tok.encode(prompt.text_prompt, add_special_tokens=True)
        parsed_prompts.append((as_token_prompt(token_ids), token_ids))
      elif prompt.WhichOneof("kind") == "tokenized_prompt":
        token_ids = list(prompt.tokenized_prompt.ids)
        parsed_prompts.append((as_token_prompt(token_ids), token_ids))
      else:
        raise ValueError(f"Unknown prompt type: {prompt.WhichOneof('prompt')}")

    prompts = [p[0] for p in parsed_prompts]
    prompt_token_ids_list = [p[1] for p in parsed_prompts]

    # Generate all outputs (blocking)
    vllm_response = self._llm.generate(prompts, sampling_params)

    # Yield individual RolloutResult messages
    for local_idx, response in enumerate(vllm_response):
      # Use offset to maintain correct prompt_index for JAX controller
      actual_prompt_idx = prompt_index_offset + local_idx
      prompt_token_ids = prompt_token_ids_list[local_idx]
      for rollout_idx, output in enumerate(response.outputs):
        result = ctrl.RolloutResult()
        result.batch_id = request.batch_id
        result.prompt_index = actual_prompt_idx
        result.rollout_index = rollout_idx
        result.prompt_tokens.ids.extend(prompt_token_ids)
        result.generated_text = output.text
        result.generated_tokens.ids.extend(output.token_ids)
        result.generated_token_logps.extend(
          [r[token_id].logprob for (token_id, r) in zip(output.token_ids, output.logprobs)]
        )
        yield result

  def shutdown(self):
    logger.warning("Shutting down rollout client...")
    self._llm.llm_engine.engine_core.shutdown()
    logger.warning("Rollout client shut down.")

class RolloutClient(ClientBase):
  def __init__(self, executor, controller_stub, broker_stub, channel=None):
    super().__init__(executor, controller_stub, broker_stub, channel)
    self._update_future = None

  def subscribe_to_control_messages(self, llm, mapping_json_path=None):
    assert self._update_future is None

    servicer = RolloutServicer(llm, mapping_json_path=mapping_json_path)

    def call():
      try:
        for delivery in self._broker_stub.SubscriptionStream(
          SubscribeRequest(topics=[
            ctrl_utils.HANDSHAKE,
            ctrl_utils.CREATE_TRANSPORT,
            ctrl_utils.WEIGHT_UPDATES,
            ctrl_utils.INFERENCE_REQUESTS,
            ctrl_utils.SHUTDOWN,
            ctrl_utils.START_CUDA_PROFILER,
            ctrl_utils.STOP_CUDA_PROFILER,
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
            if inference_request.streaming:
              # Streaming mode: publish each RolloutResult as it completes
              for rollout_result in servicer.inference_streaming(inference_request):
                self._publish_result(
                  inference_request.response_topic,
                  rollout_result,
                )
            else:
              # Non-streaming mode: publish all results as a single InferenceResponse
              self._publish_result(
                inference_request.response_topic,
                servicer.inference(inference_request),
              )
          elif delivery.topic.id == ctrl_utils.START_CUDA_PROFILER.id:
            servicer.start_cuda_profiler()
          elif delivery.topic.id == ctrl_utils.STOP_CUDA_PROFILER.id:
            servicer.stop_cuda_profiler()
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


class AsyncRolloutClient(ClientBase):
  """Client that handles async inference with streaming results and bounded staleness.
  
  This client extends the basic RolloutClient with staleness control for
  asynchronous RL post-training workflows. Key features:
  
  - Tracks prompts processed since last weight update
  - Queues inference requests when staleness limit is reached
  - Supports partial batch processing to hit exact staleness limits
  - Drains pending queue after each weight update
  """

  def __init__(self, executor, controller_stub, broker_stub, channel=None, max_staleness=0):
    """Initialize the async rollout client.
    
    Args:
        executor: ThreadPoolExecutor for background tasks.
        controller_stub: gRPC stub for controller service.
        broker_stub: gRPC stub for message broker service.
        channel: gRPC channel (optional, for cleanup).
        max_staleness: Maximum prompts to process before requiring a weight update.
            0 means no limit (unlimited staleness).
    """
    super().__init__(executor, controller_stub, broker_stub, channel)
    self._update_future = None
    self._max_staleness = max_staleness

  def _process_inference_request(self, servicer, inference_request, start_idx=0, end_idx=None, prompt_index_offset=0):
    """Process an inference request (or partial batch) and stream results back.
    
    Args:
        servicer: RolloutServicer instance.
        inference_request: The inference request to process.
        start_idx: Start index into prompts (inclusive). Default 0.
        end_idx: End index into prompts (exclusive). Default None (all).
        prompt_index_offset: Offset for prompt_index in results.
        
    Returns:
        Number of prompts processed.
    """
    if end_idx is None:
      end_idx = len(inference_request.prompts)
    num_prompts = end_idx - start_idx
    
    for rollout_result in servicer.inference_streaming(
        inference_request, 
        start_idx=start_idx, 
        end_idx=end_idx, 
        prompt_index_offset=prompt_index_offset
    ):
      self._publish_result(
        inference_request.response_topic,
        rollout_result,
      )
    return num_prompts

  def subscribe_to_control_messages(self, llm, mapping_json_path=None):
    """Subscribe to control messages with streaming inference results and staleness control."""
    assert self._update_future is None

    servicer = RolloutServicer(llm, mapping_json_path=mapping_json_path)
    max_staleness = self._max_staleness

    def process_prompts(inference_request, start_idx, prompts_since_update):
      """Process prompts from a request, respecting staleness limit.
      
      Returns:
          Tuple of (prompts_processed, remaining_start_idx or None if complete).
      """
      total_prompts = len(inference_request.prompts)
      remaining = total_prompts - start_idx
      
      if max_staleness == 0:
        # No limit - process all remaining
        can_process = remaining
      else:
        # Calculate how many we can process before hitting limit
        capacity = max_staleness - prompts_since_update
        can_process = min(remaining, capacity)
      
      if can_process <= 0:
        # Can't process any - return unchanged
        return 0, start_idx
      
      # Process the prompts we can
      self._process_inference_request(
        servicer, 
        inference_request, 
        start_idx=start_idx,
        end_idx=start_idx + can_process,
        prompt_index_offset=start_idx,
      )
      
      new_start_idx = start_idx + can_process
      if new_start_idx >= total_prompts:
        # Completed this request
        return can_process, None
      else:
        # More prompts remaining
        return can_process, new_start_idx

    def call():
      # Staleness tracking
      prompts_since_weight_update = 0
      # Queue stores tuples: (inference_request, start_idx)
      # start_idx is where to resume processing (for partial batches)
      pending_inference_requests = []
      
      try:
        for delivery in self._broker_stub.SubscriptionStream(
          SubscribeRequest(topics=[
            ctrl_utils.HANDSHAKE,
            ctrl_utils.CREATE_TRANSPORT,
            ctrl_utils.WEIGHT_UPDATES,
            ctrl_utils.INFERENCE_REQUESTS,
            ctrl_utils.SHUTDOWN,
            ctrl_utils.START_CUDA_PROFILER,
            ctrl_utils.STOP_CUDA_PROFILER,
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
            
            # Reset staleness counter after weight update
            prompts_since_weight_update = 0
            
            # Process pending inference requests (up to staleness limit)
            while pending_inference_requests:
              req, start_idx = pending_inference_requests[0]
              
              processed, new_start_idx = process_prompts(
                req, start_idx, prompts_since_weight_update
              )
              prompts_since_weight_update += processed
              
              if new_start_idx is None:
                # Completed this request
                pending_inference_requests.pop(0)
                logger.info(
                  f"Processed queued request (complete), "
                  f"prompts_since_update={prompts_since_weight_update}/{max_staleness}"
                )
              else:
                # Partial processing - update start_idx and stop
                pending_inference_requests[0] = (req, new_start_idx)
                logger.info(
                  f"Processed queued request (partial: {start_idx}->{new_start_idx}), "
                  f"prompts_since_update={prompts_since_weight_update}/{max_staleness}, "
                  f"waiting for next weight update"
                )
                break

          elif delivery.topic.id == ctrl_utils.INFERENCE_REQUESTS.id:
            inference_request = ctrl.InferenceRequest()
            delivery.message.payload.Unpack(inference_request)
            num_prompts = len(inference_request.prompts)
            
            # If we have pending requests, queue to maintain order
            if pending_inference_requests:
              pending_inference_requests.append((inference_request, 0))
              logger.info(
                f"Queued inference request (have pending), "
                f"queue size: {len(pending_inference_requests)}"
              )
              continue
            
            # Try to process (may be partial if staleness limit hit)
            processed, new_start_idx = process_prompts(
              inference_request, 0, prompts_since_weight_update
            )
            prompts_since_weight_update += processed
            
            if new_start_idx is None:
              # Completed entire request
              logger.info(
                f"Processed inference request, "
                f"prompts_since_update={prompts_since_weight_update}/{max_staleness}"
              )
            else:
              # Partial - queue remainder
              pending_inference_requests.append((inference_request, new_start_idx))
              logger.info(
                f"Processed {processed} prompts, queued remainder starting at {new_start_idx}, "
                f"prompts_since_update={prompts_since_weight_update}/{max_staleness}"
              )

          elif delivery.topic.id == ctrl_utils.START_CUDA_PROFILER.id:
            servicer.start_cuda_profiler()

          elif delivery.topic.id == ctrl_utils.STOP_CUDA_PROFILER.id:
            servicer.stop_cuda_profiler()

          elif delivery.topic.id == ctrl_utils.SHUTDOWN.id:
            if pending_inference_requests:
              logger.warning(
                f"Shutting down with {len(pending_inference_requests)} "
                f"pending inference requests"
              )
            servicer.shutdown()
            break

          else:
            raise Exception(f"Unexpected topic {delivery.topic.id}.")

      except Exception as e:
        if isinstance(e, grpc.RpcError) and e.code() in (
            grpc.StatusCode.CANCELLED, grpc.StatusCode.UNAVAILABLE
        ):
          print('Async rollout client stream closed.')
          return
        else:
          print("Failure in async rollout subscriber:", e)
          traceback.print_exc()
          raise e

    self._update_future = self._executor.submit(call)


def make_async_rollout_client(url, executor=None, timeout=60, max_staleness=0):
  """Create an async rollout client with staleness control.
  
  Args:
      url: Gateway gRPC address.
      executor: ThreadPoolExecutor for background tasks.
      timeout: Timeout for channel readiness.
      max_staleness: Maximum prompts to process before requiring a weight update.
          0 means no limit (unlimited staleness).
  
  Returns:
      AsyncRolloutClient instance.
  """
  if executor is None:
    executor = ThreadPoolExecutor()
  channel = grpc.insecure_channel(url)
  grpc.channel_ready_future(channel).result(timeout=timeout)
  controller_stub = ctrl_grpc.CouplingControllerStub(channel)
  broker_stub = broker_grpc.MessageBrokerStub(channel)
  return AsyncRolloutClient(executor, controller_stub, broker_stub, channel, max_staleness)
