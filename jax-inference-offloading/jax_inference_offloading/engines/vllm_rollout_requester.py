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
"""VLLMRolloutRequester: Lightweight client for sending inference requests to vLLM."""

import json
import time
from logging import getLogger
from typing import List, Optional, Union

import grpc
from google.protobuf.wrappers_pb2 import StringValue

import jax_inference_offloading.api.controller_pb2 as ctrl
import jax_inference_offloading.api.controller_pb2_grpc as ctrl_grpc
from jax_inference_offloading.api.types import InferenceConfig

logger = getLogger(__name__)

# Default KV key used by JAX controller to store the response topic
DEFAULT_RESPONSE_TOPIC_KEY = "inference_response_topic"


class VLLMRolloutRequester:
    """Sends inference requests to vLLM via gateway.

    Lightweight client with no JAX dependency. Used by a separate rollout process
    to submit prompts for inference. Does not handle responses - those are received
    by the JAX controller.

    If response_topic is not provided, the constructor will poll the gateway's
    KV store to discover it (waiting for the JAX controller to complete setup).

    Args:
        gateway_url: Gateway gRPC address (e.g., "localhost:50051").
        response_topic: Topic where vLLM will publish inference results.
            If None, will be discovered from gateway KV store.
        timeout: Timeout in seconds for gRPC channel readiness.
        kv_poll_max_retries: Max retries when polling KV for response topic.
        kv_poll_delay: Delay in seconds between KV poll attempts.

    Example:
        >>> # Option 1: Auto-discover response topic from gateway KV
        >>> requester = VLLMRolloutRequester(gateway_url="localhost:50051")
        >>>
        >>> # Option 2: Explicit response topic
        >>> requester = VLLMRolloutRequester(
        ...     gateway_url="localhost:50051",
        ...     response_topic="inference/results/shared",
        ... )
        >>>
        >>> config = InferenceConfig(max_tokens=128, temperature=0.9)
        >>> requester.request(["What is 2+2?"], config)
        >>> requester.shutdown()
    """

    def __init__(
        self,
        gateway_url: str,
        response_topic: Optional[str] = None,
        timeout: int = 60,
        kv_poll_max_retries: int = 120,
        kv_poll_delay: float = 1.0,
    ):
        self._gateway_url = gateway_url

        # Set up gRPC channel and stub
        self._channel = grpc.insecure_channel(gateway_url)
        grpc.channel_ready_future(self._channel).result(timeout=timeout)
        self._controller_stub = ctrl_grpc.CouplingControllerStub(self._channel)

        # Discover response topic from KV if not provided
        if response_topic is None:
            response_topic = self._discover_response_topic(
                kv_poll_max_retries, kv_poll_delay
            )
        self._response_topic = response_topic

        logger.warning(
            f"VLLMRolloutRequester initialized: gateway={gateway_url}, "
            f"response_topic={response_topic}"
        )

    def _discover_response_topic(
        self,
        max_retries: int,
        retry_delay: float,
    ) -> str:
        """Poll gateway KV store to discover the response topic.

        The JAX controller stores the response topic in the KV store after
        completing its setup (handshake, NCCL transport creation, etc.).

        Args:
            max_retries: Maximum number of poll attempts.
            retry_delay: Delay in seconds between attempts.

        Returns:
            The response topic string.

        Raises:
            TimeoutError: If topic not found after max_retries.
        """
        key = DEFAULT_RESPONSE_TOPIC_KEY
        for i in range(max_retries):
            response = self._controller_stub.KVGet(ctrl.KVGetRequest(key=key))
            if response.found:
                result = StringValue()
                response.value.Unpack(result)
                logger.info(f"Discovered response topic from KV: {result.value}")
                return result.value
            time.sleep(retry_delay)
        raise TimeoutError(
            f"Response topic not found in KV store after {max_retries} attempts. "
            f"Ensure the JAX controller is running and has completed setup."
        )

    def request(
        self,
        prompts: Union[str, List[str], List[int], List[List[int]], List[dict], List[List[dict]]],
        config: InferenceConfig,
        batch_id: Optional[str] = None,
        streaming: bool = False,
    ) -> str:
        """Send inference request to vLLM (non-blocking, fire-and-forget).

        The request is sent to the gateway, which forwards it to vLLM.
        Results are published to the response_topic and must be received
        separately (typically by the transfer process).

        Args:
            prompts: Prompts in various formats:
                - str: Single text prompt
                - List[str]: Multiple text prompts
                - List[int]: Single pre-tokenized prompt
                - List[List[int]]: Multiple pre-tokenized prompts
                - List[dict]: Single chat conversation (list of messages)
                - List[List[dict]]: Multiple chat conversations
            config: Inference configuration (max_tokens, temperature, etc.)
            batch_id: Optional correlation ID for async mode. If not provided,
                a unique ID will be generated. Used to correlate streamed
                RolloutResult messages with their originating request.
            streaming: If True, vLLM will publish each rollout as a separate
                RolloutResult message as it completes. If False (default),
                all results are batched into a single InferenceResponse.

        Returns:
            The batch_id used for this request (useful for correlation).
        """
        # Generate batch_id if not provided
        if batch_id is None:
            import uuid
            batch_id = str(uuid.uuid4())

        # Build protobuf config
        proto_config = ctrl.RolloutConfig(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            num_outputs=config.n,
            seed=config.seed or 42,
        )
        proto_config.stop_token_ids.extend(config.stop_token_ids)

        # Build inference request
        request = ctrl.InferenceRequest()
        request.response_topic = self._response_topic
        request.config.CopyFrom(proto_config)
        request.batch_id = batch_id
        request.streaming = streaming

        # Add prompts to request
        self._add_prompts_to_request(prompts, request)

        # Fire-and-forget
        self._controller_stub.AsyncInference(request)

        logger.info(
            f"Sent inference request with {len(request.prompts)} prompts, "
            f"batch_id={batch_id}, streaming={streaming}"
        )

        return batch_id

    def _add_prompts_to_request(
        self,
        prompts: Union[str, List[str], List[int], List[List[int]], List[dict], List[List[dict]]],
        request: ctrl.InferenceRequest,
    ):
        """Add prompts to the inference request in the appropriate format."""
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
                    raise ValueError(
                        f"Invalid prompt format. Expected a list of dicts or ints. Got {p}."
                    )
        else:
            raise ValueError(f"Invalid prompt format: {prompts}")

    def shutdown(self):
        """Close the gRPC channel."""
        try:
            self._channel.close()
        except Exception:
            pass
        logger.warning("VLLMRolloutRequester shut down")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
