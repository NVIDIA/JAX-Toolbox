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
"""VLLMRolloutEngine: Handles inference requests to vLLM."""

import json
import secrets
import traceback
from logging import getLogger
from queue import Empty, Queue
from typing import List, Optional, Union

import jax_inference_offloading.api.controller_pb2 as ctrl
from jax_inference_offloading.api.message_broker_pb2 import SubscribeRequest
from jax_inference_offloading.api.types import (
    CompletionOutput,
    InferenceConfig,
    InferenceOutput,
)
from jax_inference_offloading.controller.spmd import on_spmd_leader
from jax_inference_offloading.controller.utils import create_topic
from jax_inference_offloading.session import OffloadingSession
from jax_inference_offloading.timer import Timer

logger = getLogger(__name__)


class VLLMRolloutEngine:
    """vLLM-based rollout engine for inference offloading.

    This engine handles inference/rollout generation by sending requests to vLLM
    via the gateway. It is designed to work with an OffloadingSession and does
    not handle weight transfer (use VLLMTransferEngine for that).

    Args:
        session: An initialized OffloadingSession.
        timer: Optional timer for performance profiling.

    Example:
        >>> session = OffloadingSession(
        ...     gateway_url="localhost:50051",
        ...     mesh=mesh,
        ...     model_name="meta-llama/Llama-3.1-8B",
        ... )
        >>> rollout_engine = VLLMRolloutEngine(session)
        >>> config = InferenceConfig(max_tokens=128, temperature=0.9)
        >>> output = rollout_engine.generate(["What is 2+2?"], config)
        >>> print(output.texts[0])
    """

    def __init__(
        self,
        session: OffloadingSession,
        timer: Optional[Timer] = None,
    ):
        self._session = session
        self._timer = timer or Timer()

        # Set up inference response stream
        self._inference_topic_id = f"inference/results/{secrets.token_hex(16)}"
        self._response_queue: Queue = Queue()
        self._stream = None
        self._response_future = None

        # Start background thread to handle responses
        self._setup_response_stream()

        logger.warning("VLLMRolloutEngine initialized")

    def _setup_response_stream(self):
        """Set up the background thread for receiving inference responses."""
        self._stream = self._session.broker_stub.SubscriptionStream(
            SubscribeRequest(topics=[create_topic(self._inference_topic_id)])
        )

        def handle_responses():
            try:
                for delivery in self._stream:
                    result = ctrl.InferenceResponse()
                    delivery.message.payload.Unpack(result)
                    self._response_queue.put(result)
            except Exception as e:
                # Treat intentional cancellations/unavailable server as graceful closure
                import grpc
                if isinstance(e, grpc.RpcError) and e.code() in (
                    grpc.StatusCode.CANCELLED,
                    grpc.StatusCode.UNAVAILABLE,
                ):
                    return
                else:
                    logger.error(f"Error in inference response stream: {e}")
                    traceback.print_exc()
                    raise e

        self._response_future = self._session.executor.submit(handle_responses)

    def generate(
        self,
        prompts: Union[List[str], List[List[int]]],
        config: InferenceConfig,
    ) -> InferenceOutput:
        """Generate completions using vLLM.

        This is a blocking call that waits until vLLM returns the response.

        Args:
            prompts: Text prompts or pre-tokenized prompts. Supports:
                - List[str]: Text prompts
                - List[List[int]]: Pre-tokenized prompts

            config: Inference configuration.

        Returns:
            InferenceOutput with generated completions.
        """
        with self._timer.section("inference"):
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
            request.response_topic = self._inference_topic_id
            request.config.CopyFrom(proto_config)

            # Add prompts to request
            self._add_prompts_to_request(prompts, request)

            # Send inference request (only on leader, but all ranks wait for response)
            self._send_inference_request(request)

            # Wait for response
            response = self._response_queue.get()

        # Convert response to framework-agnostic output
        return self._convert_response(response)

    @on_spmd_leader(broadcast_result=False)
    def _send_inference_request(self, request: ctrl.InferenceRequest):
        """Send inference request to gateway. Only executed on leader."""
        self._session.controller_stub.AsyncInference(request)

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

    def _convert_response(self, response: ctrl.InferenceResponse) -> InferenceOutput:
        """Convert protobuf response to framework-agnostic output."""
        completions = []
        for output in response.outputs:
            completions.append(
                CompletionOutput(
                    text=output.generated_text,
                    token_ids=list(output.generated_tokens.ids),
                    logprobs=(
                        list(output.generated_token_logps)
                        if output.generated_token_logps
                        else None
                    ),
                    prompt_token_ids=(
                        list(output.tokenized_prompt.ids)
                        if output.tokenized_prompt.ids
                        else None
                    ),
                )
            )
        return InferenceOutput(completions=completions)

    @property
    def session(self) -> OffloadingSession:
        """Access the underlying session."""
        return self._session

    @property
    def timer(self) -> Timer:
        """Access the timer for performance analysis."""
        return self._timer

    def shutdown(self) -> None:
        """Shutdown the rollout engine."""
        # Cancel the gRPC stream to unblock the background thread
        if self._stream is not None:
            try:
                self._stream.cancel()
            except Exception:
                pass
        
        # Wait for the background thread to finish
        if self._response_future is not None:
            try:
                self._response_future.result(timeout=5)
            except Exception:
                pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
