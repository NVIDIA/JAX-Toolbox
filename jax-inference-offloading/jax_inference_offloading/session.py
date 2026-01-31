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
"""OffloadingSession: Manages gRPC connection and handshake for JAX-vLLM offloading."""

import secrets
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from typing import Optional, Tuple

import grpc
import jax

import jax_inference_offloading.api.controller_pb2 as ctrl
import jax_inference_offloading.api.controller_pb2_grpc as ctrl_grpc
import jax_inference_offloading.api.message_broker_pb2_grpc as broker_grpc
from jax_inference_offloading.api.message_broker_pb2 import SubscribeRequest
from jax_inference_offloading.api.utils import proto_to_dataclass
from jax_inference_offloading.controller.spmd import on_spmd_leader
from jax_inference_offloading.controller.utils import create_topic

logger = getLogger(__name__)


class OffloadingSession:
    """Manages gRPC connection and handshake for JAX-vLLM offloading.

    This class handles the initial setup for offloading inference from JAX to vLLM,
    including gRPC channel management and the handshake protocol. It does not use
    TrainerClient, instead managing gRPC stubs directly.

    Args:
        gateway_url: URL of the gateway server (e.g., "localhost:50051").
        mesh: JAX device mesh for parallelism info.
        model_path: Path to model checkpoint.
        param_mapping_path: Path to custom JSON parameter mapping file.
        model_name: Optional HuggingFace model name. If not provided and
            param_mapping_path is set, uses param_mapping_path for resolution.
        timeout: Timeout in seconds for gRPC channel readiness.

    Raises:
        ValueError: If neither model_name nor param_mapping_path is provided.

    Example:
        >>> session = OffloadingSession(
        ...     gateway_url="localhost:50051",
        ...     mesh=jax.make_mesh((8,), ("tp",)),
        ...     model_path="/path/to/checkpoint",
        ...     param_mapping_path="/path/to/mapping.json",
        ... )
        >>> # Use session with VLLMTransferEngine
    """

    def __init__(
        self,
        gateway_url: str,
        mesh: jax.sharding.Mesh,
        *,
        model_path: Optional[str] = None,
        param_mapping_path: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 60,
    ):
        # Validation: need at least param_mapping_path or model_name
        if param_mapping_path is None and model_name is None:
            raise ValueError(
                "Either param_mapping_path or model_name must be provided"
            )

        # Store configuration
        self.gateway_url = gateway_url
        self.mesh = mesh
        self.model_name = model_name
        self.model_path = model_path
        self.param_mapping_path = param_mapping_path

        # Set up gRPC channel and stubs (no TrainerClient)
        self._channel = grpc.insecure_channel(gateway_url)
        grpc.channel_ready_future(self._channel).result(timeout=timeout)
        self._controller_stub = ctrl_grpc.CouplingControllerStub(self._channel)
        self._broker_stub = broker_grpc.MessageBrokerStub(self._channel)
        # Use daemon threads so they don't block process exit
        self._executor = ThreadPoolExecutor(thread_name_prefix="offloading_session")
        self._shutdown = False

        # Perform handshake
        self._handshake_result = self._do_handshake()

        # Parse handshake results
        self.mapping_specs = proto_to_dataclass(
            self._handshake_result.mapping_specs, 'mapping_specs'
        )
        self.jax_parallelism = self._handshake_result.jax_parallelism
        self.vllm_parallelism = self._handshake_result.vllm_parallelism

        logger.warning(
            f"OffloadingSession initialized: JAX TP={self.jax_parallelism.tp}, "
            f"vLLM TP={self.vllm_parallelism.tp}"
        )

    @on_spmd_leader(
        serializer=lambda m: m.SerializeToString(),
        deserializer=lambda b: (lambda r: (r.ParseFromString(b), r)[1])(ctrl.HandshakeResponse()),
    )
    def _do_handshake(self) -> ctrl.HandshakeResponse:
        """Perform handshake with vLLM. Executed on leader, broadcast to all ranks."""
        response_topic_id = f"handshake/results/{secrets.token_hex(16)}"

        # Set up response stream subscription
        stream = self._broker_stub.SubscriptionStream(
            SubscribeRequest(topics=[create_topic(response_topic_id)])
        )

        # Build handshake request
        request = ctrl.HandshakeRequest(
            response_topic=response_topic_id,
            model_name=self.model_name or "",
            jax_parallelism=ctrl.JaxParallelism(tp=self.mesh.devices.size),
        )

        # Include param_mapping_path if provided
        if self.param_mapping_path:
            request.param_mapping_path = self.param_mapping_path

        # Send handshake request
        self._controller_stub.AsyncHandshake(request)

        # Wait for response
        for delivery in stream:
            result = ctrl.HandshakeResponse()
            delivery.message.payload.Unpack(result)

            # Validate TP sizes
            is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
            assert is_power_of_2(result.vllm_parallelism.tp), \
                "vLLM TP size must be a power of 2."
            assert is_power_of_2(result.jax_parallelism.tp), \
                "JAX TP size must be a power of 2."

            return result

    def get_nccl_id(self) -> Tuple[int, ...]:
        """Get NCCL unique ID from gateway."""
        return tuple(self._controller_stub.GetNcclId(ctrl.GetNcclIdRequest()).ids)

    @property
    def controller_stub(self):
        """Access the gRPC controller stub."""
        return self._controller_stub

    @property
    def broker_stub(self):
        """Access the gRPC message broker stub."""
        return self._broker_stub

    @property
    def executor(self):
        """Access the thread pool executor."""
        return self._executor

    def shutdown(self, shutdown_gateway: bool = True, grace_period: int = 1):
        """Close the gRPC channel and executor.
        
        Args:
            shutdown_gateway: If True, send shutdown signal to gateway server.
            grace_period: Grace period in seconds for gateway shutdown.
        """
        if self._shutdown:
            return
        self._shutdown = True
        
        # Send shutdown signal to gateway (this also shuts down vLLM rollout)
        if shutdown_gateway:
            try:
                self._controller_stub.Shutdown(
                    ctrl.ShutdownRequest(grace_period=grace_period)
                )
            except Exception:
                pass
        
        # Close gRPC channel - this will cause streams to fail
        try:
            self._channel.close()
        except Exception:
            pass
        
        # Shutdown executor and wait for threads to terminate
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            # Python < 3.9 doesn't support cancel_futures
            self._executor.shutdown(wait=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures shutdown is called."""
        self.shutdown()
        return False
