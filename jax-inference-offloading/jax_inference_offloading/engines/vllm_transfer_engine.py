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
"""VLLMTransferEngine: Handles weight transfer from JAX to vLLM."""

import json
from logging import getLogger
from typing import Dict, Optional, Union

import jax

import jax_inference_offloading.api.controller_pb2 as ctrl
from jax_inference_offloading.controller.spmd import on_spmd_leader
from jax_inference_offloading.models import flatten_state, get_named_parameters
from jax_inference_offloading.session import OffloadingSession
from jax_inference_offloading.timer import Timer
from jax_inference_offloading.transport.model.nccl_fused import NcclFusedModelTransport
from jax_inference_offloading.transport.model.nccl_grouped import NcclGroupedModelTransport
from jax_inference_offloading.transport.model.nccl_unfused import NcclUnfusedModelTransport
from jax_inference_offloading.transport.tensor.nccl_star import NcclStarTransport

logger = getLogger(__name__)


class VLLMTransferEngine:
    """Engine for transferring model weights from JAX to vLLM.

    This engine handles NCCL transport creation and weight transfer operations.
    It is designed to work with an OffloadingSession and does not use TrainerClient.

    Args:
        session: An initialized OffloadingSession.
        transfer_mode: Weight transfer mode ('fused', 'unfused', 'grouped').
            Default is 'grouped' which batches all transfers for efficiency.
        timer: Optional timer for performance profiling.

    Example:
        >>> session = OffloadingSession(
        ...     gateway_url="localhost:50051",
        ...     mesh=mesh,
        ...     model_name="meta-llama/Llama-3.1-8B",
        ... )
        >>> transfer_engine = VLLMTransferEngine(session)
        >>> transfer_engine.update_weights(my_params)
    """

    def __init__(
        self,
        session: OffloadingSession,
        transfer_mode: str = "grouped",
        timer: Optional[Timer] = None,
    ):
        self._session = session
        self._transfer_mode = transfer_mode
        self._timer = timer or Timer()

        # Create NCCL transports
        with self._timer.section("create_transport"):
            self._transports, self._transport_config = self._create_transport()

        # Create model transport based on transfer mode
        with self._timer.section("create_model_transport"):
            if transfer_mode == 'fused':
                self._model_transport = NcclFusedModelTransport(
                    session.mesh,
                    session.mapping_specs,
                    self,  # Pass self as gateway (provides start_weight_transfer)
                    self._transports,
                    self._transport_config,
                    timer=self._timer,
                )
            elif transfer_mode == 'unfused':
                self._model_transport = NcclUnfusedModelTransport(
                    session.mesh,
                    session.mapping_specs,
                    self,
                    self._transports,
                    self._transport_config,
                    timer=self._timer,
                )
            elif transfer_mode == 'grouped':
                self._model_transport = NcclGroupedModelTransport(
                    session.mesh,
                    session.mapping_specs,
                    self,
                    self._transports,
                    self._transport_config,
                    timer=self._timer,
                )
            else:
                raise ValueError(f"Unknown transfer_mode: {transfer_mode}")

        logger.warning(
            f"VLLMTransferEngine initialized with mode={transfer_mode}, "
            f"transports={len(self._transports)}"
        )

    def _create_transport(self):
        """Create NCCL transports for JAX-vLLM communication."""
        transport_cls = NcclStarTransport

        # Configure transport (calls get_nccl_id via self)
        @on_spmd_leader()
        def _configure():
            cfg = transport_cls.configure(
                self,
                trainer_ranks=self._session.jax_parallelism.tp,
                rollout_ranks=self._session.vllm_parallelism.tp,
            )
            # Signal vLLM to create its side of the transport
            self._session.controller_stub.CreateTransport(
                ctrl.CreateTransportRequest(config_json=json.dumps(cfg))
            )
            return cfg

        transport_config = _configure()

        # Create JAX-side transports
        transports = transport_cls.create_trainer_transport(transport_config)

        logger.warning(
            f"Created {len(transports)} NCCL transports in {transport_config['MODE']} mode"
        )

        return transports, transport_config

    def get_nccl_id(self):
        """Get NCCL unique ID from gateway. Used by NcclStarTransport.configure()."""
        return self._session.get_nccl_id()

    @on_spmd_leader(broadcast_result=False)
    def start_weight_transfer(self, mode: str):
        """Signal vLLM to start receiving weights. Used by model transport."""
        self._session.controller_stub.StartWeightUpdate(
            ctrl.StartWeightUpdateRequest(mode=mode)
        )

    def update_weights(
        self,
        params: Union[Dict[str, jax.Array], "nnx.State", "nnx.Module"],  # noqa: F821
    ) -> None:
        """Transfer model weights to vLLM.

        This is a blocking call that waits until all weights are transferred.

        Args:
            params: Model parameters in various formats:
                - Dict[str, jax.Array]: Direct flattened params
                - flax.nnx.State: Flax state object
                - flax.nnx.Module: Flax module (state extracted automatically)
        """
        with self._timer.section("update_weights"):
            # Handle different input formats
            with self._timer.section("to_named_parameters"):
                if isinstance(params, dict):
                    named_params = params
                else:
                    # Try flax.nnx formats
                    try:
                        from flax import nnx

                        if isinstance(params, nnx.Module):
                            named_params = get_named_parameters(params)
                        elif isinstance(params, nnx.State):
                            named_params = flatten_state(params)
                        else:
                            raise TypeError(f"Unsupported params type: {type(params)}")
                    except ImportError:
                        raise TypeError(
                            f"Unsupported params type: {type(params)}. "
                            "Expected Dict[str, jax.Array] or install flax for nnx support."
                        )

            # Transfer via model transport
            with self._timer.section("transfer"):
                self._model_transport(named_params)

    @property
    def session(self) -> OffloadingSession:
        """Access the underlying session."""
        return self._session

    @property
    def timer(self) -> Timer:
        """Access the timer for performance analysis."""
        return self._timer

    @property
    def transfer_mode(self) -> str:
        """Get the current transfer mode."""
        return self._transfer_mode
