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
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from typing import Any, Dict, List, Optional, Union

from flax import nnx

import jax
import jax_inference_offloading.api.controller_pb2 as ctrl
from jax_inference_offloading.api.utils import proto_to_dataclass
from jax_inference_offloading.controller.trainer_client import make_trainer_client
from jax_inference_offloading.models import flatten_state, get_named_parameters
from jax_inference_offloading.timer import Timer
from jax_inference_offloading.transport.model.nccl_fused import NcclFusedModelTransport
from jax_inference_offloading.transport.model.nccl_grouped import NcclGroupedModelTransport
from jax_inference_offloading.transport.model.nccl_unfused import NcclUnfusedModelTransport

logger = getLogger(__name__)


class OffloadingBridge:

  def __init__(
    self,
    *,
    gateway_url: str,
    model_name: str = None,
    mesh: jax.sharding.Mesh,
    transfer_mode: str = 'fused',  # 'fused' or 'unfused'
    timer: Timer | None = None,
  ):
    self._timer = timer or jax_inference_offloading.timer.Timer()
    with self._timer.section("handshake"):
      self._executor = ThreadPoolExecutor()
      self._gateway = make_trainer_client(gateway_url, self._executor)
      self._handshake_result = self._gateway.handshake(mesh.devices.size, model_name=model_name)
      self._mapping_specs = proto_to_dataclass(self._handshake_result.mapping_specs, 'mapping_specs')
      logger.warning(f'JAX creating transport ({jax.process_index()}/{jax.process_count()}): {self._handshake_result.jax_parallelism.tp} (train) x {self._handshake_result.vllm_parallelism.tp} (rollout)')
      self._transports, self._transport_config = self._gateway.create_transport(
        backend='NCCL',
        trainer_ranks=self._handshake_result.jax_parallelism.tp,
        rollout_ranks=self._handshake_result.vllm_parallelism.tp
      )
      logger.warning(f'JAX {jax.process_index()}/{jax.process_count()} transports created: {self._transports}')
    if transfer_mode == 'fused':
      self._model_transport = NcclFusedModelTransport(
        mesh,
        self._mapping_specs,
        self._gateway,
        self._transports,
        self._transport_config,
        timer=self._timer,
      )
    elif transfer_mode == 'unfused':
      self._model_transport = NcclUnfusedModelTransport(
        mesh,
        self._mapping_specs,
        self._gateway,
        self._transports,
        self._transport_config,
        timer=self._timer,
      )
    elif transfer_mode == 'grouped':
      self._model_transport = NcclGroupedModelTransport(
        mesh,
        self._mapping_specs,
        self._gateway,
        self._transports,
        self._transport_config,
        timer=self._timer,
      )
    else:
      raise ValueError(f"Unknown transfer_mode: {transfer_mode}")

  @property
  def gateway(self):
    return self._gateway

  def transfer(self, state: Union[nnx.Module, nnx.State, Dict[str, jax.Array]]):
    with self._timer.section("to_named_parameters"):
      if isinstance(state, nnx.Module):
        named_parameters = get_named_parameters(state)
      elif isinstance(state, nnx.State):
        named_parameters = flatten_state(state)
      elif isinstance(state, dict):
        named_parameters = state
      else:
        raise ValueError(f"Unsupported state type: {type(state)}")
    with self._timer.section("transfer"):
      self._model_transport(named_parameters)
