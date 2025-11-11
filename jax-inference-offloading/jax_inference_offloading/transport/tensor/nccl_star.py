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
import itertools as it
from logging import getLogger
from typing import Any, List, Optional, Tuple

import numpy as np
from cupy import cuda
from cupy.cuda import nccl
from cupy.cuda import runtime as cudart

from .nccl_base import NcclTransport, nccl_type

try:
  import torch
  has_torch = True
except ImportError:
  has_torch = False

try:
  import jax
  import jax.numpy as jnp
  has_jax = True
except ImportError:
  has_jax = False

logger = getLogger(__name__)


class NcclStarTransport(NcclTransport):
  """NCCL transport with star topology.

  This implements two modes:
  - "fan-in": multi-sender, single-receiver
  - "fan-out": single-sender, multi-receiver
  """

  @classmethod
  def configure(
      cls,
      gateway_client: Any,
      *,
      trainer_ranks: int,
      rollout_ranks: int,
  ) -> dict[str, Any]:
    mode = 'fan-in' if trainer_ranks >= rollout_ranks else 'fan-out'
    unique_ids = [
      cls.encode_nccl_id(gateway_client.get_nccl_id())
      for _ in range(min(trainer_ranks, rollout_ranks))
    ]
    config = dict(
      BACKEND='NCCL',
      UNIQUE_IDS=unique_ids,
      MODE=mode,
      TRAINER_RANKS=trainer_ranks,
      ROLLOUT_RANKS=rollout_ranks,
    )
    return config

  # TODO: with a unified process model/abstraction, trainer/rollout transports can be unified
  # and the only difference is whether a process on in the center or on the arm of the star.
  @classmethod
  def create_trainer_transport(cls, config: dict[str, Any]) -> List['NcclStarTransport']:
    """Create NCCL communicators on the trainer (JAX) process.

    Returns a list of `NcclStarTransport` instances, one per local JAX device
    participating in the star group.
    """
    import jax

    logger.debug('create_trainer_transport: %s', config)

    if config['MODE'] == 'fan-in':
      # JAX TP size is larger than vLLM TP size
      # within each transport, multiple JAX GPUs will send to one vLLM GPU
      # JAX side: ncclSend
      # vLLM side: gather (=ncclRecv + concat)
      k = config['TRAINER_RANKS'] // config['ROLLOUT_RANKS']
      world_size = k + 1  # k JAX ranks + 1 rollout rank
      transports = []
      for device_group in it.batched(jax.local_devices(), k):
        comms = []
        nccl.groupStart()
        for i, dev in enumerate(device_group):
          star_id = dev.id // k
          unique_id = cls.decode_nccl_id(config["UNIQUE_IDS"][star_id])
          rank = dev.id % k + 1  # rank 0 reserved for the rollout peer, which is at the center of the star
          with cuda.Device(dev.local_hardware_id):
            comms.append(nccl.NcclCommunicator(world_size, unique_id, rank))
        nccl.groupEnd()
        for comm in comms:
          transports.append(cls(comm=comm))
      return transports
    elif config['MODE'] == 'fan-out':
      # JAX TP size is smaller than vLLM TP size
      # within each transport, each JAX GPU will send to multiple vLLM GPUs
      # JAX side: scatter (=split + ncclSend)
      # vLLM side: ncclRecv
      k = config['ROLLOUT_RANKS'] // config['TRAINER_RANKS']
      world_size = k + 1  # 1 trainer rank + k rollout ranks
      transports = []
      for dev in jax.local_devices():
        star_id = dev.id
        unique_id = cls.decode_nccl_id(config["UNIQUE_IDS"][star_id])
        with cuda.Device(dev.local_hardware_id):
          comm = nccl.NcclCommunicator(world_size, unique_id, 0)  # trainer rank is at the center of the star in fan-out mode and is always rank 0
        transports.append(cls(comm=comm))
      return transports

  @classmethod
  def create_rollout_transport(cls, config: dict[str, Any], tp_rank: int) -> 'NcclStarTransport':
    """Create a NCCL communicator on a vLLM TP worker process."""
    logger.warning('create_rollout_transport: %s', config)

    if config['MODE'] == 'fan-in':
      k = config['TRAINER_RANKS'] // config['ROLLOUT_RANKS']
      world_size = k + 1  # k JAX ranks + 1 rollout rank
      star_id = tp_rank
      unique_id = cls.decode_nccl_id(config["UNIQUE_IDS"][star_id])
      comm = nccl.NcclCommunicator(world_size, unique_id, 0)  # rollout rank is at the center of the star in fan-in mode and is always rank 0
      return cls(comm)
    elif config['MODE'] == 'fan-out':
      k = config['ROLLOUT_RANKS'] // config['TRAINER_RANKS']
      world_size = k + 1  # 1 trainer rank + k vLLM ranks
      star_id = tp_rank // k  # global rank in the JAX TP mesh
      rank = 1 + tp_rank % k  # rollout ranks in [1, k]
      unique_id = cls.decode_nccl_id(config["UNIQUE_IDS"][star_id])
      comm = nccl.NcclCommunicator(world_size, unique_id, rank)
      return cls(comm)

  def __init__(self, comm: nccl.NcclCommunicator):
    self._comm = comm
    self._pending_xfers: List[Any] = []

  def __repr__(self) -> str:
    return f"NcclStarTransport({self._comm})"

  def send(self, buffer: Any, peer: int = 0, stream: Optional[int] = None) -> None:
    self._pending_xfers.append(buffer)
    assert self._comm.device_id() == buffer.device.local_hardware_id
    with cuda.Device(buffer.device.local_hardware_id):
      stream = stream or cuda.get_current_stream().ptr
      # print(f'  FAN-IN from {self._comm.rank_id()} shape {buffer.shape} stream {stream}')
      self._comm.send(
        sendbuf=buffer.unsafe_buffer_pointer(),
        count=buffer.size,
        datatype=nccl_type(buffer.dtype.name),
        peer=peer,
        stream=stream,
      )

  def gather(
      self,
      shape: np.ndarray | List[int] | Tuple[int, ...],
      dtype: str,
      dim: int,
      parallelism: int,
  ) -> 'torch.Tensor':

    assert self._comm.rank_id() == 0, \
      "Star gather must converge on the root (rank 0)."
    shards = []
    shard_shape = np.array(shape, dtype=np.int32)
    shard_shape[dim] //= parallelism
    shard_shape = shard_shape.tolist()
    with cuda.Device(self._comm.device_id()):
      for peer in range(1, self._comm.size()):  # rank 0 is the current GPU
        # logger.warning(f'GATHER from {peer} shape {shard_shape}')
        shard = torch.empty(shard_shape, dtype=getattr(torch, dtype), device=torch.cuda.current_device())
        self._comm.recv(
          recvbuf=shard.data_ptr(),
          count=shard.numel(),
          datatype=nccl_type(dtype),
          peer=peer,
          stream=cuda.get_current_stream().ptr,
        )
        shards.append(shard)
      cudart.deviceSynchronize()
    return torch.cat(shards, dim=dim)

  def gather_grouped(
      self,
      param_specs: List[Tuple[np.ndarray | List[int] | Tuple[int, ...], str, int, int]],
  ) -> List['torch.Tensor']:
    """Receive multiple tensors from multiple peers in a grouped NCCL operation.

    Args:
        param_specs: List of (shape, dtype, dim, parallelism) tuples for each parameter

    Returns:
        List of gathered tensors, one per parameter

    """
    assert self._comm.rank_id() == 0, \
      "Star gather must converge on the root (rank 0)."

    gathered_tensors = []

    with cuda.Device(self._comm.device_id()):
      # For each parameter, we need to receive shards from all peers
      for shape, dtype, dim, parallelism in param_specs:
        shards = []
        shard_shape = np.array(shape, dtype=np.int32)
        shard_shape[dim] //= parallelism
        shard_shape = shard_shape.tolist()

        # Start grouped NCCL operations for this parameter across all peers
        nccl.groupStart()
        for peer in range(1, self._comm.size()):  # rank 0 is the current GPU
          shard = torch.empty(shard_shape, dtype=getattr(torch, dtype), device=torch.cuda.current_device())
          self._comm.recv(
            recvbuf=shard.data_ptr(),
            count=shard.numel(),
            datatype=nccl_type(dtype),
            peer=peer,
            stream=cuda.get_current_stream().ptr,
          )
          shards.append(shard)
        nccl.groupEnd()

        gathered_tensors.append(torch.cat(shards, dim=dim))

      cudart.deviceSynchronize()

    return gathered_tensors

  def scatter(self, buffers: List[Any]) -> None:
    self._pending_xfers.append(buffers)

    assert self._comm.rank_id() == 0, \
      "Star scatter must originate from the root (rank 0)."

    nccl.groupStart()
    for peer, buffer in zip(range(1, self._comm.size()), buffers):  # rank 0 is the current GPU
      assert self._comm.device_id() == buffer.device.local_hardware_id
      with cuda.Device(buffer.device.local_hardware_id):
        stream = cuda.get_current_stream().ptr
        # print(f'  FAN-OUT to {peer} shape {buffer.shape}')
        self._comm.send(
          sendbuf=buffer.unsafe_buffer_pointer(),
          count=buffer.size,
          datatype=nccl_type(buffer.dtype.name),
          peer=peer,
          stream=stream,
        )
    nccl.groupEnd()

  def scatter_grouped(self, all_buffers: List[List[Any]]) -> None:
    """Scatter multiple tensors to multiple peers in a grouped NCCL operation.
    
    Args:
        all_buffers: List of buffer lists, where each inner list contains buffers for each peer
                     all_buffers[i] contains the buffers to send for the i-th parameter

    """
    self._pending_xfers.append(all_buffers)

    assert self._comm.rank_id() == 0, \
      "Star scatter must originate from the root (rank 0)."

    num_peers = self._comm.size() - 1
    with cuda.Device(self._comm.device_id()):
      stream = cuda.get_current_stream().ptr

      # Group all sends together
      nccl.groupStart()
      for param_idx, buffers in enumerate(all_buffers):
        for peer, buffer in zip(range(1, self._comm.size()), buffers):
          assert self._comm.device_id() == buffer.device.local_hardware_id
          self._comm.send(
            sendbuf=buffer.unsafe_buffer_pointer(),
            count=buffer.size,
            datatype=nccl_type(buffer.dtype.name),
            peer=peer,
            stream=stream,
          )
      nccl.groupEnd()

  def recv(
      self,
      shape: np.ndarray | List[int] | Tuple[int, ...],
      dtype: str,
      peer: int = 0,
  ) -> 'torch.Tensor':
    if isinstance(shape, np.ndarray):
      shape = shape.tolist()
    else:
      shape = list(shape)
    with cuda.Device(self._comm.device_id()):
      buffer = torch.empty(shape, dtype=getattr(torch, dtype), device=torch.cuda.current_device())
      self._comm.recv(
        recvbuf=buffer.data_ptr(),
        count=buffer.numel(),
        datatype=nccl_type(dtype),
        peer=peer,
        stream=cuda.get_current_stream().ptr,
      )
    return buffer

  def recv_grouped(
      self,
      param_specs: List[Tuple[np.ndarray | List[int] | Tuple[int, ...], str]],
  ) -> List['torch.Tensor']:
    """Receive multiple tensors from a single peer in a grouped NCCL operation.

    Args:
        param_specs: List of (shape, dtype) tuples for each parameter

    Returns:
        List of received tensors, one per parameter

    """
    received_tensors = []

    with cuda.Device(self._comm.device_id()):
      stream = cuda.get_current_stream().ptr

      # Group all receives together
      nccl.groupStart()
      for shape, dtype in param_specs:
        if isinstance(shape, np.ndarray):
          shape = shape.tolist()
        else:
          shape = list(shape)

        buffer = torch.empty(shape, dtype=getattr(torch, dtype), device=torch.cuda.current_device())
        self._comm.recv(
          recvbuf=buffer.data_ptr(),
          count=buffer.numel(),
          datatype=nccl_type(dtype),
          peer=0,  # receive from rank 0 (trainer) in fan-out mode
          stream=stream,
        )
        received_tensors.append(buffer)
      nccl.groupEnd()

      cudart.deviceSynchronize()

    return received_tensors

  def wait_for_completion(self) -> None:
    with cuda.Device(self._comm.device_id()):
      cudart.deviceSynchronize()
    self._pending_xfers = []
