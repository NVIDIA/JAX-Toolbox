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
import abc
import platform
from array import array
from base64 import a85decode, a85encode
from logging import getLogger
from typing import Any, List, Tuple

import numpy as np
from cupy.cuda import nccl

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


def nccl_type(dtype: str):
  if dtype == "int8":
    return nccl.NCCL_INT8
  if dtype == "uint8":
    return nccl.NCCL_UINT8
  if dtype == "int32":
    return nccl.NCCL_INT32
  if dtype == "int64":
    return nccl.NCCL_INT64
  if dtype == "float16":
    return nccl.NCCL_FLOAT16
  if dtype == "float32":
    return nccl.NCCL_FLOAT32
  if dtype == "float64":
    return nccl.NCCL_FLOAT64
  if dtype == "bfloat16":
    return nccl.NCCL_BFLOAT16
  if has_torch:
    if dtype == torch.int8:
      return nccl.NCCL_INT8
    if dtype == torch.uint8:
      return nccl.NCCL_UINT8
    if dtype == torch.int32:
      return nccl.NCCL_INT32
    if dtype == torch.int64:
      return nccl.NCCL_INT64
    if dtype == torch.float16:
      return nccl.NCCL_FLOAT16
    if dtype == torch.float32:
      return nccl.NCCL_FLOAT32
    if dtype == torch.float64:
      return nccl.NCCL_FLOAT64
    if dtype == torch.bfloat16:
      return nccl.NCCL_BFLOAT16
  raise ValueError(f"Unsupported dtype: {dtype}")


class NcclTransport(abc.ABC):

  @staticmethod
  def encode_nccl_id(unique_id: Tuple[int, ...]) -> str:
    """Encode a NCCL unique ID to an ASCII A85 string."""
    return a85encode(array('B', (int(i) & 0xFF for i in unique_id)).tobytes()).decode('ascii')

  @staticmethod
  def decode_nccl_id(compressed_id: str) -> Tuple[int, ...]:
    """Decode an ASCII A85 NCCL unique ID to a tuple of ints."""
    if platform.machine().lower() in ('amd64', 'x86_64', 'x64', 'i386', 'i686'):
      return tuple(array('b', a85decode(compressed_id.encode('ascii'))))
    else:
      return tuple(array('B', a85decode(compressed_id.encode('ascii'))))

  @abc.abstractmethod
  def send(self, buffer: Any, peer: int) -> None:
    """Enqueue a device-to-device send to a peer."""
    raise NotImplementedError

  @abc.abstractmethod
  def gather(
      self,
      shape: np.ndarray | List[int] | Tuple[int, ...],
      dtype: str,
      dim: int,
      parallelism: int,
  ) -> 'torch.Tensor':
    """Gather shards from peers and concatenate along ``dim``.

    The resulting tensor should have ``shape`` and be assembled from
    ``parallelism`` shards received from peers.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def scatter(self, buffers: List[Any]) -> None:
    """Scatter per-peer buffers."""
    raise NotImplementedError

  @abc.abstractmethod
  def recv(
      self,
      shape: np.ndarray | List[int] | Tuple[int, ...],
      dtype: str,
      peer: Any,
  ) -> 'torch.Tensor':
    """Enqueue a device-to-device receive from a peer."""
    raise NotImplementedError

  @abc.abstractmethod
  def wait_for_completion(self) -> None:
    """Block until all outstanding transfers complete."""
    raise NotImplementedError
