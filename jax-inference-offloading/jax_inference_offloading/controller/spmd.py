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
from functools import wraps
from typing import Any, Callable

import cloudpickle as cp
import jax
import numpy as np
from jax.experimental import multihost_utils as mu


def on_spmd_leader(
  fn: Callable = None,
  *,
  leader_fn: Callable = None,
  broadcast_result: bool = True,
  serializer: Callable = None,
  deserializer: Callable = None,
) -> Callable:
  """Ensure that a function is only evaluated on one process in SPMD context.

  Usage:
    - Higher-order: ``on_spmd_leader(fn)(*args, **kwargs)``
    - Decorator without args: ``@on_spmd_leader``
    - Decorator with args: ``@on_spmd_leader(serializer=..., deserializer=...)``
  """

  def _decorate(f: Callable) -> Callable:
    @wraps(f)
    def _wrapped(*args, **kwargs) -> Any:
      is_leader = (leader_fn or (lambda: jax.process_index() == 0))()
      _ser = serializer or cp.dumps
      _des = deserializer or cp.loads

      value = f(*args, **kwargs) if is_leader else None

      if broadcast_result:
        payload = _ser(value) if is_leader else None

        size = np.array(len(payload) if payload is not None else 0, dtype=np.int64)
        size = mu.broadcast_one_to_all(size, is_source=is_leader)

        if size == 0:
          return None

        buf = np.frombuffer(payload, dtype=np.uint8) if is_leader else np.zeros(size, dtype=np.uint8)
        buf = mu.broadcast_one_to_all(buf, is_source=is_leader)
        value = _des(buf.tobytes())
      
      return value

    return _wrapped

  if fn is None:
    return _decorate
  else:
    return _decorate(fn)
