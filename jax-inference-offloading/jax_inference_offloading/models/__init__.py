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
def flatten_state(nnx_state):
  """Flatten an NNX state tree into a dictionary with dot-separated keys."""

  def _flatten_dict(nnx_state, prefix=""):
    try:
      yield prefix, nnx_state.value
    except AttributeError:
      for k in nnx_state.keys():
        new_prefix = f"{prefix}.{k}" if prefix else str(k)
        yield from _flatten_dict(nnx_state[k], prefix=new_prefix)

  return dict(_flatten_dict(nnx_state))


def get_named_parameters(nnx_model, *filters):
  """Flatten an NNX model into a dictionary with dot-separated keys."""
  from flax import nnx

  nnx_state = nnx.state(nnx_model, *filters)
  return flatten_state(nnx_state)
