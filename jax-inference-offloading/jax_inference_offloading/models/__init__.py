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
import jax_inference_offloading.api.param_mapping_pb2 as mapping
from jax_inference_offloading.models.mapping_util import slice_to_proto


def make_transform(slice=[], transpose=[], reshape=[], replication_axis=None, replication_count=None):
  result = mapping.JaxParam.Transform()
  if slice:
    result.slice.CopyFrom(slice_to_proto(slice))
  result.transpose.extend(transpose)
  result.reshape.extend(reshape)
  if replication_axis is not None:
    result.replication_axis = int(replication_axis)
  if replication_count is not None:
    result.replication_count = int(replication_count)
  return result


def make_mapping(
  jax_name, vllm_name, vllm_shape, *, transform=None, jax_prefix="model", vllm_prefix="model"
):
  result = mapping.ParamMapping()
  result.vllm_param.name = f"{vllm_prefix}.{vllm_name}".lstrip(".")
  result.vllm_param.shape.extend(vllm_shape)
  result.jax_param.name = f"{jax_prefix}.{jax_name}".lstrip(".")
  if transform is not None:
    result.jax_param.transform.CopyFrom(transform)
  return result


def flatten_state(nnx_state, prefix="model"):
  """Flatten an NNX state tree into a dictionary with dot-separated keys."""

  def _flatten_dict(nnx_state, prefix=""):
    try:
      yield prefix, nnx_state.value
    except AttributeError:
      for k in nnx_state.keys():
        yield from _flatten_dict(nnx_state[k], prefix=".".join([prefix, str(k)]))

  return dict(_flatten_dict(nnx_state, prefix=prefix))


def get_named_parameters(nnx_model, prefix="model", *filters):
  """Flatten an NNX model into a dictionary with dot-separated keys."""
  from flax import nnx

  nnx_state = nnx.state(nnx_model, *filters)
  return flatten_state(nnx_state, prefix=prefix)
