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
from functools import partial

import jax_inference_offloading.api.param_mapping_pb2 as mapping
from jax_inference_offloading.models import make_mapping, make_transform


def _get_gemma3_mapping(
  vocab_size,
  n_layers,
  hidden_size,
  q_heads,
  kv_heads,
  head_dim,
  ffn_size,
  qkv_merged,
  jax_prefix: str = "model",
  vllm_prefix: str = "model",
) -> mapping.TpModelMappingSpecs:
  param_mapping = partial(make_mapping, jax_prefix=jax_prefix, vllm_prefix=vllm_prefix)

  def qkv_param(layer_id, jax_name, vllm_name, slice_spec=None, heads=kv_heads, replication_axis=None):
    return param_mapping(
      jax_name=f"layers.{layer_id}.attn.{jax_name}.w",
      vllm_name=f"layers.{layer_id}.self_attn.{vllm_name}.weight",
      vllm_shape=[heads * head_dim, hidden_size],
      transform=make_transform(slice=slice_spec, transpose=[0, 2, 1], reshape=[-1, hidden_size], replication_axis=replication_axis),
    )

  def qkv_specs(layer_id):
    if qkv_merged:  # merged QKV tensor
      assert q_heads == kv_heads
      return [
        qkv_param(
          layer_id,
          "qkv_einsum",
          f"{shard}_proj",
          (shard_id, Ellipsis),
          replication_axis=(2 if shard in ("k", "v") else None),
        )
        for shard_id, shard in enumerate(("q", "k", "v"))
      ]
    else:  # separate Q / KV tensors
      return [
        qkv_param(layer_id, "q_einsum", "q_proj", None, q_heads),
        qkv_param(layer_id, "kv_einsum", "k_proj", (0, Ellipsis), replication_axis=2),
        qkv_param(layer_id, "kv_einsum", "v_proj", (1, Ellipsis), replication_axis=2),
      ]

  params = [
    # singletons
    param_mapping(
      "embedder.input_embedding",
      "embed_tokens.weight",
      [vocab_size, hidden_size],
    ),
    param_mapping("final_norm.scale", "norm.weight", [hidden_size]),
  ]

  # per-layer
  for layer_id in range(n_layers):
    params.extend(
      [
        # Layer norms
        param_mapping(
          f"layers.{layer_id}.pre_attention_norm.scale",
          f"layers.{layer_id}.input_layernorm.weight",
          [hidden_size],
        ),
        param_mapping(
          f"layers.{layer_id}.post_attention_norm.scale",
          f"layers.{layer_id}.post_attention_layernorm.weight",
          [hidden_size],
        ),
        param_mapping(
          f"layers.{layer_id}.pre_ffw_norm.scale",
          f"layers.{layer_id}.pre_feedforward_layernorm.weight",
          [hidden_size],
        ),
        param_mapping(
          f"layers.{layer_id}.post_ffw_norm.scale",
          f"layers.{layer_id}.post_feedforward_layernorm.weight",
          [hidden_size],
        ),
        # MLP projections
        param_mapping(
          f"layers.{layer_id}.mlp.gate_proj.kernel",
          f"layers.{layer_id}.mlp.gate_proj.weight",
          [ffn_size, hidden_size],
          transform=make_transform(transpose=[1, 0]),
        ),
        param_mapping(
          f"layers.{layer_id}.mlp.up_proj.kernel",
          f"layers.{layer_id}.mlp.up_proj.weight",
          [ffn_size, hidden_size],
          transform=make_transform(transpose=[1, 0]),
        ),
        param_mapping(
          f"layers.{layer_id}.mlp.down_proj.kernel",
          f"layers.{layer_id}.mlp.down_proj.weight",
          [hidden_size, ffn_size],
          transform=make_transform(transpose=[1, 0]),
        ),
        # Q / K / V
        *qkv_specs(layer_id),
        # Attention output
        param_mapping(
          f"layers.{layer_id}.attn.attn_vec_einsum.w",
          f"layers.{layer_id}.self_attn.o_proj.weight",
          [hidden_size, q_heads * head_dim],
          transform=make_transform(transpose=[2, 0, 1], reshape=[hidden_size, -1]),
        ),
        # Q / K normalization
        param_mapping(
          f"layers.{layer_id}.attn._key_norm.scale",
          f"layers.{layer_id}.self_attn.k_norm.weight",
          [head_dim],
        ),
        param_mapping(
          f"layers.{layer_id}.attn._query_norm.scale",
          f"layers.{layer_id}.self_attn.q_norm.weight",
          [head_dim],
        ),
      ]
    )

  # assemble protobuf
  model_mapping = mapping.TpModelMappingSpecs()
  model_mapping.mappings.extend(params)
  return model_mapping


def get_gemma3_1b_mapping(jax_prefix: str = "model", vllm_prefix: str = "model") -> mapping.TpModelMappingSpecs:
  return _get_gemma3_mapping(
    vocab_size=262_144,
    n_layers=26,
    hidden_size=1152,
    q_heads=4,
    kv_heads=1,
    head_dim=256,
    ffn_size=6912,
    qkv_merged=False,
    jax_prefix=jax_prefix,
    vllm_prefix=vllm_prefix,
  )
