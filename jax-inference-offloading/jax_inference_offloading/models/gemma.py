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


def _get_gemma_mapping(
  vocab_size,
  n_layers,
  hidden_size,
  q_heads,
  kv_heads,
  head_dim,
  ffn_size,
  qkv_merged,
  jax_prefix="model",
  vllm_prefix="model",
) -> mapping.TpModelMappingSpecs:
  param_mapping = partial(make_mapping, jax_prefix=jax_prefix, vllm_prefix=vllm_prefix)

  def qkv_param(layer_id, jax_name, vllm_name, slice=[], heads=kv_heads, replication_axis=None):
    return param_mapping(
      jax_name=f"layers.{layer_id}.attn.{jax_name}.w",
      vllm_name=f"layers.{layer_id}.self_attn.{vllm_name}.weight",
      vllm_shape=[heads * head_dim, hidden_size],
      transform=make_transform(slice=slice, transpose=[0, 2, 1], reshape=[-1, hidden_size], replication_axis=replication_axis),
    )

  def qkv_specs(layer_id):
    if qkv_merged:
      assert q_heads == kv_heads
      return [
        qkv_param(
          layer_id,
          "qkv_einsum",
          f"{shard}_proj",
          (shard_id, Ellipsis),
          replication_axis=(2 if shard in ("k", "v") else None),
        )
        for (shard_id, shard) in enumerate(["q", "k", "v"])
      ]
    else:
      return [
        qkv_param(layer_id, "q_einsum", "q_proj", None, q_heads),
        qkv_param(layer_id, "kv_einsum", "k_proj", (0, Ellipsis), replication_axis=2),
        qkv_param(layer_id, "kv_einsum", "v_proj", (1, Ellipsis), replication_axis=2),
      ]

  result = mapping.TpModelMappingSpecs()

  params = [
    # ───────────────────────── singletons ────────────────────────────
    param_mapping(
      jax_name="embedder.input_embedding",  # (256 128, hidden_size)
      vllm_name="embed_tokens.weight",  # (vocab_size, hidden_size)
      vllm_shape=[vocab_size, hidden_size],
      transform=make_transform(slice=(slice(0, vocab_size), slice(None))),
    ),
    param_mapping(
      jax_name="final_norm.scale",
      vllm_name="norm.weight",
      vllm_shape=[hidden_size],
    ),
  ] + [
    # ──────────────────────── per-layer parameters ────────────────────
    param
    for layer_id in range(n_layers)
    for param in [
      # — layer norms —
      param_mapping(
        jax_name=f"layers.{layer_id}.pre_attention_norm.scale",
        vllm_name=f"layers.{layer_id}.input_layernorm.weight",
        vllm_shape=[hidden_size],
      ),
      param_mapping(
        jax_name=f"layers.{layer_id}.pre_ffw_norm.scale",
        vllm_name=f"layers.{layer_id}.post_attention_layernorm.weight",
        vllm_shape=[hidden_size],
      ),
      # — MLP projections —
      param_mapping(
        jax_name=f"layers.{layer_id}.mlp.gate_proj.kernel",  # (hidden_size, ffn_size)
        vllm_name=f"layers.{layer_id}.mlp.gate_proj.weight",
        vllm_shape=[ffn_size, hidden_size],
        transform=make_transform(transpose=[1, 0]),
      ),
      param_mapping(
        jax_name=f"layers.{layer_id}.mlp.up_proj.kernel",  # (hidden_size, ffn_size)
        vllm_name=f"layers.{layer_id}.mlp.up_proj.weight",
        vllm_shape=[ffn_size, hidden_size],
        transform=make_transform(transpose=[1, 0]),
      ),
      param_mapping(
        jax_name=f"layers.{layer_id}.mlp.down_proj.kernel",  # (ffn_size, hidden_size)
        vllm_name=f"layers.{layer_id}.mlp.down_proj.weight",
        vllm_shape=[hidden_size, ffn_size],
        transform=make_transform(transpose=[1, 0]),
      ),
      # — Q / K / V —
      *qkv_specs(layer_id),
      # — attention output —
      param_mapping(
        jax_name=f"layers.{layer_id}.attn.attn_vec_einsum.w",  # (q_heads, head_dim, hidden_size)
        vllm_name=f"layers.{layer_id}.self_attn.o_proj.weight",  # (hidden_size, q_heads × head_dim)
        vllm_shape=[hidden_size, q_heads * head_dim],
        transform=make_transform(
          transpose=[2, 0, 1],
          reshape=[hidden_size, -1],
        ),
      ),
    ]
  ]
  result.mappings.extend(params)
  return result


def get_gemma_2b_mapping(jax_prefix="model", vllm_prefix="model") -> mapping.TpModelMappingSpecs:
  return _get_gemma_mapping(
    vocab_size=256_000,
    n_layers=18,
    hidden_size=2048,
    q_heads=8,
    kv_heads=1,
    head_dim=256,
    ffn_size=16384,
    jax_prefix=jax_prefix,
    vllm_prefix=vllm_prefix,
    qkv_merged=False,
  )


def get_gemma_7b_mapping(jax_prefix="model", vllm_prefix="model") -> mapping.TpModelMappingSpecs:
  return _get_gemma_mapping(
    vocab_size=256_000,
    n_layers=28,
    hidden_size=3072,
    q_heads=16,
    kv_heads=16,
    head_dim=256,
    ffn_size=24576,
    jax_prefix=jax_prefix,
    vllm_prefix=vllm_prefix,
    qkv_merged=True,
  )
