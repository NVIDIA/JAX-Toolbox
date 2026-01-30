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


def _get_llama3_mapping(
  vocab_size,
  n_layers,
  hidden_size,
  q_heads,
  kv_heads,
  head_dim,
  ffn_size,
  vllm_prefix: str = "model",
  tie_word_embeddings: bool = False,
) -> mapping.TpModelMappingSpecs:
  param_mapping = partial(make_mapping, vllm_prefix=vllm_prefix)

  params = [
    # singletons
    param_mapping(
      "embedder.input_embedding",
      "embed_tokens.weight",
      [vocab_size, hidden_size],
    ),
    param_mapping("final_norm.w", "norm.weight", [hidden_size]),
  ]

  # Only add lm_head mapping if embeddings are not tied
  if not tie_word_embeddings:
    params.append(
      make_mapping(
        "lm_head.w", "lm_head.weight", [vocab_size, hidden_size],
        transform=make_transform(transpose=[1, 0]),
        vllm_prefix=''
      )
    )

  # per-layer
  for layer_id in range(n_layers):
    params.extend(
      [
        # Layer norms
        param_mapping(
          f"layers.{layer_id}.input_layernorm.w",
          f"layers.{layer_id}.input_layernorm.weight",
          [hidden_size],
        ),
        param_mapping(
          f"layers.{layer_id}.post_attention_layernorm.w",
          f"layers.{layer_id}.post_attention_layernorm.weight",
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
        param_mapping(
          f"layers.{layer_id}.attn.q_proj.w",
          f"layers.{layer_id}.self_attn.q_proj.weight",
          [q_heads * head_dim, hidden_size],
          transform=make_transform(transpose=[1, 2, 0], reshape=[-1, hidden_size]),
        ),
        param_mapping(
          f"layers.{layer_id}.attn.k_proj.w",
          f"layers.{layer_id}.self_attn.k_proj.weight",
          [kv_heads * head_dim, hidden_size],
          transform=make_transform(transpose=[1, 2, 0], reshape=[-1, hidden_size], replication_axis=2),
        ),
        param_mapping(
          f"layers.{layer_id}.attn.v_proj.w",
          f"layers.{layer_id}.self_attn.v_proj.weight",
          [kv_heads * head_dim, hidden_size],
          transform=make_transform(transpose=[1, 2, 0], reshape=[-1, hidden_size], replication_axis=2),
        ),
        # Attention output
        param_mapping(
          f"layers.{layer_id}.attn.o_proj.w",
          f"layers.{layer_id}.self_attn.o_proj.weight",
          [hidden_size, q_heads * head_dim],
          transform=make_transform(transpose=[2, 0, 1], reshape=[hidden_size, -1]),
        ),
      ]
    )

  # assemble protobuf
  model_mapping = mapping.TpModelMappingSpecs()
  model_mapping.mappings.extend(params)
  return model_mapping

def get_llama3_1b_mapping(vllm_prefix: str = "model") -> mapping.TpModelMappingSpecs:
  return _get_llama3_mapping(
    vocab_size=128_256,
    n_layers=16,
    hidden_size=2048,
    q_heads=32,
    kv_heads=8,
    head_dim=64,
    ffn_size=8192,
    vllm_prefix=vllm_prefix,
    tie_word_embeddings=True,
  )

def get_llama3_8b_mapping(vllm_prefix: str = "model") -> mapping.TpModelMappingSpecs:
  return _get_llama3_mapping(
    vocab_size=128_256,
    n_layers=32,
    hidden_size=4096,
    q_heads=32,
    kv_heads=8,
    head_dim=128,
    ffn_size=14336,
    vllm_prefix=vllm_prefix,
  )

def get_llama3_70b_mapping(vllm_prefix: str = "model") -> mapping.TpModelMappingSpecs:
  return _get_llama3_mapping(
    vocab_size=128_256,
    n_layers=80,
    hidden_size=8192,
    q_heads=64,
    kv_heads=8,
    head_dim=128,
    ffn_size=28672,
    vllm_prefix=vllm_prefix,
  )

def get_llama3_405b_mapping(vllm_prefix: str = "model") -> mapping.TpModelMappingSpecs:
  return _get_llama3_mapping(
    vocab_size=128_256,
    n_layers=126,
    hidden_size=16384,
    q_heads=128,
    kv_heads=8,
    head_dim=128,
    ffn_size=53248,
    vllm_prefix=vllm_prefix,
  )
