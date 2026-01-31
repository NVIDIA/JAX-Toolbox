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
"""Standalone LLama3 model implementation using Flax NNX.

This is a minimal, self-contained implementation without config dataclasses.
Shardings are not hardcoded - they come from the loaded parameter pytree.
"""

from typing import Tuple

from flax import nnx
import jax
from jax import numpy as jnp
import jaxtyping

K_MASK = -2.3819763e38

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


class Einsum(nnx.Module):
    """Einsum is a convenience module for parameterized tensor multiplication."""

    def __init__(
        self,
        einsum_str: str,
        shape: Tuple[int, ...],
        *,
        rngs: nnx.Rngs,
    ):
        self.einsum_str = einsum_str
        self.shape = shape
        self.w = nnx.Param(nnx.initializers.normal()(rngs.params(), shape))

    @jax.named_scope("einsum")
    def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        return jnp.einsum(self.einsum_str, x, self.w.value)


class Embedder(nnx.Module):
    """Embedder module."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.input_embedding = nnx.Param(
            nnx.initializers.normal()(rngs.params(), (vocab_size, embed_dim))
        )

    @jax.named_scope("embedder_encode")
    def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        return self.input_embedding[(x,)]

    @jax.named_scope("embedder_decode")
    def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        return jnp.dot(x, self.input_embedding.value.T)


def apply_rope(
    inputs: jaxtyping.Array,  # [B, L, N, H]
    positions: jaxtyping.Array,  # [B, L]
    head_dim: int,
    rope_theta: int = 500_000,
) -> jaxtyping.Array:
    """Applies Rotary Position Embedding (RoPE)."""
    fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction

    sinusoid_inp = (
        positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    )
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)


class RMSNorm(nnx.Module):
    """RMSNorm layer."""

    def __init__(
        self,
        dim: int,
        *,
        norm_eps: float = 1e-06,
        rngs: nnx.Rngs,
    ):
        self.w = nnx.Param(nnx.initializers.ones_init()(rngs.params(), (dim,)))
        self.norm_eps = norm_eps

    @jax.named_scope("rms_norm")
    def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
        dtype = x.dtype
        rms = jnp.sqrt(
            jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True)
            + self.norm_eps
        )
        return jnp.astype(self.w * x / rms, dtype)


class Attention(nnx.Module):
    """Multi-head attention with Grouped Query Attention (GQA) support."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        rope_theta: int = 500_000,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.rope_theta = rope_theta
        self.n_rep = num_heads // num_kv_heads
        self.scale = head_dim**-0.5

        self.q_proj = Einsum(
            einsum_str="BTD,DNH->BTNH",
            shape=(embed_dim, num_heads, head_dim),
            rngs=rngs,
        )
        self.k_proj = Einsum(
            einsum_str="BSD,DKH->BSKH",
            shape=(embed_dim, num_kv_heads, head_dim),
            rngs=rngs,
        )
        self.v_proj = Einsum(
            einsum_str="BSD,DKH->BSKH",
            shape=(embed_dim, num_kv_heads, head_dim),
            rngs=rngs,
        )
        self.o_proj = Einsum(
            einsum_str="BTNH,NHD->BTD",
            shape=(num_heads, head_dim, embed_dim),
            rngs=rngs,
        )

    @jax.named_scope("attention")
    def __call__(
        self,
        x: jaxtyping.Array,
        segment_pos: jaxtyping.Array,
        cache: LayerCache | None,
        attn_mask: jaxtyping.Array | None,
    ) -> tuple[LayerCache | None, jaxtyping.Array]:
        """Attention forward pass."""
        seq_len = x.shape[1]

        query_proj = self.q_proj(x)
        key_proj = self.k_proj(x)
        value_proj = self.v_proj(x)

        query_proj = apply_rope(
            query_proj,
            segment_pos,
            head_dim=self.head_dim,
            rope_theta=self.rope_theta,
        )
        key_proj = apply_rope(
            key_proj,
            segment_pos,
            head_dim=self.head_dim,
            rope_theta=self.rope_theta,
        )

        if cache is not None:
            end_index = cache["end_index"][0]
            slice_indices = (0, end_index % cache["v"].shape[1], 0, 0)
            value_proj = jax.lax.dynamic_update_slice(
                cache["v"],
                value_proj,
                slice_indices,
            )
            key_proj = jax.lax.dynamic_update_slice(
                cache["k"], key_proj, slice_indices
            )

        b, t, qh, d = query_proj.shape
        _, s, kh, _ = key_proj.shape

        # Grouped Query Attention
        query_proj = query_proj.reshape((b, t, kh, qh // kh, d))
        attn = jnp.einsum("BTHGD,BSHD->BHGTS", query_proj, key_proj) * self.scale
        attn = attn.reshape((b, qh, t, s))

        if attn_mask is not None:
            attn = jnp.where((jnp.expand_dims(attn_mask, -3)), attn, K_MASK)

        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
            key_proj.dtype
        )

        attn = attn.reshape((b, kh, qh // kh, t, s))
        qkv = jnp.einsum("BHGTS,BSHD->BTHGD", attn, value_proj)
        qkv = qkv.reshape((b, t, qh, d))

        outputs = self.o_proj(qkv)

        if cache is not None:
            new_cache = {
                "v": value_proj,
                "k": key_proj,
                "end_index": cache["end_index"] + seq_len,
            }
        else:
            new_cache = None

        return new_cache, outputs


class MLP(nnx.Module):
    """MLP module with SwiGLU activation."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.gate_proj = nnx.Linear(
            in_features=embed_dim,
            out_features=hidden_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            in_features=embed_dim,
            out_features=hidden_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            in_features=hidden_dim,
            out_features=embed_dim,
            use_bias=False,
            rngs=rngs,
        )

    @jax.named_scope("feed_forward")
    def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
        outputs = self.down_proj(activations)
        return outputs


class DecoderLayer(nnx.Module):
    """Single transformer decoder layer."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        rope_theta: int = 500_000,
        norm_eps: float = 1e-5,
        *,
        rngs: nnx.Rngs,
    ):
        self.input_layernorm = RMSNorm(
            embed_dim,
            norm_eps=norm_eps,
            rngs=rngs,
        )
        self.attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            rope_theta=rope_theta,
            rngs=rngs,
        )
        self.mlp = MLP(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.post_attention_layernorm = RMSNorm(
            embed_dim,
            norm_eps=norm_eps,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jaxtyping.Array,
        segment_pos: jaxtyping.Array,
        cache: LayerCache | None,
        attn_mask: jaxtyping.Array,
    ) -> tuple[LayerCache | None, jaxtyping.Array]:
        inputs_normalized = self.input_layernorm(x)
        cache, attn_output = self.attn(
            inputs_normalized,
            segment_pos,
            cache,
            attn_mask,
        )
        attn_output = attn_output + x
        residual = attn_output
        attn_output = self.post_attention_layernorm(attn_output)
        outputs = residual + self.mlp(attn_output)
        return cache, outputs


class Llama3(nnx.Module):
    """Standalone LLama3 model.

    This implementation takes all model hyperparameters directly as constructor
    arguments rather than using a config dataclass. Shardings are not hardcoded;
    they are inherited from the loaded parameter pytree.

    Example usage for Llama3.2 1B:
        model = Llama3(
            num_layers=16,
            vocab_size=128256,
            embed_dim=2048,
            hidden_dim=8192,
            num_heads=32,
            head_dim=64,
            num_kv_heads=8,
            rope_theta=500_000,
            norm_eps=1e-5,
            weight_tying=True,
            rngs=nnx.Rngs(0),
        )
    """

    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        rope_theta: int = 500_000,
        norm_eps: float = 1e-5,
        weight_tying: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.weight_tying = weight_tying

        self.embedder = Embedder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            rngs=rngs,
        )

        self.layers = [
            DecoderLayer(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                rope_theta=rope_theta,
                norm_eps=norm_eps,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ]

        self.final_norm = RMSNorm(
            embed_dim,
            rngs=rngs,
            norm_eps=norm_eps,
        )

        if not weight_tying:
            self.lm_head = Einsum(
                einsum_str="BTD,DV->BTV",
                shape=(embed_dim, vocab_size),
                rngs=rngs,
            )

    def __call__(
        self,
        input_tokens: jaxtyping.Array,  # [B, L]
        positions: jaxtyping.Array,  # [B, L]
        cache: Cache | None,  # (sequence length L')
        attention_mask: jaxtyping.Array,  # [B, L, L']
    ) -> tuple[jaxtyping.Array, Cache | None]:
        """LLama3 forward pass.

        Args:
            input_tokens: Input sequence of tokens.
            positions: Input absolute positions.
            cache: Attention KV cache or None.
            attention_mask: Transformer input mask.

        Returns:
            Tuple of (predicted_logits, new_cache):
                - predicted_logits: Output logits predicted by the model.
                - new_cache: Updated cache if input cache is not None, else None.
        """
        new_cache = None if cache is None else {}
        x = self.embedder.encode(input_tokens)

        for i, layer in enumerate(self.layers):
            layer_name = f"layer_{i}"
            layer_cache = cache[layer_name] if cache else None
            layer_cache, x = layer(
                x,
                positions,
                layer_cache,
                attention_mask,
            )
            if cache is not None:
                new_cache[layer_name] = layer_cache

        x = self.final_norm(x)

        if self.weight_tying:
            logits = self.embedder.decode(x)
        else:
            logits = self.lm_head(x)

        return logits, new_cache
