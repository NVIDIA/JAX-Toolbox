# Copyright 2022 Google LLC.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.partitioning import param_with_axes

from rosetta.projects.vit.config import GoogleViTConfig
from t5x.examples.t5.layers import _convert_to_activation_function


Array = Any
PRNGKey = Any
Shape = tuple[int]
Dtype = Any


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x):
    return x


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    posemb_init: positional embedding initializer.
  """

  posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]
  dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs):
    """Applies the AddPositionEmbs module.

    Args:
      inputs: Inputs to the layer.

    Returns:
      Output tensor with shape `(bs, timesteps, in_dim)`.
    """
    # inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    pos_emb_shape = (inputs.shape[1], inputs.shape[2])
    pe = param_with_axes('pos_embedding', self.posemb_init, pos_emb_shape, jnp.float32, axes=('length', 'abspos_buckets'))
    pe = jnp.asarray(pe, self.dtype)
    return inputs + pe


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  dtype: Dtype = jnp.float32
  out_dim: int | None = None
  dropout_rate: float = 0.1
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.normal(stddev=1e-6)
  hidden_act: str = 'gelu_new'

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        features=self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        kernel_axes=('embed', 'mlp'),
        bias_axes=('mlp',))(  # pytype: disable=wrong-arg-types
            inputs)
    x = _convert_to_activation_function(self.hidden_act)(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        features=actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        kernel_axes=('mlp', 'embed'),
        bias_axes=('embed',))(  # pytype: disable=wrong-arg-types
            x)
    output = nn.Dropout(
        rate=self.dropout_rate)(
            output, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    inputs: input data.
    mlp_dim: dimension of the mlp on top of attention block.
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout for attention heads.
    deterministic: bool, deterministic or not (to apply dropout).
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
  """

  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  hidden_act: str = 'gelu_new'
  layer_norm_eps: float = 1e-6

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.

    Returns:
      output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    x = nn.LayerNorm(epsilon=self.layer_norm_eps,
                     dtype=self.dtype,
                     pjit_axis_name=('embed',))(inputs)
    x = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads,
        in_proj_kernel_axes=('embed', 'heads', 'kv'),
        in_proj_bias_axes=('heads', 'kv'),
        out_proj_kernel_axes=('heads', 'kv', 'embed'),
        out_proj_bias_axes=('embed',),
        decode_axes=('batch', 'length', 'heads', 'kv'))(
            x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(epsilon=self.layer_norm_eps,
                     dtype=self.dtype,
                     pjit_axis_name=('embed',))(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate, hidden_act=self.hidden_act)(
            y, deterministic=deterministic)

    return x + y


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    num_layers: number of layers
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate in self attention.
  """

  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  add_position_embedding: bool = True
  dtype: Dtype = jnp.float32

  hidden_act: str = 'gelu_new'
  pre_layernorm: bool = False
  layer_norm_eps: float = 1e-6

  @nn.compact
  def __call__(self, x, *, train):
    """Applies Transformer model on the inputs.

    Args:
      x: Inputs to the layer.
      train: Set to `True` when training.

    Returns:
      output of a transformer encoder.
    """
    assert x.ndim == 3  # (batch, len, emb)

    if self.add_position_embedding:
      x = AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          dtype=self.dtype,
          name='posembed_input')(
              x)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    if self.pre_layernorm:
      x = nn.LayerNorm(epsilon=self.layer_norm_eps,
                       dtype=self.dtype,
                       pjit_axis_name=('embed',),
                       name='pre_layernorm')(x)

    # Input Encoder
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          num_heads=self.num_heads,
          dtype=self.dtype,
          hidden_act=self.hidden_act,
          layer_norm_eps=self.layer_norm_eps)(
              x, deterministic=not train)
    encoded = nn.LayerNorm(epsilon=self.layer_norm_eps,
                           dtype=self.dtype,
                           pjit_axis_name=('embed',),
                           name='encoder_norm')(x)
    return encoded


@dataclass
class TransformerEncoderSubConfig:
  mlp_dim: int
  num_heads: int
  num_layers: int
  attention_dropout_rate: float
  dropout_rate: float
  hidden_act: str = 'gelu_new'
  pre_layernorm: bool = False
  layer_norm_eps: float = 1e-6


@dataclass
class ResnetConfig:
  width_factor: int | None
  num_layers: list[int] | None

@dataclass
class PatchesConfig:
  size: tuple[int, ...]


def VisionTransformer(config: GoogleViTConfig, name=None):
  transformer = None
  if config.intermediate_size or config.num_attention_heads \
    or config.num_hidden_layers or config.attention_probs_dropout_prob \
      or config.hidden_dropout_prob:
    transformer = TransformerEncoderSubConfig(
      config.intermediate_size,
      config.num_attention_heads,
      config.num_hidden_layers,
      config.attention_probs_dropout_prob,
      config.hidden_dropout_prob,
      config.hidden_act,
      config.pre_layernorm,
      config.layer_norm_eps,
    )
  return _VisionTransformer(
    patches=PatchesConfig((config.patch_size, config.patch_size)),
    transformer=transformer,
    hidden_size=config.hidden_size,
    representation_size=config.representation_size,
    classifier=config.classifier,
    dtype=config.dtype,
    name=name,
  )

class _VisionTransformer(nn.Module):
  """VisionTransformer original, without the classification head"""
  patches: Any
  transformer: Any
  hidden_size: int
  representation_size: int | None = None
  classifier: str = 'token'
  encoder: type[nn.Module] = Encoder
  dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self,
               pixel_values,
               *,
               deterministic,
               return_dict=True):
    train = not deterministic

    x = pixel_values
    ### convert x to the appropriate dtype
    x = jnp.asarray(x, self.dtype)

    n, h, w, c = x.shape

    # We can merge s2d+emb into a single conv; it's the same.
    x = nn.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        dtype=self.dtype,
        kernel_axes=('height', 'width', 'input', 'embed'),
        bias_axes=('embed', ),
        padding='VALID',
        name='embedding')(
            x)
    # Here, x is a grid of embeddings.

    # (Possibly partial) Transformer.
    if self.transformer is not None:
      n, h, w, c = x.shape
      x = jnp.reshape(x, [n, h * w, c])

      # If we want to add a class token, add it here.
      if self.classifier in ['token', 'token_unpooled']:
        cls = param_with_axes('cls', nn.initializers.zeros, (c,), jnp.float32, axes=('embed',))
        cls = jnp.asarray(cls, self.dtype)
        cls = jnp.tile(cls, [n, 1, 1])
        x = jnp.concatenate([cls, x], axis=1)

      x = Encoder(name='Transformer', dtype=self.dtype, **asdict(self.transformer))(x, train=train)

    last_hidden_state = x

    if self.classifier == 'token':
      x = x[:, 0]
    elif self.classifier == 'gap':
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
    elif self.classifier in ['unpooled', 'token_unpooled']:
      pass
    else:
      raise ValueError(f'Invalid classifier={self.classifier}')

    if self.representation_size is not None:
      x = nn.Dense(features=self.representation_size, dtype=self.dtype, name='pre_logits', kernel_axes=('embed', 'mlp'), bias_axes=('mlp',))(x)
      x = nn.tanh(x)
    else:
      x = IdentityLayer(name='pre_logits')(x)

    return last_hidden_state, x


class VisionTransformerForImageClassification(nn.Module):
  """
  VisionTransformer with additional mlp(dense+tanh) and a final dense layer for classification

  In the original implementation, this was part of VisionTransformer, but it is separated here to be pluggable with CLIP
  """
  config: GoogleViTConfig

  @nn.compact
  def __call__(self,
               pixel_values,
               *,
               deterministic):

    outputs = VisionTransformer(self.config, name='VisionTransformer')(
      pixel_values,
      deterministic=deterministic,
    )

    x = outputs[1]

    if self.config.num_classes:
      x = nn.Dense(
          features=self.config.num_classes,
          name='head',
          kernel_init=nn.initializers.zeros,
          bias_init=nn.initializers.zeros,
          kernel_axes=('mlp', 'vocab'), bias_axes=('vocab',),
          dtype=self.config.dtype)(x)

    return x

class FlaxGViTModule(nn.Module):
    config: GoogleViTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.vision_model = VisionTransformer(self.config)

    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
    ):
        return self.vision_model(
            pixel_values=pixel_values,
            deterministic=deterministic,
        )

class FlaxGViTForImageClassificationModule(nn.Module):
    config: GoogleViTConfig

    def setup(self):
        self.vision_model = VisionTransformerForImageClassification(self.config)

    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
    ):
        return self.vision_model(
            pixel_values=pixel_values,
            deterministic=deterministic,
        )
