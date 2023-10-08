# Copyright (c) 2022-2023 NVIDIA Corporation
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

"Diffusion Efficient U-Net layers. Mostly used for superresolution"

# pylint: disable=attribute-defined-outside-init,g-bare-generic

#TODO Dropout
import dataclasses
import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np
from einops import rearrange

from t5x.contrib.gpu.t5.layers import MultiHeadDotProductAttention, make_attention_mask, combine_biases, dot_product_attention, RelativePositionBiases, MlpBlock
from flax.linen import DenseGeneral, GroupNorm, Conv, ConvTranspose, LayerNorm
from rosetta.projects.imagen.layers import FusedSelfCrossMultiHeadDotProductAttention

param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint

# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]
PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
LaxPadding = Union[str, Sequence[Tuple[int, int]]]
Dtype = Any

default_embed_init = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0)

dynamic_vector_slice_in_dim = jax.vmap(
    lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))

def _convert_to_activation_function(
    fn_or_string: Union[str, Callable]) -> Callable:
  """Convert a string to an activation function."""
  if fn_or_string == 'linear':
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError("don't know how to convert %s to an activation function" %
                     (fn_or_string,))

def FP32Wrap(operator, should_fp32=False):
    if should_fp32:
        def ret(x):
            h = jnp.asarray(x, dtype=jnp.float32)
            return jnp.asarray(operator(h), dtype=x.dtype)
        return ret
    else:
        return operator

class EfficientResBlock(nn.Module):
    """ Residual block in the style of Saharia et. al. from Imagen

    Attributes:
      out_channels:   Output channel count
      kernel_init:    Kernel init function for embedding layers and first conv
    """
    out_channels: int
    dtype: Any = jnp.float32
    norm_32: bool = False
    activation: str = 'silu'
    cond_strategy: str = 'shift_scale'
    dropout_rate: float = 0.0
    kernel_init: Optional[Initializer] = nn.initializers.lecun_normal()
    zero_out: bool = True
    scale_skip: bool = True

    @nn.compact
    def __call__(self, inputs,
                 conditioning,
                 deterministic: bool=False):
        """ Apply Efficient ResBlock.
        
        input shape:  B H W C_model
        conditioning: B C_cond_embedding
        """

        activation = _convert_to_activation_function(self.activation)
        spatial_conv = functools.partial(Conv,
                                         self.out_channels,
                                         kernel_size=(3,3),
                                         strides=(1,1),
                                         padding='SAME',
                                         dtype=self.dtype)

        # conditioning embedding calculation
        cond_dim = self.out_channels * (2 if self.cond_strategy == 'shift_scale' else 1)
        cond = activation(conditioning)
        cond = DenseGeneral(cond_dim,
                            axis=-1,
                            dtype=self.dtype,
                            use_bias=True,
                            kernel_axes=('embed',))(cond)
        cond = rearrange(cond, 'b c -> b 1 1 c')

        # in block
        # ensure input channels % 32 == 0
        h = FP32Wrap(GroupNorm(num_groups=32, name='resblock_pre_in_conv_groupnorm'), should_fp32=self.norm_32)(inputs) 
        h = activation(h)
        h = spatial_conv(kernel_init=self.kernel_init, name='resblock_in_conv')(h)

        # combine embedding + out_block
        out_norm = FP32Wrap(GroupNorm(num_groups=32, name='resblock_pre_out_conv_groupnorm'), should_fp32=self.norm_32)
        if self.cond_strategy == 'shift_scale':
            h = out_norm(h)
            # combine embedding
            shift, scale = jnp.split(cond, 2, axis=-1)
            h = h * (scale + 1) + shift
        elif self.cond_strategy == 'addition':
            h = h + cond
            h = out_norm(h)
        else:
            NotImplementedError(self.cond_strategy + " conditioning strategy not implemented.\
                                Use \'shift_scale\' or \'addition\' instead.")
        h = activation(h)
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
        out_init = nn.initializers.zeros if self.zero_out else self.kernel_init
        h = spatial_conv(kernel_init=out_init, name='resblock_out_conv')(h)

        # residual channel adjustment
        res = Conv(self.out_channels,
                   kernel_size=(1,1),
                   dtype=self.dtype,
                   kernel_init=self.kernel_init,
                   name='resblock_skip_conv')(inputs)

        # residual addition
        if self.scale_skip:
            res = res / jnp.sqrt(2)
        return h + res

class EfficientBlock(nn.Module):
    """
    Down and Up Block from Saharia et. al. Imagen Efficient U-Net.
    """
    out_channels: int
    dtype: Any = jnp.float32
    norm_32: bool = False
    activation: str = 'silu'
    cond_strategy: str = 'shift_scale'
    zero_out: bool = True
    scale_skip: bool = True
    dropout_rate: float = 0.0

    use_attn: bool = False
    attn_heads: int = 8
    mha_head_dim: int = 64
    attn_type: str = 'fused' # 'self', 'fused', or 'cross'

    up_down: str = 'down' # 'up', 'down' or None
    num_resblocks: int = 2
    strides: tuple = (1,1) 

    @nn.compact
    def __call__(self, 
                 inputs: Array,
                 conditioning: Array,
                 text_enc: Optional[Array]=None,
                 text_mask: Optional[Array]=None,
                 deterministic=False) -> Array:
        # dblock in conv
        if self.up_down == 'down':
            inputs = Conv(self.out_channels,
                          kernel_size=(3,3),
                          strides=self.strides,
                          padding='SAME',
                          dtype=self.dtype,
                          name='dblock_in_conv')(inputs)
        
        h = inputs
        for res_idx in range(self.num_resblocks):
            h = EfficientResBlock(out_channels=self.out_channels,
                                  dtype=self.dtype,
                                  norm_32=self.norm_32,
                                  activation=self.activation,
                                  cond_strategy=self.cond_strategy,
                                  dropout_rate=self.dropout_rate,
                                  zero_out=self.zero_out,
                                  scale_skip=self.scale_skip,
                                  name=f'resblock_{res_idx}')(h, conditioning=conditioning, deterministic=deterministic)
        if self.use_attn:
            h = ImgAttentionBlock(attn_heads=self.attn_heads,
                                  head_dim=self.mha_head_dim,
                                  attn_type=self.attn_type,
                                  dtype=self.dtype,
                                  dropout_rate=self.dropout_rate,
                                  scale_attn_logits=True,
                                  name='attention')(h, text_enc, text_mask, deterministic=deterministic)
        if self.up_down == 'up':
            h = ConvTranspose(self.out_channels, kernel_size=(3,3), strides=self.strides, padding='SAME', dtype=self.dtype, name='ublock_out_conv')(h)
        return h

class ImgAttentionBlock(nn.Module):
    """ Residual MHA block with normalization and reshaping for images
    and optional text conditioning.
    """
    norm_32: bool = False
    attn_heads: int = 8
    head_dim: int = 64
    attn_type: str = 'fused' # 'self', 'fused', or 'cross'
    float32_attention_logits: bool = False
    scale_attn_logits: bool = True
    zero_out: bool = True
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self,
                 inputs: Array, 
                 text_enc: Optional[Array]=None, 
                 text_mask: Optional[Array]=None,
                 deterministic=True) -> Array:
        """
        Applies self-attention to an image an optionally text encodings. Assumes text_enc
        has been normalized already.

        Args:
        inputs: Images [b, h, w, c]
        text_enc: Text encodings [b, l, c]
        text_mask: Array of text masks (for masking) [b, l]
        deterministic: Whether to enable dropout
        """
        x = rearrange(inputs, 'b h w c -> b (h w) c')
        x_q = FP32Wrap(GroupNorm(num_groups=32, name='img_attn_gn_q'), should_fp32=self.norm_32)(x)
        x_kv = FP32Wrap(GroupNorm(num_groups=32, name='img_attn_gn_kv'), should_fp32=self.norm_32)(x)

        mask = None
        if text_enc is not None:
            text_enc = FP32Wrap(GroupNorm(num_groups=32, name='text_enc_ln'), should_fp32=self.norm_32)(text_enc)
            if text_mask is None and text_enc is not None:
                text_mask = jnp.ones((text_enc.shape[0], text_enc.shape[1]))
        else:
            if self.attn_type == 'cross':
                raise ValueError('Cannot have both cross attention and no text conditioning.')
            if self.attn_type == 'fused':
                self.attn_type = 'self'

        m = jnp.ones((x.shape[0], x.shape[1]))
        q = x_q
        if self.attn_type == 'self':
            mask = make_attention_mask(m, m, dtype=self.dtype)
            kv_self = x_kv
            kv_cross = None
        elif self.attn_type == 'fused':
            mt = jnp.concatenate((m, text_mask), axis = 1)
            mask = make_attention_mask(m, mt, dtype=self.dtype)
            kv_self = x_kv
            kv_cross = text_enc
        elif self.attn_type == 'cross':
            mt = text_mask
            mask = make_attention_mask(m, mt, dtype=self.dtype)
            kv_self = text_enc
            kv_cross = None
        else:
            raise NotImplementedError(f'attention type {self.attn_type} is not implemented. Please choose from self, cross, and fused')

        x = FusedSelfCrossMultiHeadDotProductAttention(num_heads=self.attn_heads, 
                                         head_dim=self.head_dim,
                                         dtype=self.dtype,
                                         dropout_rate=self.dropout_rate,
                                         float32_logits=self.float32_attention_logits,
                                         scale_attn_logits=self.scale_attn_logits, 
                                         zero_out=self.zero_out,
                                         name='mha_layer')(q, kv_self, kv_cross, mask=mask, deterministic=deterministic)

        x = rearrange(x, 'b (h w) c -> b h w c', h=inputs.shape[1], w=inputs.shape[2])
        return x + inputs