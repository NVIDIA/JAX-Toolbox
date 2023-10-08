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

"Diffusion U-Net layers"

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
from flax.linen import DenseGeneral, GroupNorm, Conv, LayerNorm
from rosetta.projects.imagen.network import DiffusionConfig

param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint

# Type annotations
Array = jnp.ndarray
OptionalArray = Optional[jnp.ndarray]
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

def get_timestep_embedding(timesteps, embedding_dim: int, dtype=jnp.float32):
  """Build sinusoidal embeddings
  Args:
    timesteps: jnp.ndarray: generate embedding vectors at these timesteps
    embedding_dim: int: dimension of the embeddings to generate
    dtype: data type of the generated embeddings
  Returns:
    embedding vectors with shape `[len(timesteps), embedding_dim]`
  """
  timesteps = jnp.reshape(timesteps, timesteps.shape[0])
  assert len(timesteps.shape) == 1, "timesteps don't have one dimension, " + str(timesteps.shape)
  half = embedding_dim // 2
  freqs = jnp.exp(
          -np.log(10000) * jnp.arange(0, half, dtype=dtype) / half
          )
  args = timesteps[:, None] * freqs[None]
  embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

  if embedding_dim % 2 == 1:  # zero pad
    embedding = jax.lax.pad(embedding, dtype(0), ((0, 0, 0), (0, 1, 0)))
  assert embedding.shape == (timesteps.shape[0], embedding_dim)

  return embedding

def FP32Wrap(operator, should_fp32=False):
    if should_fp32:
        def ret(x):
            h = jnp.asarray(x, dtype=jnp.float32)
            return jnp.asarray(operator(h), dtype=x.dtype)
        return ret
    else:
        return operator

class FusedSelfCrossMultiHeadDotProductAttention(nn.Module):
  """Fused self attention with cross attention for image-text self and cross attn.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      head_dim: dimension of each head.
      dtype: the dtype of the computation.
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
      zero_out: bool. Will initialize out projection to zero if true
  """

  num_heads: int
  head_dim: int
  dtype: DType = jnp.float32
  dropout_rate: float = 0.
  kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'normal')
  float32_logits: bool = False  # computes logits in float32 for stability.
  scale_attn_logits: bool = False
  zero_out: bool = True
  project_output: bool = True

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv_self: Array,
               inputs_kv_cross: Optional[Array] = None,
               mask: Optional[Array] = None,
               bias: Optional[Array] = None,
               *,
               deterministic: bool = False) -> Array:
    """Applies self and cross attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape `[batch, q_length, q_features]`.
      inputs_kv_self: key/values of shape `[batch, kv_self_length, kv_self_features]`.
      inputs_kv_cross: key/values of shape `[batch, kv_cross_length, kv_cross_features]'
      mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
      bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
      decode: Whether to prepare and use an autoregressive cache.
      deterministic: Disables dropout if set to True.

    Returns:
      output of shape `[batch, length, q_features]`.
    """
    projection = functools.partial(
        DenseGeneral,
        axis=-1,
        features=(self.num_heads, self.head_dim),
        kernel_axes=('embed', 'joined_kv'),
        use_bias=True,
        dtype=self.dtype)

    # NOTE: T5 does not explicitly rescale the attention logits by
    #       1/sqrt(depth_kq)!  This is folded into the initializers of the
    #       linear transformations, which is equivalent under Adafactor.
    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    query_init = lambda *args: self.kernel_init(*args) / depth_scaling

    # Project inputs_q to multi-headed q/k/v
    # dimensions are then [batch, length, num_heads, head_dim]
    query = projection(kernel_init=query_init, name='query')( \
            (inputs_q / depth_scaling) if self.scale_attn_logits else inputs_q)
    key = projection(kernel_init=self.kernel_init, name='key')(inputs_kv_self)
    value = projection(kernel_init=self.kernel_init, name='value')(inputs_kv_self)

    query = with_sharding_constraint(query, ('batch', 'length', 'heads', 'kv'))

    if inputs_kv_cross is not None:
        key_cross = projection(kernel_init=self.kernel_init, name='key_cross')(inputs_kv_cross)
        value_cross = projection(kernel_init=self.kernel_init, name='value_cross')(inputs_kv_cross)
        
        # Concatenate on length axis
        key = jnp.concatenate((key, key_cross), axis=1)
        value = jnp.concatenate((value, value_cross), axis = 1)

    key = with_sharding_constraint(key, ('batch', 'length', 'heads', 'kv'))
    value = with_sharding_constraint(value, ('batch', 'length', 'heads', 'kv'))

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype))
    else:
      attention_bias = None

    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = combine_biases(attention_bias, bias)

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # Apply attention.
    x = dot_product_attention(
        query,
        key,
        value,
        bias=attention_bias,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        deterministic=deterministic,
        dtype=self.dtype,
        float32_logits=self.float32_logits)

    if self.project_output:
      # Back to the original inputs dimensions.
      out = DenseGeneral(
          features=inputs_q.shape[-1],  # output dim is set to the input dim.
          axis=(-2, -1),
          kernel_init=self.kernel_init if not self.zero_out else nn.initializers.zeros,
          kernel_axes=('joined_kv', 'embed'),
          use_bias=True,
          dtype=self.dtype,
          name='out')(
              x)
      return out
    else:
      return x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))

class AttentionPoolingBlock(nn.Module):
    """
    Attention Pooling via self+cross attention with the mean. Assumes inputs 
    are normalized already. Uses RelativePositionBiases.

    Attributes:
    cfg: Model-wide configuration. This will use the same parameters 
         as all other attention layers.
    num_heads: optional override on the number of heads in this layer
    """
    cfg: DiffusionConfig
    num_heads: Optional[int] = None

    @nn.compact
    def __call__(self,
                 inputs: Array,
                 text_lens: Optional[Array] = None,
                 *,
                 deterministic: bool = False) -> Array:
      """ Performs attention pooling by doing cross attention between mean embedding and 
      all tokens.
  
      Args:
        inputs: input sequence of shape `[batch, seq_length, features]`.
        text_lens: Array of text masks (for masking) [batch, seq_length]
        bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
        deterministic: Disables dropout if set to True.
  
      Returns:
        output of shape `[batch, features]`.
      """
      cfg = self.cfg
      num_heads = self.num_heads if self.num_heads is not None else cfg.attn_heads

      pos_bias = RelativePositionBiases(
          num_buckets=32,
          max_distance=128,
          num_heads=num_heads,
          dtype=cfg.dtype,
          embedding_init=nn.initializers.variance_scaling(1.0, 'fan_avg',
                                                          'uniform'),
          name='relpos_bias')(1, inputs.shape[1] + 1, False)

      mask = None
      if text_lens is not None:
          m = jnp.ones((inputs.shape[0], 1))
          mt = jnp.concatenate((m, text_lens), axis = 1)
          mask = make_attention_mask(m, mt, dtype=cfg.dtype)

      text_mask = rearrange(text_lens, 'b l -> b l 1')
      masked_inputs = inputs * text_mask
      mean = jnp.mean(masked_inputs, axis=1, keepdims=True) # [batch, 1, q_features]
      out = MultiHeadDotProductAttention(num_heads=num_heads,
                                         head_dim=cfg.mha_head_dim,
                                         dtype=cfg.dtype,
                                         dropout_rate=cfg.dropout_rate,
                                         float32_logits=cfg.float32_attention_logits,
                                         scale_attn_logits=cfg.scale_attn_logits) \
                                         (mean, jnp.concatenate((mean, inputs), axis=1), mask=mask, bias=pos_bias, deterministic=deterministic)
      out = rearrange(out, 'b l e -> b (l e)') # l should equal 1
      assert out.shape[1] == inputs.shape[2]
      return out

class DeepFloydAttentionPoolingBlock(nn.Module):
    """
    Attention Pooling via self+cross attention with the mean. Assumes inputs 
    are normalized already. Uses Absolute Position Embeddings.

    Attributes:
    cfg: Model-wide configuration. This will use the same parameters 
         as all other attention layers.
    num_heads: optional override on the number of heads in this layer
    """
    cfg: DiffusionConfig
    num_heads: Optional[int] = None

    @nn.compact
    def __call__(self,
                 inputs: Array,
                 text_lens: Optional[Array] = None,
                 *,
                 deterministic: bool = False) -> Array:
      """ Performs attention pooling by doing cross attention between mean embedding and 
      all tokens.
  
      Args:
        inputs: input sequence of shape `[batch, seq_length, features]`.
        text_lens: Array of text masks (for masking) [batch, seq_length]
        bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
        deterministic: Disables dropout if set to True.
  
      Returns:
        output of shape `[batch, features]`.
      """
      cfg = self.cfg
      num_heads = self.num_heads if self.num_heads is not None else cfg.attn_heads

      embedding_init = nn.initializers.normal(stddev=jnp.sqrt(inputs.shape[2]))
      pos_bias = param_with_axes('position_embed', embedding_init, (1, inputs.shape[2],), jnp.float32, axes=('empty', 'embed'))
      pos_bias = jnp.asarray(pos_bias, dtype=inputs.dtype)
      pos_bias = rearrange(pos_bias, '1 e -> 1 1 e')
      mask = None
      if text_lens is not None:
          m = jnp.ones((inputs.shape[0], 1))
          mt = jnp.concatenate((m, text_lens), axis = 1)
          mask = make_attention_mask(m, mt, dtype=cfg.dtype)

      text_mask = rearrange(text_lens, 'b l -> b l 1')
      masked_inputs = inputs * text_mask
      mean = jnp.mean(masked_inputs, axis=1, keepdims=True) + pos_bias # [batch, 1, q_features]
      out = FusedSelfCrossMultiHeadDotProductAttention(num_heads=num_heads,
                                         head_dim=cfg.mha_head_dim,
                                         dtype=cfg.dtype,
                                         dropout_rate=cfg.dropout_rate,
                                         float32_logits=cfg.float32_attention_logits,
                                         scale_attn_logits=cfg.scale_attn_logits,
                                         project_output=False) \
                                         (mean, jnp.concatenate((mean, inputs), axis=1), None, mask=mask, bias=None, deterministic=deterministic)
      out = rearrange(out, 'b l e -> b (l e)') # l should equal 1
      assert out.shape[1] == inputs.shape[2]
      return out


class ImgAttentionBlock(nn.Module):
    """ Residual MHA block with normalization and reshaping for images
    and optional text conditioning.

    cfg: DiffusionConfig
    num_heads: Optional number of heads for attention. Will default to
               cfg.attn_heads
    """
    cfg: DiffusionConfig
    num_heads: Optional[int]

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

        cfg = self.cfg
        x = rearrange(inputs, 'b h w c -> b (h w) c')
        if cfg.unified_qkv_norm:
          x_q = FP32Wrap(GroupNorm(num_groups=32, name='img_attn_gn'), should_fp32=cfg.norm_32)(x)
          x_kv = x_q
        else:
          x_q = FP32Wrap(GroupNorm(num_groups=32, name='img_attn_gn_q'), should_fp32=cfg.norm_32)(x)
          x_kv = FP32Wrap(GroupNorm(num_groups=32, name='img_attn_gn_kv'), should_fp32=cfg.norm_32)(x)

        mask = None
        if text_enc is not None:
            text_enc = FP32Wrap(GroupNorm(num_groups=32, name='text_enc_ln'), should_fp32=cfg.norm_32)(text_enc)
            if text_mask is not None:
                m = jnp.ones((x.shape[0], x.shape[1]))
                mt = jnp.concatenate((m, text_mask), axis = 1)
                mask = make_attention_mask(m, mt, dtype=cfg.dtype)

        num_heads = self.num_heads if self.num_heads is not None else cfg.attn_heads
        x = FusedSelfCrossMultiHeadDotProductAttention(num_heads=num_heads, 
                                         head_dim=cfg.mha_head_dim,
                                         dtype=cfg.dtype,
                                         dropout_rate=cfg.dropout_rate,
                                         float32_logits=cfg.float32_attention_logits,
                                         scale_attn_logits=cfg.scale_attn_logits, 
                                         name='mha_layer')(x_q, x_kv, text_enc, mask=mask, deterministic=deterministic)

        x = rearrange(x, 'b (h w) c -> b h w c', h=inputs.shape[1], w=inputs.shape[2])
        return x + inputs

def identity(x: Array) -> Array:
    return x
    
class ResBlock(nn.Module):
    """ Residual block in the style of Nichol et. al. Used in a UNet 

    Attributes:
      cfg:            DiffuionConfig
      out_channels:   Output channel count
      up_down_sample: 'up', 'down', or 'none'. Sets if upscaling, downscaling or neither
      kernel_init:    Kernel init function for embedding layers and first conv
    """
    cfg: DiffusionConfig
    out_channels: int
    up_down_sample: str = 'none'
    kernel_init: Optional[Initializer] = nn.initializers.lecun_normal()

    def _get_scaling_block(self, up_down:str):
        cfg = self.cfg
        if up_down == 'up':
            return Upsample(mode=self.cfg.upsample_mode,
                            kernel_init=self.kernel_init,
                            dtype=cfg.dtype,
                            norm_32=self.cfg.norm_32)
        elif up_down == 'down':
            return Downsample(mode=self.cfg.downsample_mode,
                              kernel_init=self.kernel_init,
                              dtype=cfg.dtype)
        elif up_down == 'none':
            return identity
        else:
            raise ValueError(f'Attempting to construct a resblock with up_down \
                    type {up_down}. Please use one of \'up\', \'down\', or \'none\'')

    @nn.compact
    def __call__(self, inputs,
                 conditioning,
                 deterministic: bool=False):
        """ Apply ResBlock.
        
        input shape:  B H W C_model
        conditioning: B C_cond_embedding
        """
        cfg = self.cfg

        # normalization = GroupNormFP32 if cfg.norm_32 else GroupNorm
        activation = _convert_to_activation_function(cfg.resblock_activation)
        spatial_scaling_fn = self._get_scaling_block(self.up_down_sample)
        spatial_conv = functools.partial(Conv,
                                         self.out_channels,
                                         kernel_size=(3,3),
                                         strides=1,
                                         padding='SAME',
                                         dtype=cfg.dtype)

        # conditioning embedding calculation
        cond_dim = self.out_channels * (2 if cfg.cond_strategy == 'shift_scale' else 1)
        cond = activation(conditioning)
        cond = DenseGeneral(cond_dim,
                            axis=-1,
                            dtype=cfg.dtype,
                            use_bias=True,
                            kernel_axes=('embed',))(cond)
        cond = rearrange(cond, 'b c -> b 1 1 c')

        # spatial scaling residual
        res = spatial_scaling_fn(inputs)

        # in block
        # ensure input channels % 32 == 0
        h = FP32Wrap(GroupNorm(num_groups=32, name='resblock_pre_in_conv_groupnorm'), should_fp32=cfg.norm_32)(inputs) 
        h = activation(h)
        h = spatial_scaling_fn(h)
        h = spatial_conv(kernel_init=self.kernel_init, name='resblock_in_conv')(h)
        
        # combine embedding + out_block
        out_norm = FP32Wrap(GroupNorm(num_groups=32, name='resblock_pre_out_conv_groupnorm'), should_fp32=cfg.norm_32)
        if cfg.cond_strategy == 'shift_scale':
            h = out_norm(h)
            # combine embedding
            shift, scale = jnp.split(cond, 2, axis=-1)
            h = h * (scale + 1) + shift
        elif cfg.cond_strategy == 'addition':
            h = h + cond
            h = out_norm(h)
        else:
            NotImplementedError(cfg.cond_strategy + " conditioning strategy not implemented.\
                                Use \'shift_scale\' or \'addition\' instead.")
        h = activation(h)
        h = nn.Dropout(rate=cfg.dropout_rate)(h, deterministic=deterministic)
        h = spatial_conv(kernel_init=nn.initializers.zeros, name='resblock_out_conv')(h)

        # residual channel adjustment
        if self.out_channels != inputs.shape[-1]:
            if cfg.spatial_skip:
                res = spatial_conv(kernel_init=self.kernel_init, name='resblock_skip_conv')(res)
            else:
                res = Conv(self.out_channels,
                              kernel_size=(1,1),
                              dtype=cfg.dtype,
                              kernel_init=self.kernel_init,
                              name='resblock_skip_conv')(res)

        # residual addition
        out_sum = h + res
        if cfg.resblock_scale_skip:
            return out_sum * .7071 # 1/sqrt(2)
        else:
            return out_sum

def _pixel_shuffle_kernel_init(key, shape, dtype, window=2):
    """
    Conv kernel init with replication over the shuffled axis.
    Replicated such that initial initial upscales will be like interpolated onces.
    """
    h, w, i, o = shape
    jax.debug.print('Conv kernel shape: ', str(shape))
    partial_shape = h, w, i, o // (window ** 2)
    init = nn.initializers.kaiming_uniform()(key, partial_shape, dtype)
    repl = jnp.repeat(init, window ** 2, axis=3) # H, W, I, O
    return repl

class Upsample(nn.Module):
    """
    Upsampling module done optionally with convolution
  
    Attributes:
    scaling_factor: Defaults to 2. Identical for all axes
    mode: 'shuffle': pixel shuffle
          'conv'   : interpolate -> 3x3 convolution
          'resize' : interpolated scaling
    kernel_init: conv kernel init
    dtype: conv dtype
    """
    scaling_factor: int = 2
    mode: str = 'shuffle'
    kernel_init: Optional[Initializer] = nn.initializers.lecun_normal()
    dtype: Any = jnp.float32
    norm_32: bool = True
  
    @nn.compact
    def __call__(self, x: Array) -> Array:
        """ Upscales input by self.scaling_factor in HW dims assuming NHWC format """
        in_ch = x.shape[-1] 
  
        if self.mode == 'resize' or self.mode == 'conv':
            n, h, w, c = x.shape
            h1 = h * self.scaling_factor
            w1 = w * self.scaling_factor
            x = jax.image.resize(x, (n, h1, w1, c), method='bilinear')
            if self.mode == 'resize':
                return x # early return for simple interpolation
  
            kernel=(3,3)
            out_ch=x.shape[-1]
            
        elif self.mode == 'shuffle':
            kernel=(1,1)
            out_ch=x.shape[-1] * self.scaling_factor ** 2
  
        else:
            ValueError("Upsample mode must be \'resize\',\'conv\', or \
                        \'shuffle\'. " + self.mode + " is not supported.")
            exit()
  
        # 'conv'        -> out_ch=in_ch, kernel=3 
        # 'pix_shuffle' -> out_ch=in_ch * scaling_factor **2, kernel=1
        x = Conv(out_ch,
                 kernel_size=kernel,
                 strides=1,
                 dtype=self.dtype,
                 kernel_init=self.kernel_init,
                 name='upsample_convolution')(x)
  
        if self.mode == 'shuffle':
            x = FP32Wrap(GroupNorm(num_groups=32, name='pix_shuffle_gn'), should_fp32=self.norm_32)(x)
            x = _convert_to_activation_function('silu')(x)
  
            # shifting channel dims into square spatial pixel dims
            return rearrange(x, 'b h w (s1 s2 c) -> b (h s1) (w s2) c', \
                             s1=self.scaling_factor, s2=self.scaling_factor)
        else: #mode == 'conv'
            return x
   

class Downsample(nn.Module):
    """
    Downsampling module done optionally with convolution
  
    Attributes:
    scaling_factor: Defaults to 2. Identical for all axes
    mode: 'shuffle': SP-conv from: https://arxiv.org/pdf/2208.03641.pdf. 
                     Basically pixel unshuffle
          'conv'   : strided convolution downsampling
          'resize' : average pooling
    kernel_init: conv kernel init
    dtype: conv dtype
    """
    scaling_factor: int = 2
    mode: str = 'shuffle'
    kernel_init: Optional[Initializer] = nn.initializers.lecun_normal()
    dtype: Any = jnp.float32
  
    @nn.compact
    def __call__(self, x:Array) -> Array:
        channels = x.shape[-1]
        if self.mode == 'resize':
            window_tuple = (self.scaling_factor, self.scaling_factor)
            return nn.avg_pool(x, window_tuple, window_tuple)
        elif self.mode == 'conv':
            kernel_size = (3,3)
            stride=self.scaling_factor
            padding = 1
        elif self.mode == 'shuffle':
            kernel_size = (1,1)
            stride=1
            padding = 0
            x = rearrange(x, 'b (h s1) (w s2) c -> b h w (s1 s2 c)', 
                          s1 = self.scaling_factor, s2 = self.scaling_factor)
        else:
            raise ValueError('Downsampling mode must be \'resize\', \'conv\',\
                              or \'shuffle\'. ' + self.mode + " not supported")
  
        return Conv(channels,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    dtype=self.dtype,
                    kernel_init=self.kernel_init,
                    name='downsample_convolution')(x)