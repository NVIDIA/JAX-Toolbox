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

"""Diffusion Model Backbones"""

from typing import Any, Sequence, Union, Iterable, Optional
import functools

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax import struct
import jax.numpy as jnp
from t5x.contrib.gpu.t5.layers import MultiHeadDotProductAttention, make_attention_mask
from flax.linen import DenseGeneral, Conv, LayerNorm, GroupNorm
import jax
from einops import rearrange

param_with_axes = nn_partitioning.param_with_axes
OptionalArray = Optional[jnp.ndarray]
Array = jnp.ndarray

default_embed_init = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0)

@struct.dataclass
class DiffusionConfig:
  dtype: Any = jnp.float32
  model_dim: int = 64
  attn_cond_dim: int = 512
  cond_dim: Optional[int] = None #timestep/pooled text embedding channels. Defaults to model_dim*4
  resblocks_per_level: Union[int, Sequence[int]] = 2
  width_multipliers: Sequence[int] = (1, 2, 3, 4)
  attn_resolutions: Sequence[int] = (32, 16, 8)
  mha_head_dim: int = 64
  attn_heads: Union[int, Sequence[int]] = 2
  unified_qkv_norm: bool = False
  deepfloyd_attnpooling: bool = False
  resblock_activation: str = 'swish'
  resblock_scale_skip: bool = False
  output_ch: int = 3
  dropout_rate: float = 0.1

  # modes are 'shuffle', 'conv' and 'resize'.
  #            (upsampling   , downsampling)
  # shuffle -> pixel shuffle / unshuffle
  # conv    -> resize + conv / strided conv
  # resize  -> interpolation / average pool
  upsample_mode: str = 'shuffle'
  downsample_mode: str = 'shuffle'

  # If True, block will use a 3x3 conv instead of 1x1 conv to correct 
  # channel dim mismatch in skip connection (if one exists)
  spatial_skip: bool = False

  # 'shift_scale' or 'addition'. Strategy for incorporating the conditioning vector.
  cond_strategy: str = 'shift_scale'

  # force groupnorm in fp32
  norm_32: bool = True

  # scale attention logits by \sqrt(head_dim)
  scale_attn_logits: bool = True
  float32_attention_logits: bool =False
  text_conditionable: bool = True
  null_text_emb_ct: int = 0

def single_or_idx(possibly_iter, idx):
    if isinstance(possibly_iter, Sequence):
        return possibly_iter[idx]
    else:
        return possibly_iter

class ImagenUNet(nn.Module):
  """ An Imagen diffusion U-net """
  config: DiffusionConfig

  @nn.compact
  def __call__(self, images, time,
               text_enc: OptionalArray=None,
               text_lens: OptionalArray=None,
               low_res_images: OptionalArray=None,
               noise_aug_level: OptionalArray=None,
               enable_dropout=True):
    """
    Args:
    images: samples to denoise. [b, h, w, c]
    time:   time conditioning. [b, 1]
    text_enc: text embeddings (required for text_conditionable) [b, seq_len, embed]
    text_lens: text sequence lengths in binary mask format      [b, seq_len]
    """
    from rosetta.projects.imagen import layers  # to avoid circular import
    cfg = self.config
    activation = layers._convert_to_activation_function(cfg.resblock_activation)
    deterministic=not enable_dropout
    linear = functools.partial(DenseGeneral,
                               axis=-1,
                               use_bias=True,
                               dtype=cfg.dtype,
                               kernel_axes=('embed',))

    spatial_conv = functools.partial(Conv,
                                     kernel_size=(3,3),
                                     strides=1,
                                     padding='SAME',
                                     dtype=cfg.dtype)
    input_ch = images.shape[-1]

    # create time embedding
    if cfg.cond_dim is None:
        time_embed_dim = cfg.model_dim * 4
    else:
        time_embed_dim = cfg.cond_dim
    cond_embed = layers.get_timestep_embedding(time, cfg.model_dim, dtype=jnp.float32)
    cond_embed = linear(features=time_embed_dim, name='time_dense_1')(cond_embed)
    cond_embed = activation(cond_embed)
    cond_embed = linear(features=time_embed_dim, name='time_dense_2')(cond_embed)

    if low_res_images is not None:
        print('Low res images available. Running as a superresolution network')
        if noise_aug_level is None:
            jax.debug.print('noise_aug not given but it *really* should be')
            noise_aug_level = jnp.ones_like(time) * (.25 * jnp.log(0.002)) # fallback EDM c_noise_fn on minimal noise
        aug_embed = layers.get_timestep_embedding(noise_aug_level, cfg.model_dim, dtype=jnp.float32)
        aug_embed = linear(features=time_embed_dim, name='aug_dense_1')(aug_embed)
        aug_embed = activation(aug_embed)
        aug_embed = linear(features=time_embed_dim, name='aug_dense_2')(aug_embed)
    
        cond_embed = cond_embed + aug_embed

        scaled_low_res = jax.image.resize(low_res_images, images.shape, "bicubic")
        images = jnp.concatenate([images, scaled_low_res], axis=-1)

    # create attn pooled text embedding and project text to cond_dim
    if cfg.text_conditionable:
        assert text_lens is not None, "text_lens cannot be None. If you're trying to null condition, pass in a 0 mask"
        assert text_enc is not None, "text_enc cannot be None. If you're trying to null condition, pass in 0s of appropriate shape. This network will add in the requisite null tokens"

        # setup null tokens
        if cfg.null_text_emb_ct > 0:
            null_text_mask = jnp.sum(text_lens, axis=1, keepdims=True)

            embedding_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0)
            null_text_embedding = param_with_axes(
                'null_text_embeddings',
                embedding_init,
                (cfg.null_text_emb_ct, text_enc.shape[-1]),
                jnp.float32,
                axes=('vocab', 'embed'))

            # first null_text_emb_ct tokens to null_text_embedding
            null_text_enc = jnp.zeros_like(text_enc[0:1]).at[:, :cfg.null_text_emb_ct].set(null_text_embedding)
            null_text_lens = jnp.zeros_like(text_lens[0:1]).at[:, :cfg.null_text_emb_ct].set(1)
            text_enc = jnp.where(jnp.expand_dims(null_text_mask, axis=-1) > 0, text_enc, null_text_enc)
            text_lens = jnp.where(null_text_mask > 0, text_lens, null_text_lens)

        # attention pooling
        if isinstance(cfg.attn_heads, Iterable):
            num_heads = text_enc.shape[-1] // cfg.mha_head_dim
        else:
            num_heads = None
            
        if cfg.deepfloyd_attnpooling:
            attn_pooled = LayerNorm(dtype=jnp.float32 if cfg.norm_32 else cfg.dtype, name='attn_pool_ln_0')(text_enc)
            attn_pooled = layers.DeepFloydAttentionPoolingBlock(cfg=cfg, num_heads=num_heads)(attn_pooled, text_lens)
            attn_pooled = linear(features=cfg.attn_cond_dim, name='attn_pool_dense_0')(attn_pooled)
            attn_pooled = LayerNorm(dtype=jnp.float32 if cfg.norm_32 else cfg.dtype, name='attn_pool_ln_1')(attn_pooled)
        else:
            attn_pooled = layers.AttentionPoolingBlock(cfg=cfg, num_heads=num_heads)(text_enc, text_lens)
            attn_pooled = LayerNorm(dtype=jnp.float32 if cfg.norm_32 else cfg.dtype, name='attn_pool_ln_0')(attn_pooled)
            attn_pooled = linear(features=cfg.attn_cond_dim, name='attn_pool_dense_0')(attn_pooled)
            attn_pooled = LayerNorm(dtype=jnp.float32 if cfg.norm_32 else cfg.dtype, name='attn_pool_ln_1')(attn_pooled)
            attn_pooled = linear(features=time_embed_dim, name='attn_pool_dense_1')(attn_pooled)
            attn_pooled = activation(attn_pooled)
            attn_pooled = linear(features=time_embed_dim, name='attn_pool_dense_2')(attn_pooled)

        cond_embed = attn_pooled + cond_embed # Has dimension of time_embed_dim
 
        # text embedding projection to cond_dim
        text_enc = linear(features=cfg.attn_cond_dim, name='text_enc_projection')(text_enc)

    # make image_channel -> model_dim convolution
    x = spatial_conv(features=cfg.model_dim * cfg.width_multipliers[0], name='unet_in_conv')(images)

    # down branch resblocks + attn on designated resolutions
    down_outputs = [x]
    for level, width_mult in enumerate(cfg.width_multipliers):
        level_channels = cfg.model_dim * width_mult
        img_res = images.shape[1] // (2 ** level)

        for res_idx in range(single_or_idx(cfg.resblocks_per_level, level)):
            x = layers.ResBlock(cfg=cfg, out_channels=level_channels, up_down_sample='none',
                                name='resblock_enc_{}_{}'.format(img_res, res_idx)) \
                                (x, cond_embed, deterministic=deterministic)
            print("Encoder ResBlock #", res_idx, " at resolution: ", img_res, " level: ", level, " width: ", level_channels)

            # attend if image is of designated resolution
            if img_res in cfg.attn_resolutions:
                attn_idx = level - (len(cfg.width_multipliers) - len(cfg.attn_resolutions))
                x = layers.ImgAttentionBlock(cfg=cfg, num_heads=single_or_idx(cfg.attn_heads, attn_idx), name='attnblock_enc_{}_{}'.format(img_res, res_idx)) \
                                            (x, text_enc, text_lens, deterministic=deterministic)
                print("SelfAttentionBlock #", res_idx, " at resolution: ", img_res, " level: ", level, " width: ", level_channels)

            down_outputs.append(x)

        # if not on the last level, downsample
        if level != len(cfg.width_multipliers) - 1:
            x = layers.ResBlock(cfg=cfg, out_channels=level_channels, up_down_sample='down',
                                name='resblock_enc_downsampling_{}_{}'.format(img_res, res_idx)) \
                                (x, cond_embed, deterministic=deterministic)
            print("Downsampling ResBlock at resolution (to half): ", img_res, " level: ", level, " width: ", level_channels)
            down_outputs.append(x)

    # middle layers
    mid_channels = cfg.model_dim * cfg.width_multipliers[-1]
    x = layers.ResBlock(cfg=cfg, out_channels=mid_channels, up_down_sample='none',
                        name='resblock_mid_1')(x, cond_embed, deterministic=deterministic)
    x = layers.ImgAttentionBlock(cfg=cfg, num_heads=single_or_idx(cfg.attn_heads, -1),name='attnblock_mid_1')(x, text_enc, text_lens, deterministic=deterministic)
    x = layers.ResBlock(cfg=cfg, out_channels=mid_channels, up_down_sample='none',
                        name='resblock_mid_2')(x, cond_embed, deterministic=deterministic)

    print('Encoder Skip Shapes: ',list(map(lambda x : x.shape, down_outputs)))
    # up branch resblocks + attn on designated resolutions + skip connections
    for level, width_mult in list(enumerate(cfg.width_multipliers))[::-1]:
        level_channels = cfg.model_dim * width_mult
        img_res = images.shape[1] // (2 ** level)

        for res_idx in range(single_or_idx(cfg.resblocks_per_level, level) + 1):
            u_skip = down_outputs.pop()
            print("Decoder ResBlock #", res_idx, " at resolution: ", img_res, " level: ", level, " width: ", level_channels, " skip shape: ", u_skip.shape)
            x = jnp.concatenate([x, u_skip], axis=-1)
            x = layers.ResBlock(cfg=cfg, out_channels=level_channels, up_down_sample='none',
                                name='resblock_dec_{}_{}'.format(img_res, res_idx)) \
                                (x, cond_embed, deterministic=deterministic)

            # attend if image is of designated resolution
            if img_res in cfg.attn_resolutions:
                print("SelfAttentionBlock #", res_idx, " at resolution: ", img_res, " level: ", level, " width: ", level_channels)
                attn_idx = level - (len(cfg.width_multipliers) - len(cfg.attn_resolutions))
                x = layers.ImgAttentionBlock(cfg=cfg, num_heads=single_or_idx(cfg.attn_heads, attn_idx), name='attnblock_dec_{}_{}'.format(img_res, res_idx))\
                                            (x, text_enc, text_lens, deterministic=deterministic)

            # upsample if on last resblock and not the highest level
            if res_idx == single_or_idx(cfg.resblocks_per_level, level) and level != 0:
                print("Upsamling ResBlock at resolution (to double): ", img_res, " level: ", level, " width: ", level_channels, " no skip")
                x = layers.ResBlock(cfg=cfg, out_channels=level_channels, up_down_sample='up',
                                    name='resblock_dec_upsampling_{}_{}'.format(img_res, res_idx)) \
                                    (x, cond_embed, deterministic=deterministic)

    # out convolution model_dim -> image_channels
    x = layers.FP32Wrap(GroupNorm(num_groups=32, name='unet_out_gn'))(x)
    x = activation(x)
    # x = spatial_conv(features=cfg.output_ch, name='unet_out_conv', kernel_init=nn.initializers.zeros, no_embed_axis=True)(x)
    x = spatial_conv(features=cfg.output_ch, name='unet_out_conv', kernel_init=nn.initializers.zeros)(x)
    return x