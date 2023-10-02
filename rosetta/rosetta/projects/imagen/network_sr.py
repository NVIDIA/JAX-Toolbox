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

"""Super resolution Diffusion Model Backbones"""

from typing import Any, Sequence, Union, Iterable, Optional, Mapping
import functools

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax import struct
import jax.numpy as jnp
from t5x.contrib.gpu.t5.layers import MultiHeadDotProductAttention, make_attention_mask
from flax.linen import DenseGeneral, Conv, LayerNorm
import jax
from einops import rearrange

param_with_axes = nn_partitioning.param_with_axes
OptionalArray = Optional[jnp.ndarray]
Array = jnp.ndarray

default_embed_init = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0)

def single_or_idx(possibly_iter, idx):
    if isinstance(possibly_iter, Iterable):
        return possibly_iter[idx]
    else:
        return possibly_iter

@struct.dataclass
class ImagenEfficientUNetConfig:
  dtype: Any = jnp.float32
  model_dim: int = 64
  cond_dim: int = 256 #timestep/pooled text embedding channels
  attn_cond_dim: int = 256
  resblocks_per_level: Union[int, Iterable[int]] = (2, 4, 8, 8)
  width_multipliers: Iterable[int] = (1, 2, 3, 4)
  attn_resolutions_divs: Mapping[int, str] = None #{8: 'fused'} #use attn at resolutions of input / {elems of list}
  mha_head_dim: int = 64
  attn_heads: Union[int, Sequence[int]] = 8
  resblock_activation: str = 'swish'
  resblock_zero_out: bool = True
  resblock_scale_skip: bool = True # enables scaling residuals by 1/\sqrt{2}
  dropout_rate: float = 0.1

  # 'shift_scale' or 'addition'. Strategy for incorporating the conditioning vector.
  cond_strategy: str = 'shift_scale'

  # force groupnorm in fp32
  norm_32: bool = True

  # scale attention logits by \sqrt(head_dim)
  scale_attn_logits: bool = True
  float32_attention_logits: bool =False
  text_conditionable: bool = True
  null_text_emb_ct: int = 0
 
class ImagenEfficientUNet(nn.Module):
  """ An Imagen diffusion U-net """
  config: ImagenEfficientUNetConfig

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
    from rosetta.projects.imagen import layers_sr  # to avoid circular import
    from rosetta.projects.imagen import layers  # to avoid circular import
    cfg = self.config
    activation = layers_sr._convert_to_activation_function(cfg.resblock_activation)
    deterministic=not enable_dropout
    linear = functools.partial(DenseGeneral,
                               axis=-1,
                               use_bias=True,
                               dtype=cfg.dtype,
                               kernel_axes=('embed',))

    spatial_conv = functools.partial(Conv,
                                     kernel_size=(3,3),
                                     strides=(1,1),
                                     padding='SAME',
                                     dtype=cfg.dtype)
    input_ch = images.shape[-1]

    # create time embedding
    time_embed_dim = cfg.cond_dim
    cond_embed = layers.get_timestep_embedding(time, cfg.model_dim, dtype=jnp.float32)
    cond_embed = linear(features=time_embed_dim, name='time_dense_1')(cond_embed)
    cond_embed = activation(cond_embed)
    cond_embed = linear(features=time_embed_dim, name='time_dense_2')(cond_embed)

    if low_res_images is not None:
        print('Low res images available. Running as a superresolution network')
        if noise_aug_level is None:
            print('noise_aug not given but it *really* should be')
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
        attn_pooled = layers.AttentionPoolingBlock(cfg=cfg)(text_enc, text_lens)
        attn_pooled = LayerNorm(dtype=jnp.float32 if cfg.norm_32 else cfg.dtype, name='attn_pool_ln_0')(attn_pooled)
        attn_pooled = linear(features=cfg.cond_dim, name='attn_pool_dense_0')(attn_pooled)
        attn_pooled = LayerNorm(dtype=jnp.float32 if cfg.norm_32 else cfg.dtype, name='attn_pool_ln_1')(attn_pooled)
        attn_pooled = linear(features=time_embed_dim, name='attn_pool_dense_1')(attn_pooled)
        attn_pooled = activation(attn_pooled)
        attn_pooled = linear(features=time_embed_dim, name='attn_pool_dense_2')(attn_pooled)

        cond_embed = attn_pooled + cond_embed # Has dimension of time_embed_dim
 
        # text embedding projection to cond_dim
        text_enc = linear(features=cfg.attn_cond_dim, name='text_enc_projection')(text_enc)

    # make image_channel -> model_dim convolution (and/or CrossEmbedLayer inception style?)
    x = spatial_conv(features=cfg.model_dim * cfg.width_multipliers[0], name='unet_in_conv')(images)

    # down branch resblocks + attn on designated resolutions
    down_outputs = []
    use_attn, attn_type = None, None

    common_block_kwargs = {
        'dtype':cfg.dtype,
        'norm_32':cfg.norm_32,
        'activation':cfg.resblock_activation,
        'cond_strategy':cfg.cond_strategy,
        'dropout_rate':cfg.dropout_rate,
        'attn_heads':cfg.attn_heads,
        'mha_head_dim':cfg.mha_head_dim,
        'zero_out':cfg.resblock_zero_out,
        'scale_skip':cfg.resblock_scale_skip,
    }

    resolution_div = 1
    for level, width_mult in enumerate(cfg.width_multipliers[:-1]):
        level_channels = cfg.model_dim * width_mult

        use_attn = (2 ** level) in cfg.attn_resolutions_divs.keys()
        attn_type = cfg.attn_resolutions_divs[2 ** level] if use_attn else None
        x = layers_sr.EfficientBlock(out_channels=level_channels,
                                     use_attn=use_attn,
                                     attn_type=attn_type,
                                     up_down='down',
                                     num_resblocks=single_or_idx(cfg.resblocks_per_level, level),
                                     strides=(2,2),
                                     name=f'DBlock_l{level}',
                                     **common_block_kwargs) \
                                     (x, cond_embed, text_enc, text_lens, deterministic=deterministic)
        print("Encoder DBlock #", level, " out resolution: ", x.shape[1:3], " level: ", level, " width: ", level_channels)
        down_outputs.append(x)
        resolution_div = 2 ** level

    # middle layers 
    use_attn = (2 * resolution_div) in cfg.attn_resolutions_divs.keys()
    attn_type = cfg.attn_resolutions_divs[2 ** (level + 1)] if use_attn else None
    mid_channels = cfg.model_dim * cfg.width_multipliers[-1]
    x = layers_sr.EfficientBlock(out_channels=mid_channels,
                                 use_attn=use_attn,
                                 attn_type=attn_type,
                                 up_down='down',
                                 num_resblocks=single_or_idx(cfg.resblocks_per_level, -1),
                                 strides=(1,1),
                                 name='dblock_mid',
                                 **common_block_kwargs) \
                                (x, cond_embed, text_enc, text_lens, deterministic=deterministic)

    x = layers_sr.EfficientBlock(out_channels=mid_channels,
                                 use_attn=use_attn,
                                 attn_type=attn_type,
                                 up_down='up',
                                 num_resblocks=single_or_idx(cfg.resblocks_per_level, -1),
                                 strides=(1,1),
                                 name='ublock_mid',
                                 **common_block_kwargs) \
                                (x, cond_embed, text_enc, text_lens, deterministic=deterministic)

    print('Encoder Skip Shapes: ',list(map(lambda x : x.shape, down_outputs)))
    # up branch resblocks + attn on designated resolutions + skip connections
    for level, width_mult in list(enumerate(cfg.width_multipliers[:-1]))[::-1]:
        level_channels = cfg.model_dim * width_mult

        # for res_idx in range(cfg.resblocks_per_level + 1):
        u_skip = down_outputs.pop()
        x = jnp.concatenate([x, u_skip], axis=-1)
        use_attn = (2 ** level) in cfg.attn_resolutions_divs.keys()
        attn_type = cfg.attn_resolutions_divs[2 ** level] if use_attn else None
        x = layers_sr.EfficientBlock(out_channels=level_channels,
                                     use_attn=use_attn,
                                     attn_type=attn_type,
                                     up_down='up',
                                     num_resblocks=single_or_idx(cfg.resblocks_per_level, level),
                                     strides=(2,2),
                                     name=f'UBlock_l{level}',
                                     **common_block_kwargs) \
                                     (x, cond_embed, text_enc, text_lens, deterministic=deterministic)
        print("Decoder UBlock #", level, " out resolution: ", x.shape[1:3], " level: ", level, " width: ", level_channels, " skip shape: ", u_skip.shape)

    # out convolution model_dim -> image_channels
    x = spatial_conv(features=input_ch, name='unet_out_conv', kernel_init=nn.initializers.zeros)(x)
    return x

