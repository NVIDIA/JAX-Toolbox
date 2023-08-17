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

from typing import Any

from jax import numpy as jnp


class GoogleViTConfig:
    r"""
    This is the configuration class to store the configuration of a [`ViTModel`]. It is used to instantiate an ViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViT
    [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) architecture.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            Supported activation functions can be found at https://jax.readthedocs.io/en/latest/jax.nn.html
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to `224`):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to `16`):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to `3`):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        encoder_stride (`int`, `optional`, defaults to 16):
           Factor to increase the spatial resolution by in the decoder head for masked image modeling.

    """
    model_type = 'google-vit'

    def __init__(
        self,
        # parameters from ViTModel
        hidden_size=768, # hidden_size
        num_hidden_layers=12, # transformer.num_layers
        num_attention_heads=12, # transformer.num_heads
        intermediate_size=3072, # transformer.mlp_dim
        hidden_act='gelu',  # HF CLIPFlaxVisionModel uses quick_gelu, but google uses gelu_new (for mlp)
        hidden_dropout_prob=0.0,  # transformer.dropout_rate
        attention_probs_dropout_prob=0.0, # transformer.attention_dropout_rate
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224, # not in config, but it's the dimension they fine-tuned imagenet on
        patch_size=16, #patches[0] == patches[1]
        num_channels=3,
        qkv_bias=True,
        encoder_stride=16,  # embeded in model name, i.e. Vit B/16
        # Additional parameters from Google ViT
        classifier: str = 'token',
        head_bias_init: float = 0.0,
        representation_size: int | None = None,
        num_classes: int | None = None,
        # pre_layernorm=True for parity with (HF's FlaxCLIPVisionModel), pre_layernorm=False for original Google ViT impl
        pre_layernorm: bool = False,
        dtype: Any = jnp.float32,
        **kwargs,
    ):

        if not qkv_bias:
            raise NotImplementedError("FlaxViTModel supports this, but currently turning it off isn't supported")

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.encoder_stride = encoder_stride

        # classifier: str = 'token' | 'token_unpooled' | 'unpooled' | 'gap' (token* = use trainable cls token | gap = return mean feature)
        self.classifier = classifier
        self.head_bias_init = head_bias_init
        self.representation_size = representation_size
        self.num_classes = num_classes

        self.pre_layernorm = pre_layernorm
        self.dtype = dtype

