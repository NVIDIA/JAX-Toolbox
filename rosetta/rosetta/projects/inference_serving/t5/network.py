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

# Copyright 2022 The T5X Authors.
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

"""T5.1.1 Transformer Encoder only model for getting text embeddings"""

from flax import linen as nn
import jax.numpy as jnp
from t5x.contrib.gpu.t5 import layers
from t5x.contrib.gpu.t5.network import T5Config, Encoder, SeqDataFormat
from t5x.te_helper import TransformerEngineHelper

class TransformerEncoderOnly(nn.Module):
  """An encoder-only T5 Transformer model."""
  config: T5Config

  def setup(self):
    cfg = TransformerEngineHelper.get_t5x_config(self.config)

    self.shared_embedding = layers.Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=False,
        name='token_embedder')

    self.encoder = Encoder(config=cfg, shared_embedding=self.shared_embedding)

  def encode(self,
             encoder_input_tokens,
             encoder_segment_ids=None,
             enable_dropout=True,
             output_format=SeqDataFormat.BATCH_SEQ_HIDDEN):
    """Applies Transformer encoder-branch on the inputs."""
    cfg = self.config
    assert encoder_input_tokens.ndim == 2  # (batch, len)

    # Make padding attention mask.
    encoder_mask = layers.make_attention_mask(
        encoder_input_tokens > 0, encoder_input_tokens > 0, dtype=cfg.dtype)
    # Add segmentation block-diagonal attention mask if using segmented data.
    if encoder_segment_ids is not None:
      encoder_mask = layers.combine_masks(
          encoder_mask,
          layers.make_attention_mask(
              encoder_segment_ids,
              encoder_segment_ids,
              jnp.equal,
              dtype=cfg.dtype))

    encoder_mask = TransformerEngineHelper.get_attn_mask(encoder_mask)

    return self.encoder(
        encoder_input_tokens, encoder_mask, deterministic=not enable_dropout,
        output_format=output_format)

  def __call__(self,
               encoder_input_tokens,
               encoder_segment_ids=None,
               *,
               enable_dropout: bool = True):
    """Applies Transformer encoder-only model on the inputs.

    This method requires just encoder inputs

    Args:
      encoder_input_tokens: input data to the encoder.
      encoder_segment_ids: encoder segmentation info for packed examples.
      enable_dropout: Ensables dropout if set to True.

    Returns:
      logits array from just the encoder of a T5 model
    """
    encoded = self.encode(
        encoder_input_tokens,
        encoder_segment_ids=encoder_segment_ids,
        enable_dropout=enable_dropout)

    return encoded
