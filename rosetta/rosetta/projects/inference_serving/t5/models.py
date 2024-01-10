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

"""T5X EncoderOnly Model"""

from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple, Union

from flax import linen as nn
from flax.core import scope as flax_scope
import jax
import jax.numpy as jnp
import seqio
from t5x import decoding
from t5x import optimizers
from t5x.models import BaseTransformerModel, DecodeFnCallable, Array, PyTreeDef

class EncoderOnlyModel(BaseTransformerModel):
  """Wrapper class for the TransformerEncoderOnly nn.module."""

  FEATURE_CONVERTER_CLS = seqio.EncDecFeatureConverter

  def __init__(
      self,
      module: nn.Module,
      input_vocabulary: seqio.Vocabulary,
      output_vocabulary: seqio.Vocabulary,
      optimizer_def: optimizers.OptimizerDefType,
      decode_fn: DecodeFnCallable = decoding.beam_search,
      feature_converter_cls: Optional[Callable[...,
                                               seqio.FeatureConverter]] = None,
      label_smoothing: float = 0.0,
      z_loss: float = 0.0,
      loss_normalizing_factor: Optional[float] = None,
  ):
    if feature_converter_cls is not None:
      self.FEATURE_CONVERTER_CLS = feature_converter_cls  # pylint: disable=invalid-name
    super().__init__(
        module=module,
        input_vocabulary=input_vocabulary,
        output_vocabulary=output_vocabulary,
        optimizer_def=optimizer_def,
        decode_fn=decode_fn,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
    )

  def get_initial_variables(
      self,
      rng: jax.Array,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, jnp.dtype]] = None
  ) -> flax_scope.FrozenVariableDict:
    """Get the initial variables for an encoder-decoder model."""
    input_types = {} if input_types is None else input_types
    encoder_shape = input_shapes['encoder_input_tokens']
    encoder_type = input_types.get('encoder_input_tokens', jnp.float32)
    if 'encoder_positions' in input_shapes:
      encoder_positions = jnp.ones(
          input_shapes['encoder_positions'],
          input_types.get('encoder_positions', jnp.int32))
    else:
      encoder_positions = None

    if 'encoder_segment_ids' in input_shapes:
      encoder_segment_ids = jnp.ones(
          input_shapes['encoder_segment_ids'],
          input_types.get('encoder_segment_ids', jnp.int32))
    else:
      encoder_segment_ids = None
    initial_variables = self.module.init(
        rng,
        jnp.ones(encoder_shape, encoder_type),
        encoder_segment_ids=encoder_segment_ids,
        enable_dropout=False)
    return initial_variables

  def _compute_logits(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array] = None,
      mutable: flax_scope.CollectionFilter = False,
      other_variables: Optional[PyTreeDef] = None,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
    """Computes logits via a forward pass of `self.module_cls`."""
    # Dropout is provided only for the training mode.
    rngs = {'dropout': dropout_rng} if dropout_rng is not None else None
    if other_variables is None:
      other_variables = {}

    variables = {
      'params': params,
      **other_variables
    }

    return self.module.apply(
        variables,
        batch['encoder_input_tokens'],
        encoder_segment_ids=batch.get('encoder_segment_ids', None),
        enable_dropout=False,
        rngs=rngs,
        mutable=mutable)

  def _compute_logits_from_slice(
      self, flat_ids: jnp.ndarray, flat_cache: Mapping[str, jnp.ndarray],
      params: PyTreeDef, encoded_inputs: jnp.ndarray, raw_inputs: jnp.ndarray,
      max_decode_length: int) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Token slice to logits from decoder model."""
    # flat_ids: [batch * beam, seq_len=1]
    # cache is expanded inside beam_search to become flat_cache
    # flat_cache: [batch * beam, num_heads, depth_per_head, max_decode_len]
    # flat_logits: [batch * beam, seq_len=1, vocab]
    flat_logits, new_vars = self.module.apply(
        {
            'params': params,
            'cache': flat_cache
        },
        encoded_inputs,
        raw_inputs,  # only needed for encoder padding mask
        flat_ids,
        flat_ids,
        enable_dropout=False,
        decode=True,
        max_decode_length=max_decode_length,
        mutable=['cache'],
        method=self.module.decode)
    # Remove sequence length dimension since it's always 1 during decoding.
    flat_logits = jnp.squeeze(flat_logits, axis=1)
    new_flat_cache = new_vars['cache']
    return flat_logits, new_flat_cache

  def score_batch(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, Any]]]:
    """Compute log likelihood score on a batch."""

    output = self._compute_logits(params, batch)  # type: jnp.ndarray

    return output

  def predict_batch_with_aux(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.Array] = None,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
      raise NotImplementedError("Predict Batch is not implemented for encoder only. Use score_batch")
