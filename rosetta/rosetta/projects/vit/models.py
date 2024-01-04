# Copyright (c) 2023, The T5X Authors.
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

from collections.abc import Mapping
from typing import Any, Optional

import clu.metrics as clu_metrics
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from flax.core import scope as flax_scope

from rosetta.projects.vit.layers import FlaxGViTForImageClassificationModule
from t5x import metrics as metrics_lib
from t5x import optimizers
from t5x.models import BaseModel


_ShardedDeviceArray = Any
Array = np.ndarray | jnp.ndarray | _ShardedDeviceArray | tf.Tensor
MetricsMap = metrics_lib.MetricsMap
PyTreeDef = type(jax.tree_util.tree_structure(None))

class ViTModel(BaseModel):

  FEATURE_CONVERTER_CLS = None

  def __init__(
      self,
      module: FlaxGViTForImageClassificationModule,
      optimizer_def: optimizers.OptimizerDefType,
  ):
    super().__init__(optimizer_def)
    self.module = module

  def loss_fn(
      self, params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: jax.Array | None,
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> tuple[jnp.ndarray, MetricsMap]:
    """Computes loss and metrics.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      dropout_rng: rng to use for dropout, or None for deterministic mode.

    Returns:
      loss: the loss computed for the given inputs and parameters.
      aux:
        weight_sum: sum of the per-token weights applied to the loss.
        metrics: a mapping of metrics computed for this batch.
    """

    assert not flax_mutables, "ViT currently does not support 'flax_mutables'"

    if dropout_rng is not None:
      dropout_rng = {'dropout': dropout_rng}

    logits = self.module.apply(
        {'params': params},
        rngs=dropout_rng,
        pixel_values=batch['images'],
        ## train == not deterministic
        deterministic=(dropout_rng is None))

    def cross_entropy_loss(*, logits, labels):
      logp = jax.nn.log_softmax(logits)
      return -jnp.mean(jnp.sum(logp * labels, axis=1))

    labels = batch['labels']
    loss = cross_entropy_loss(logits=logits, labels=labels)

    labels = jnp.argmax(labels, axis=-1).astype(jnp.int32)
    metrics = self._compute_metrics(
        logits=logits,
        targets=labels,
        loss=loss)

    return loss, metrics

  def eval_fn(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
  ) -> tuple[jnp.ndarray, MetricsMap]:
    return self.loss_fn(params, batch, dropout_rng=None)


  def _compute_metrics(
      self,
      logits: jnp.ndarray,
      targets: jnp.ndarray,
      loss: jnp.ndarray,
  ) -> MetricsMap:
    return compute_base_metrics_vit(
        logits=logits, targets=targets, loss=loss)


  def predict_batch_with_aux(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      rng: jax.Array | None = None,
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Predict a batch from the modelwith auxiliary outputs.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      rng: an optional RNG key to use during prediction (e.g., for decoding).

    Returns:
      predictions: the model predictions
      aux: auxiliary data
    """
    assert not flax_mutables, "ViT currently does not support 'flax_mutables'"

    logits = self.module.apply(
        {'params': params},
        rngs=rng,
        pixel_values=batch['images'],
        ## train == not deterministic
        deterministic=True)

    predictions = jnp.argmax(logits, axis=-1)
    return predictions, None

  def score_batch(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> jnp.ndarray:
    """Compute log likelihood score on a batch."""

    assert not return_intermediates, '`return_intermediates` is not supported'
    assert not flax_mutables, "ViT currently does not support 'flax_mutables'"

    logits = self.module.apply(
        {'params': params},
        rngs=None,
        pixel_values=batch['images'],
        deterministic=True)

    logp = jax.nn.log_softmax(logits)
    labels = batch['labels'].astype(jnp.int32)
    sequence_scores = jnp.sum(logp * labels, axis=1)

    return sequence_scores

  def get_metrics_per_batch(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
  ) -> jnp.ndarray:
    """Computes evaluation metrics for a batch.

    Returns: dict mapping metric name to per-example metric value. """

    logits = self.module.apply(
        {'params': params},
        rngs=None,
        pixel_values=batch['images'],
        deterministic=True)
    logp = jax.nn.log_softmax(logits)

    labels = batch['labels'].astype(jnp.int32)
    loss = -jnp.sum(logp * labels, axis=1)

    labels = jnp.argmax(labels, axis=-1).astype(jnp.int32)
    accuracy = (jnp.argmax(logits, axis=-1) == labels)

    return {'loss': loss, 'accuracy': accuracy}

  def get_initial_variables(
      self,
      rng: jax.Array,
      input_shapes: Mapping[str, Array],
      input_types: Mapping[str, jnp.dtype] | None = None,
      flax_mutables: Optional[PyTreeDef] = None,
  ) -> flax_scope.FrozenVariableDict:

    assert not flax_mutables, "ViT currently does not support 'flax_mutables'"

    return self.module.init(
        rng,
        jnp.ones(input_shapes['images'], dtype=input_types['images']),
        deterministic=True)

def compute_base_metrics_vit(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    loss: jnp.ndarray,
) -> MetricsMap:
  """Compute summary metrics.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array of categories.
     values (float-valued weights not supported).
   loss: loss (float)
     for packing, i.e. [batch, length] arrays.
  Returns:
    Dict of metrics.
  """
  num_examples = targets.shape[0]
  num_devices = jax.device_count()
  metrics = {
      'accuracy':
          clu_metrics.Accuracy.from_model_output(
              logits=logits, labels=targets),
      'loss':
          metrics_lib.AveragePerStep(total=loss),
      'timing/samples_per_second':
          metrics_lib.TimeRate.from_model_output(numerator=num_examples),
      'timing/steps_per_second':
          metrics_lib.StepsPerTime.from_model_output(),
      'timing/seconds':
          metrics_lib.Time(),
      'timing/samples':
          metrics_lib.Sum(num_examples),
      'timing/samples_per_second_per_core':
          metrics_lib.TimeRate.from_model_output(numerator=num_examples /
                                                 num_devices),
  }

  return metrics

