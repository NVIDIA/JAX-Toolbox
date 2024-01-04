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

"""Diffusion Models.

This module wraps around networks.py to integrate training, sampling, and construction.
"""

from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple, Union, Type

import abc
import clu.metrics as clu_metrics
from flax import core as flax_core
from flax import linen as nn
from flax.core import scope as flax_scope
from flax.linen import partitioning as flax_partitioning
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from t5x import metrics as metrics_lib
from t5x import optimizers
from t5x.models import BaseModel
import tensorflow as tf
import typing_extensions
import functools

from rosetta.projects.diffusion import denoisers
from rosetta.projects.diffusion import losses
from rosetta.projects.diffusion import samplers

Array = Union[np.ndarray, jnp.ndarray, tf.Tensor]
MetricsMap = metrics_lib.MetricsMap
PyTreeDef = Type[type(jax.tree_util.tree_structure(None))]
BatchType = Mapping[str, jnp.ndarray]

class DiffusionBase(BaseModel):
    def __init__(self,
                 optimizer_def: optimizers.OptimizerDefType):
        super().__init__(optimizer_def=optimizer_def)
        self.FEATURE_CONVERTER_CLS=None # for t5x trainer compatibility

    def eval_fn(self,
                params: PyTreeDef,
                batch: BatchType,
                ) -> Tuple[jnp.ndarray, MetricsMap]:
        return self.loss_fn(params, batch, dropout_rng=None)

    def score_batch(self,
                  params: PyTreeDef,
                  batch: BatchType,
                  return_intermediates: bool = False) -> jnp.ndarray:
        raise NotImplementedError("Batch scoring not supported by Diffusion Models")


class DenoisingDiffusionModel(DiffusionBase):
    """ Wrapper for a denoiser with an arbirary training scheme """
    def __init__(self, 
                 denoiser: denoisers.Denoiser,
                 diffusion_loss: losses.DiffusionLossCallable,
                 diffusion_sampler: samplers.DiffusionSampler,
                 optimizer_def: optimizers.OptimizerDefType,
                 sampling_cfg: Optional[samplers.SamplingConfig]=None):
        self.denoiser = denoiser
        self.diffusion_loss = diffusion_loss
        self.sampler = diffusion_sampler
        self.sampling_cfg = sampling_cfg
        super().__init__(optimizer_def=optimizer_def)

    def _compute_metrics(self, 
                         loss: jnp.ndarray,
                         loss_unweighted: jnp.ndarray,
                         avg_sigma: jnp.ndarray,
                         num_examples: int) -> MetricsMap:
        return compute_basic_diffusion_metrics(loss, loss_unweighted, avg_sigma, num_examples)

    def _denoise_fn(self,
                    params: PyTreeDef,
                    flax_mutables: Optional[PyTreeDef] = None,
                    ):
        return functools.partial(self.denoiser.denoise_sample, params, flax_mutables=flax_mutables)

    def loss_fn(self,
                params: PyTreeDef,
                batch: BatchType,
                dropout_rng: Optional[jax.Array],
                flax_mutables: Optional[PyTreeDef] = None,
                ) -> Tuple[jnp.ndarray, MetricsMap]:
        denoise_fn = self._denoise_fn(params, flax_mutables)

        samples = batch['samples']
        other_cond = {k: batch[k] for k in batch if k != 'samples'}
        batch_dim = samples.shape[0]

        loss, loss_unweighted, sigma = self.diffusion_loss(denoise_fn, dropout_rng, samples, other_cond)

        loss = jnp.mean(loss)
        avg_sigma = jnp.mean(sigma)
        return loss, self._compute_metrics(loss, loss_unweighted, avg_sigma, batch_dim)

    def predict_batch(self,
                      params: PyTreeDef,
                      batch: BatchType,
                      rng: Optional[jax.Array] = None,
                      *,
                      sampling_cfg: Optional[samplers.SamplingConfig] = None,
                      ) -> jnp.ndarray:
        return self.predict_batch_with_aux(params, batch, rng=rng, sampling_cfg=sampling_cfg)[0]

    def predict_batch_with_aux(self,
                               params: PyTreeDef,
                               batch: BatchType,
                               rng: Optional[jax.Array] = None,
                               *,
                               sampling_cfg: Optional[samplers.SamplingConfig] = None,
                               ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        denoise_fn = self._denoise_fn(params)
        sampling_cfg = sampling_cfg if sampling_cfg is not None else self.sampling_cfg

        if rng is None:
            ValueError("RNG is not optional for diffusion model sampling")
            exit()
        else:
            l_rng, rng = jax.random.split(rng)

        batch_samples=batch['samples']
        other_cond = {k: batch[k] for k in batch if k != 'samples'}
        latent = jax.random.normal(l_rng, batch_samples.shape)

        print("Running sampling at resolution: ", batch_samples.shape)
        step_idxs = jnp.arange(0, sampling_cfg.num_steps)
        samples = self.sampler.sample(denoise_fn, step_idxs, latent, rng, other_cond,
                                      sampling_cfg=sampling_cfg)
        return samples, {'None': None}

    def get_initial_variables(
        self,
        rng: jax.Array,
        input_shapes: Mapping[str, Array],
        input_types: Optional[Mapping[str, jnp.dtype]] = None
    ) -> flax_scope.FrozenVariableDict:
      """Returns the initial variables of the model."""
      input_types = {} if input_types is None else input_types
      sample_shape = input_shapes['samples']
      print("sample shape", sample_shape)
      sample_dtype = input_types.get('samples', jnp.float32)
      sigma_shape = input_shapes.get('timesteps', (sample_shape[0],))
 
      if len(sigma_shape) != 1:
          print("BAD SIGMA SHAPE: ", str(sigma_shape), " going to ", sample_shape[0])
          sigma_shape = sample_shape[0]
      sigma_dtype = input_types.get('timesteps', jnp.float32)
      print("Init Shapes: Sample: ", sample_shape, " Sigma: ", sigma_shape)
     
      inits = (jnp.ones(sample_shape, sample_dtype), jnp.ones(sigma_shape, sigma_dtype))

      low_res_type = input_types.get('low_res_images', None)
      # jax.debug.print(str(input_shapes))
      
      text_enc_dtype = input_types.get('text', None)
      if text_enc_dtype is not None:
          text_enc_shape = input_shapes.get('text',None)
          text_mask_dtype = input_types.get('text_mask', None)
          text_mask_shape = input_shapes.get('text_mask', None)

          init_txt = jnp.ones(text_enc_shape, text_enc_dtype)
          init_txt_mask = jnp.ones(text_mask_shape, text_mask_dtype)
          inits = inits + (init_txt, init_txt_mask)

      if low_res_type is not None:
          low_res_shape = input_shapes.get('low_res_images', None)
          aug_level_shape = input_shapes.get('noise_aug_level', sigma_shape)
          aug_level_type = input_types.get('noise_aug_level', sigma_dtype)
          jax.debug.print(str(low_res_shape))
          inits = inits + (jnp.ones(low_res_shape, low_res_type), jnp.ones(aug_level_shape, aug_level_type))

      initial_variables = self.denoiser.module.init(
              rng,
              *inits,
              enable_dropout=False)
      return initial_variables

def compute_basic_diffusion_metrics(
    loss: jnp.ndarray,
    loss_unweighted: jnp.ndarray,
    avg_sigma: jnp.ndarray,
    num_examples: int,
) -> MetricsMap:
  """Compute summary metrics.

  Args:
   loss: loss (float)
   mean_sigma: mean sigma noises used (float)
   num_examples (int) number of examples in batch

  Returns:
    Dict of metrics.
  """
  num_devices = jax.device_count()
  assert num_devices, 'JAX is reporting no devices, but it should.'
  # Note: apply mask again even though mask has already been applied to loss.
  # This is needed to divide by mask sum, but should not affect correctness of
  # the numerator.
  metrics = {
      'loss':
          metrics_lib.AveragePerStep(total=loss),
      'loss_unweighted':
          metrics_lib.AveragePerStep(total=loss_unweighted),
      'timing/images_per_second':
          metrics_lib.TimeRate.from_model_output(numerator=num_examples),
      'timing/steps_per_second':
          metrics_lib.StepsPerTime.from_model_output(),
      'timing/seconds':
          metrics_lib.Time(),
      'timing/images':
          metrics_lib.Sum(num_examples),
      'timing/images_per_second_per_core':
          metrics_lib.TimeRate.from_model_output(numerator=num_examples /
                                                 num_devices),
      'diff_stats/avg_sigma':
          metrics_lib.AveragePerStep(total=avg_sigma),

  }
  return metrics
