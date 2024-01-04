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

"""
Diffusion training losses

This module includes loss functions for various diffusion model
training regimes
"""

from typing import Callable, Tuple, Mapping, Optional, Union
import typing_extensions
import functools

import jax
import jax.numpy as jnp
import jax.debug
from rosetta.projects.diffusion.augmentations import AugmentationCallable
from rosetta.projects.diffusion.denoisers import DenoisingFunctionCallable, NoisePredictorCallable

PyTreeDef = type(jax.tree_util.tree_structure(None))

class DiffusionLossCallable(typing_extensions.Protocol):
    """ Call signature for a diffusion loss function.
        Returns the loss and the noises used """
    def __call__(self,
                 denoise_fn: Callable,
                 rng: jax.Array,
                 samples: jnp.ndarray,
                 other_cond: Optional[Mapping[str, jnp.ndarray]]
                 ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ...

class EPSDiffusionLossCallable(typing_extensions.Protocol):
    """ Call signature for a diffusion loss function based on noise(eps) prediction.
        Returns the loss and the noises used """
    def __call__(self,
                 eps_predictor: Callable,
                 rng: jax.Array,
                 samples: jnp.ndarray,
                 other_cond: Optional[Mapping[str, jnp.ndarray]]
                 ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ...

class EDMLoss:
    """ EDM loss from Karras et. al. 2022"""
    def __init__(self, p_mean=-1.2, p_std=1.2, sigma_data=0.5,
                 sample_aug_fn: Optional[AugmentationCallable]=None,
                 cond_aug_fn: Optional[AugmentationCallable]=None,
                 dim_noise_scalar=1.):
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data
        self.sample_aug_fn = sample_aug_fn
        self.cond_aug_fn = cond_aug_fn
        self.dim_noise_scalar = dim_noise_scalar 

    def _loss_weight(self, sigma):
        sigma = sigma / self.dim_noise_scalar
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

    def _noise_sampler(self, rng: jax.Array, count: int, dim_scalar:float=1.):
        rnd_normal = jax.random.normal(rng, (count, ))
        return jnp.exp(rnd_normal * self.p_std + self.p_mean) * dim_scalar

          

    def __call__(self,
                 denoise_fn: Callable,
                 rng: jax.Array,
                 samples: jnp.ndarray,
                 other_cond: Optional[Mapping[str, jnp.ndarray]]=None
                 ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """ 
        Returns the EDM loss and the noises used given a denoiser and samples.
        Args: 
            denoise_fn: black-box function that denoises an image given
                        samples, sigmas, and other conditioning
            rng: rng for sampling sigmas, noise, and optionally dropout
            samples: array of samples to be diffused
            other_cond: arbitrary other conditioning to pass to the
                        denoiser. Could be text conditioning.
            enable_dropout: should use dropout
        """
        dropout_rng, sigma_rng, noise_rng = jax.random.split(rng, 3)

        if self.sample_aug_fn:
            samples, dropout_rng = self.sample_aug_fn(samples, dropout_rng)
        if self.cond_aug_fn:
            other_cond, dropout_rng = self.cond_aug_fn(other_cond, dropout_rng)

        batch_dim = samples.shape[0]
        sigma = self._noise_sampler(sigma_rng, batch_dim, self.dim_noise_scalar)
        sigma = expand_dims_like(sigma, samples)
        weight = jnp.reshape(self._loss_weight(sigma), batch_dim)

        noise = jax.random.normal(noise_rng, samples.shape, samples.dtype)
        noised_sample = samples + noise * sigma

        denoised = denoise_fn(noised_sample, sigma, other_cond, dropout_rng=dropout_rng)
        sq_err = (denoised - samples) ** 2
        loss_unweighted = jnp.mean(jnp.reshape(sq_err, (batch_dim, -1)), axis=-1)
        return weight * loss_unweighted, jnp.mean(loss_unweighted), sigma

class EDMSuperResolutionLoss(EDMLoss):
    def __call__(self,
                 denoise_fn: Callable,
                 rng: jax.Array,
                 samples: jnp.ndarray,
                 other_cond: Optional[Mapping[str, jnp.ndarray]]=None
                 ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        lowres_aug_rng, noise_rng, rng = jax.random.split(rng, 3)

        assert other_cond and 'low_res_images' in other_cond.keys(), f'Superresolution loss requires a low_res_image in the other_cond of the sample. One was not found'
        lowres = other_cond['low_res_images']
        batch_dim = samples.shape[0]
        sigma = self._noise_sampler(lowres_aug_rng, batch_dim, 1.)
        sigma = expand_dims_like(sigma, lowres)

        noise = jax.random.normal(noise_rng, lowres.shape, lowres.dtype)
        noised_low_res = lowres + noise * sigma

        other_cond = {'low_res_samples': noised_low_res, 'noise_aug_level': sigma, **other_cond}

        return super().__call__(denoise_fn, rng, samples, other_cond)

def expand_dims_like(target, source):
    return jnp.reshape(target, target.shape + (1, ) * (len(source.shape) - len(target.shape)))