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
Diffusion-based denoisers

This module builds a denoising model that can be used as a black box
for training and sampling with arbitrary methods
"""

from typing import Mapping, Optional, Tuple, Union, Type
import abc
import typing_extensions

from flax import linen as nn
from flax.core import scope as flax_scope
import jax
import jax.numpy as jnp

PyTreeDef = Type[type(jax.tree_util.tree_structure(None))]

class PrecondSigmaFnCallable(typing_extensions.Protocol):
    """ Call signature for sigma-dependant preconditioning function """
    def __call__(self, sigma:jnp.ndarray) -> jnp.ndarray:
        ...

class DenoisingFunctionCallableWithParams(typing_extensions.Protocol):
    """ Call signature for a denoising function """
    def __call__(self, 
                 params: PyTreeDef,
                 noised_sample: jnp.ndarray,
                 sigma: jnp.ndarray,
                 other_cond: Optional[Mapping[str, jnp.ndarray]]=None,
                 dropout_rng: Optional[jax.Array]=None
                 ) -> jnp.ndarray:
        ...

class DenoisingFunctionCallable(typing_extensions.Protocol):
    """ Call signature for a denoising function """
    def __call__(self, 
                 noised_sample: jnp.ndarray,
                 sigma: jnp.ndarray,
                 other_cond: Optional[Mapping[str, jnp.ndarray]]=None,
                 dropout_rng: Optional[jax.Array]=None
                 ) -> jnp.ndarray:
        ...

class DenoisingFunctionWithAuxCallable(typing_extensions.Protocol):
    """ Call signature for a denoising function """
    def __call__(self, 
                 noised_sample: jnp.ndarray,
                 sigma: jnp.ndarray,
                 other_cond: Optional[Mapping[str, jnp.ndarray]]=None,
                 dropout_rng: Optional[jax.Array]=None
                 ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        ...

class NoisePredictorCallable(typing_extensions.Protocol):
    """ Call signature for a 'eps' or 'v' noise predicting function """
    def __call__(self, 
                 noised_sample: jnp.ndarray,
                 sigma: jnp.ndarray,
                 other_cond: Optional[Mapping[str, jnp.ndarray]]=None,
                 dropout_rng: Optional[jax.Array]=None
                 ) -> jnp.ndarray:
        ...

class NoisePredictorWithAuxCallable(typing_extensions.Protocol):
    """ Call signature for a 'eps' or 'v' noise predicting function """
    def __call__(self, 
                 noised_sample: jnp.ndarray,
                 sigma: jnp.ndarray,
                 other_cond: Optional[Mapping[str, jnp.ndarray]]=None,
                 dropout_rng: Optional[jax.Array]=None
                 ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        ...

class Denoiser(abc.ABC):
    """
    Model that returns a denoised sample given a noised sample and noise conditioning.
    Implements:
    $$D_\\theta(x; \\sigma) = c_{skip}(\\sigma)x +
        c_{out}(\\sigma)F_\\theta(c_{in}(\\sigma)x; c_{noise}(\\sigma))$$
    from Karras et. al. EDM
    """
    def __init__(self,
                 raw_model: nn.Module,
                 c_skip_fn: PrecondSigmaFnCallable,
                 c_out_fn: PrecondSigmaFnCallable,
                 c_in_fn: PrecondSigmaFnCallable,
                 c_noise_fn: PrecondSigmaFnCallable):
        """
        Args:
          raw_model: nn.Module that corresponds to $F_\\theta$
          c_skip_fn, c_out_fn,
          c_in_fn, c_noise_fn: Functions of $\\sigma$ that correspond to their terms
                               in the $D_\\theta$ equation.
        """
        self.module = raw_model
        self.c_skip_fn = c_skip_fn
        self.c_out_fn = c_out_fn
        self.c_in_fn = c_in_fn
        self.c_noise_fn = c_noise_fn

    @abc.abstractmethod
    def apply_module(
        self,
        params: PyTreeDef,
        batch: Mapping[str, jnp.ndarray],
        rngs: Optional[jax.Array] = None,
        mutable: flax_scope.CollectionFilter = False,
        other_variables: Optional[PyTreeDef] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
        """ Apply raw neural net """

    def prob_grad(self, params, noised_sample, sigma) -> jnp.ndarray:
        """
        Computes the gradient of the probability distribution wrt x:
        $ \\nabla_x log(p(x; \\sigma) = (D_\\theta - x) / \\sigma**2 $
        """
        return (self.denoise_sample(params, noised_sample, sigma) - noised_sample) / sigma ** 2

    def denoise_sample(self,
                       params: PyTreeDef,
                       noised_sample: jnp.ndarray,
                       sigma: jnp.ndarray,
                       other_cond: Optional[Mapping[str, jnp.ndarray]]=None,
                       dropout_rng: Optional[jax.Array]=None,
                       flax_mutables: Optional[PyTreeDef]=None,
                       ) -> jnp.ndarray:
        """ Returns denoised sample given noised sample and conditioning """
        sigma = expand_dims_like(sigma, noised_sample)
        skip_scale = self.c_skip_fn(sigma)
        out_scale = self.c_out_fn(sigma)
        in_scale = self.c_in_fn(sigma)
        noise_cond = self.c_noise_fn(sigma)

        batch = {'samples': in_scale * noised_sample, 'noise_cond': noise_cond}
        if other_cond is not None:
            batch.update(other_cond)

        if 'low_res_images' in batch.keys():
            noise_aug_level = batch.get('noise_aug_level', jnp.ones_like(sigma) * 0.002)
            low_res_noise_cond = self.c_noise_fn(noise_aug_level)
            low_res_in_scale = self.c_in_fn(noise_aug_level)
            low_res_batch = {'low_res_images': low_res_in_scale * batch['low_res_images'], 'noise_aug_level': low_res_noise_cond} 
            batch.update(low_res_batch)

        return skip_scale * noised_sample + \
               out_scale * self.apply_module(params, batch, dropout_rng, other_variables=flax_mutables)

class EDMUnconditionalDenoiser(Denoiser):
    """ Denoiser that implements the training regime from Karras et. al. EDM """
    def __init__(self,
                 raw_model: nn.Module,
                 sigma_data: float=0.5):
        self.sigma_data = sigma_data
        c_skip_fn = lambda sigma: (sigma_data ** 2) / (sigma ** 2 + sigma_data ** 2)
        c_out_fn = lambda sigma: (sigma * sigma_data) / jnp.sqrt(sigma_data ** 2 + sigma ** 2)
        c_in_fn = lambda sigma: 1.0 / jnp.sqrt(sigma ** 2 + sigma_data ** 2)
        c_noise_fn = lambda sigma: 0.25 * jnp.log(sigma)

        super().__init__(raw_model, c_skip_fn, c_out_fn, c_in_fn, c_noise_fn)

    def apply_module(
        self,
        params: PyTreeDef,
        batch: Mapping[str, jnp.ndarray],
        rngs: Optional[jax.Array] = None,
        mutable: flax_scope.CollectionFilter = False,
        other_variables: Optional[PyTreeDef] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
        """Computes module output via a forward pass of `self.module`."""
        # Dropout is provided only for the training mode.
        rngs = {'dropout': rngs} if rngs is not None else None
        if other_variables is None:
            other_variables = {}
        return self.module.apply(
            {
                'params': params,
                **other_variables
            },
            batch['samples'],
            batch['noise_cond'],
            enable_dropout=rngs is not None,
            rngs=rngs,
            mutable=mutable)

class EDMTextConditionedDenoiser(EDMUnconditionalDenoiser):
    """ 
    Denoiser that implements the training regime from Karras et. al. EDM
    and accepts text conditioning.
    """

    def apply_module(
        self,
        params: PyTreeDef,
        batch: Mapping[str, jnp.ndarray],
        rngs: Optional[jax.Array] = None,
        mutable: flax_scope.CollectionFilter = False,
        other_variables: Optional[PyTreeDef] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
        """Computes module output via a forward pass of `self.module`."""
        # Dropout is provided only for the training mode.
        rngs = {'dropout': rngs} if rngs is not None else None
        if other_variables is None:
            other_variables = {}
        return self.module.apply(
            {
                'params': params,
                **other_variables
            },
            batch['samples'],
            batch['noise_cond'],
            batch['text'],
            batch['text_mask'],
            enable_dropout=rngs is not None,
            rngs=rngs,
            mutable=mutable)

class EDMTextConditionedSuperResDenoiser(EDMTextConditionedDenoiser):
    """ 
    Denoiser that implements the training regime from Karras et. al. EDM
    and accepts text and lowres conditioning.
    """

    def apply_module(
        self,
        params: PyTreeDef,
        batch: Mapping[str, jnp.ndarray],
        rngs: Optional[jax.Array] = None,
        mutable: flax_scope.CollectionFilter = False,
        other_variables: Optional[PyTreeDef] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
        """Computes module output via a forward pass of `self.module`."""
        # Dropout is provided only for the training mode.
        rngs = {'dropout': rngs} if rngs is not None else None
        if other_variables is None:
            other_variables = {}
        # jax.debug.print('noise aug {n}', n=batch.get('noise_aug_level'))
        return self.module.apply(
            {
                'params': params,
                **other_variables
            },
            batch['samples'],
            batch['noise_cond'],
            text_enc=batch['text'],
            text_lens=batch['text_mask'],
            low_res_images=batch['low_res_images'],
            noise_aug_level=batch.get('noise_aug_level', None),
            enable_dropout=rngs is not None,
            rngs=rngs,
            mutable=mutable)

class EDMLatentConditionalDenoiser(EDMUnconditionalDenoiser):
    """
    Denoiser that implements the EDM training regime and accepts latent-preconditioning
    for RIN networks
    """

    def apply_module(
        self,
        params: PyTreeDef,
        batch: Mapping[str, jnp.ndarray],
        rngs: Optional[jax.Array] = None,
        mutable: flax_scope.CollectionFilter = False,
        other_variables: Optional[PyTreeDef] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
        """Computes module output via a forward pass of `self.module`."""
        # Dropout is provided only for the training mode.
        rngs = {'dropout': rngs} if rngs is not None else None
        if other_variables is None:
            other_variables = {}
        return self.module.apply(
            {
                'params': params,
                **other_variables
            },
            batch['samples'],
            batch['noise_cond'],
            batch['prev_latents'],
            enable_dropout=rngs is not None,
            rngs=rngs,
            mutable=mutable)

class VP_EPSNoisePredictor(abc.ABC):
    """
    Model that returns a denoised sample given a noised sample and noise conditioning.
    Implements:
    $$pred = c_{out}(\\sigma)F_\\theta(c_{in}(\\sigma)x; c_{noise}(\\sigma))$$
    from Karras et. al. EDM
    """
    def __init__(self,
                 raw_model: nn.Module):
        """
        Args:
          raw_model: nn.Module that corresponds to $F_\\theta$
          c_noise_fn, c_out_fn,
          c_in_fn : Functions of $\\sigma$ that correspond to their terms
                               in the $pred$ equation.
        """
        self.module = raw_model

    @abc.abstractmethod
    def apply_module(
        self,
        params: PyTreeDef,
        batch: Mapping[str, jnp.ndarray],
        rngs: Optional[jax.Array] = None,
        mutable: flax_scope.CollectionFilter = False,
        other_variables: Optional[PyTreeDef] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
        """ Apply raw neural net """

    def prob_grad(self, params, noised_sample, sigma) -> jnp.ndarray:
        """
        Computes the gradient of the probability distribution wrt x:
        $ \\nabla_x log(p(x; \\sigma) = (D_\\theta - x) / \\sigma**2 $
        """
        return (self.pred_sample(params, noised_sample, sigma) - noised_sample) / sigma ** 2

    def pred_sample(self,
                    params: PyTreeDef,
                    noised_sample: jnp.ndarray,
                    sigma: jnp.ndarray,
                    other_cond: Optional[Mapping[str, jnp.ndarray]]=None,
                    dropout_rng: Optional[jax.Array]=None
                    ) -> jnp.ndarray:
        """ Returns denoised sample given noised sample and conditioning """
        sigma = expand_dims_like(sigma, noised_sample)

        batch = {'samples': noised_sample, 'noise_cond': sigma}
        if other_cond is not None:
            batch.update(other_cond)

        return self.apply_module(params, batch, dropout_rng)

class VP_EPSNoisePredictorTextConditional(VP_EPSNoisePredictor):
    """ 
    Denoiser that implements the VP Eps prediction
    and accepts text conditioning.
    """

    def apply_module(
        self,
        params: PyTreeDef,
        batch: Mapping[str, jnp.ndarray],
        rngs: Optional[jax.Array] = None,
        mutable: flax_scope.CollectionFilter = False,
        other_variables: Optional[PyTreeDef] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
        """Computes module output via a forward pass of `self.module`."""
        # Dropout is provided only for the training mode.
        rngs = {'dropout': rngs} if rngs is not None else None
        if other_variables is None:
            other_variables = {}
        return self.module.apply(
            {
                'params': params,
                **other_variables
            },
            batch['samples'],
            batch['noise_cond'],
            batch['text'],
            enable_dropout=rngs is not None,
            rngs=rngs,
            mutable=mutable)


def expand_dims_like(target, source):
    return jnp.reshape(target, target.shape + (1, ) * (len(source.shape) - len(target.shape)))