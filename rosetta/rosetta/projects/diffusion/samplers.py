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
Diffusion/Score Matching Samplers

This module holds samplers that use black box denoisers
"""

from typing import Mapping, Optional, Tuple, Callable, Sequence
import typing_extensions
import abc
import dataclasses

import jax
import jax.numpy as jnp
from rosetta.projects.diffusion.denoisers import DenoisingFunctionCallable
from rosetta.projects.diffusion.mm_utils import expand_dims_like

PyTreeDef = type(jax.tree_util.tree_structure(None))
BatchType = Mapping[str, jnp.ndarray]

@dataclasses.dataclass
class SamplingConfig:
    num_steps: int = 50
    generation_shape: Optional[Sequence[int]] = None
    
@dataclasses.dataclass
class CFGSamplingConfig(SamplingConfig):
    cf_guidance_weight: Optional[float] = None 
    cf_guidance_nulls: Optional[Mapping[str, Optional[jnp.ndarray]]] = None

class DiffusionSamplerCallable(typing_extensions.Protocol):
    """ Call signature for a diffusion sampling function.
        Returns the samples."""
    def __call__(self,
                 denoise_fn: Callable,
                 step_indices: jnp.ndarray,
                 latent: jnp.ndarray,
                 rng: Optional[jax.Array],
                 other_cond: Optional[Mapping[str, jnp.ndarray]],
                 sampling_cfg: Optional[SamplingConfig]=None
                 ) -> jnp.ndarray:
        ...

class DiffusionSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self,
               denoise_fn: Callable,
               step_indices: jnp.ndarray,
               latent: jnp.ndarray,
               rng: Optional[jax.Array],
               other_cond: Optional[Mapping[str, jnp.ndarray]], 
               sampling_cfg: Optional[SamplingConfig]=None
               ) -> jnp.ndarray:
        pass

    def apply_cf_guidance(self, with_cond: jnp.ndarray, no_cond: jnp.ndarray, guidance_weight:float) -> jnp.ndarray:
        """
        Applies classifier-free guidance.

        Args:
        with_cond: Model output, assumed to have shape [b, ...]
        no_cond:   Model output, assumed to have shape [b, ...]
        guidance_weight: cf guidance weight
        """
        diff = with_cond - no_cond

        guided = with_cond + guidance_weight * diff
        return guided


identity = lambda x:x
class EDMSampler(DiffusionSampler):
    """ 
    Samples using a denoising model as per Karras et. al. EDM Algorithm 2
    """ 
    def __init__(self,
                 sigma_min: float = 0.002,
                 sigma_max: float = 80,
                 rho: float = 7,
                 S_churn: float = 0,
                 S_min: float = 0,
                 S_max: float = float('inf'),
                 S_noise: float = 1.0,
                 round_sigma: Callable = identity,
                 dim_noise_scalar: float = 1.0,
                 ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.round_sigma = round_sigma
        self.dim_noise_scalar = dim_noise_scalar


    def _sample_noise(self, shape: Tuple, rng: jax.Array):
        return jax.random.normal(rng, shape) * self.S_noise

    def _scannable_single_step(self, denoise_fn, num_steps, t_steps, other_cond, null_cond, second_order_correct=True, cf_guidance_weight=None):
        """ Wraps single_step_sample to make it usable in jax.lax.scan """
        def wrapped_fn(x_rng_state: Tuple[jnp.ndarray, jax.Array], idx: int):
            return self.single_step_sample(denoise_fn, num_steps, t_steps, x_rng_state[1], idx, \
                                           second_order_correct=second_order_correct, x_curr=x_rng_state[0], \
                                           other_cond=other_cond, null_cond=null_cond, cf_guidance_weight=cf_guidance_weight)
        return wrapped_fn

    def _get_fori_body(self, denoise_fn: DenoisingFunctionCallable, 
                       num_steps: int,
                       t_steps: jnp.ndarray,
                       other_cond:Optional[BatchType]=None,
                       null_cond:Optional[BatchType]=None,
                       second_order_correct=True,
                       cf_guidance_weight:Optional[float]=None):
        def loop_body(step, args):
            x, curr_rng = args
            args, _ = self.single_step_sample(denoise_fn, num_steps, t_steps, curr_rng, step, \
                                              second_order_correct=second_order_correct, x_curr=x, \
                                              other_cond=other_cond, null_cond=null_cond, cf_guidance_weight=cf_guidance_weight)
            return args

        return loop_body

    def _get_eps(self, denoise_fn:DenoisingFunctionCallable,
                 noised_x, t,
                 other_cond:Optional[BatchType]=None,
                 null_cond:Optional[BatchType]=None,
                 cf_guidance_weight:Optional[float]=None):
        #Calculates a potentially CF-guided eps in one forward pass
        batch_dim = noised_x.shape[0]

        # Setup concats for cf_guidance
        if cf_guidance_weight is not None:
            assert null_cond is not None, f"Using CF-guidance {cf_guidance_weight}. \
                        You must provide a null_cond if doing classifier-free guidance. \
                        It's currently None"
            noised_x = jnp.concatenate([noised_x, noised_x], axis=0)
            t = jnp.concatenate([t, t], axis=0)
            concatenate_fn = lambda x, y: jnp.concatenate([x, y], axis=0)
            other_cond = jax.tree_util.tree_map(concatenate_fn, other_cond, null_cond)

        denoised = denoise_fn(noised_sample=noised_x, sigma=t, other_cond=other_cond)
        denoised = dynamic_thresholding(denoised)
        eps = (noised_x - denoised) / t

        #Apply CF Guidance
        if cf_guidance_weight is not None:
            eps = self.apply_cf_guidance(eps[:batch_dim], eps[batch_dim:], cf_guidance_weight)
        
        return eps

    def single_step_sample(self, denoise_fn: DenoisingFunctionCallable, 
                           num_steps: int,
                           t_steps: jnp.ndarray,
                           rng: jax.Array,
                           t_idx:int,
                           x_curr: jnp.ndarray=None,
                           other_cond:Optional[BatchType]=None,
                           null_cond:Optional[BatchType]=None,
                           second_order_correct=True,
                           cf_guidance_weight:Optional[float]=None
        ) ->  Tuple[Tuple[jnp.ndarray, jnp.ndarray], jax.Array]:
        """ Single step of sampling """
        rng, step_rng = jax.random.split(rng)

        t_curr = t_steps[t_idx]
        t_next = t_steps[t_idx + 1]

        # Increase noise temporarily
        m_gamma = jax.lax.min(self.S_churn / num_steps, jnp.sqrt(2) - 1)
        gamma = jax.lax.cond(self.S_min <= t_curr,
                             lambda:jax.lax.cond(t_curr <= self.S_max, lambda:m_gamma, lambda:0.0), lambda:0.0)
        t_hat = self.round_sigma(t_curr + gamma * t_curr)
        x_hat = x_curr + jnp.sqrt((t_hat ** 2 - t_curr ** 2)) * \
                self.S_noise * self._sample_noise(x_curr.shape, step_rng)
    
        # Shape matching
        t_hat = batch_expand(t_hat, x_curr)
        t_next = batch_expand(t_next, x_curr)

        # Denoising
        eps = self._get_eps(denoise_fn, x_hat, t_hat, other_cond, null_cond, cf_guidance_weight)

        # Euler Step
        x_next = x_hat + (t_next - t_hat) * eps
        
        # Second order correction if t_idx < num_steps - 1
        if second_order_correct:
            corrected = self.second_order_correct(x_next, denoise_fn, x_hat, t_hat, t_next, eps, other_cond, null_cond, cf_guidance_weight)
        else:
            corrected = x_next
        return (corrected, rng), None #denoised

    def second_order_correct(self, x_next, denoise_fn, x_hat, t_hat, t_next, eps, other_cond, null_cond, cf_guidance_weight=None
        ) -> jnp.ndarray:
        # Denoising
        eps_prime = self._get_eps(denoise_fn, x_next, t_next, other_cond, null_cond, cf_guidance_weight)

        # 2nd order correction
        x_next = x_hat + (t_next - t_hat) * (0.5 * eps + 0.5 * eps_prime)
        return x_next

    def sample(self,
               denoise_fn: Callable,
               step_indices: jnp.ndarray,
               latent: jnp.ndarray,
               rng: jax.Array,
               other_cond: Optional[BatchType]=None,
               sampling_cfg: Optional[SamplingConfig]=None
               ) -> jnp.ndarray:
        # Classifier-free guidance will be enabled if cf_guidance_weight is not None
        if sampling_cfg is None or not hasattr(sampling_cfg, 'cf_guidance_weight'):
            cf_guidance_weight = None
            cf_guidance_nulls = None
        else:
            cf_guidance_weight = sampling_cfg.cf_guidance_weight
            cf_guidance_nulls = sampling_cfg.cf_guidance_nulls
            jax.debug.print("Using CF-Guidance weight {}".format(cf_guidance_weight))

        # Find timesteps
        r_rho = 1 / self.rho
        timesteps = (self.sigma_max ** r_rho + step_indices / (step_indices.shape[0] - 1) * \
                    (self.sigma_min ** r_rho - self.sigma_max ** r_rho)) ** self.rho
        timesteps = self.dim_noise_scalar * timesteps
        timesteps = jnp.concatenate((self.round_sigma(timesteps), jnp.zeros_like(timesteps[:1])))

        # Sampling Loop
        null_cond = None
        if cf_guidance_weight is not None:
            assert other_cond is not None, "other_cond is None. Cannot do cf-guidance without any conditioning"
            null_cond = assemble_cf_guidance_conds(other_cond, cf_guidance_nulls)

        prior = latent * timesteps[0]
        loop_body = self._get_fori_body(denoise_fn, num_steps=step_indices.shape[0], \
                                        t_steps=timesteps, other_cond=other_cond, null_cond=null_cond, \
                                        second_order_correct=True,cf_guidance_weight=cf_guidance_weight)
        samples, rng = jax.lax.fori_loop(0, step_indices.shape[0] - 1, loop_body, (prior, rng))

        # Last step (no second_order_correct)
        (samples, _), denoised = self.single_step_sample(denoise_fn, step_indices.shape[0], timesteps, rng, step_indices.shape[0] - 1, other_cond=other_cond, \
                                                         null_cond=null_cond, x_curr=samples, second_order_correct=False, cf_guidance_weight=cf_guidance_weight)
        jax.debug.print("single final step")

        return (samples + 1) / 2

    # A Data Parallel sampling loop that uses a pjitted denoise_fn call
    def sample_loop(self,
                    denoise_fn: Callable,
                    sampling_cfg: SamplingConfig,
                    latent: jnp.ndarray,
                    rng: jax.Array,
                    other_cond: Optional[BatchType]=None,
                    )-> jnp.ndarray:
        # Classifier-free guidance will be enabled if cf_guidance_weight is not None
        if not hasattr(sampling_cfg, 'cf_guidance_weight'):
            cf_guidance_weight = None
            cf_guidance_nulls = None
        else:
            cf_guidance_weight = sampling_cfg.cf_guidance_weight
            cf_guidance_nulls = sampling_cfg.cf_guidance_nulls
            jax.debug.print("Using CF-Guidance weight {}".format(cf_guidance_weight))

        # Find timesteps
        step_indices = jnp.arange(sampling_cfg.num_steps)
        r_rho = 1 / self.rho
        timesteps = (self.sigma_max ** r_rho + step_indices / (step_indices.shape[0] - 1) * \
                    (self.sigma_min ** r_rho - self.sigma_max ** r_rho)) ** self.rho
        timesteps = jnp.concatenate((self.round_sigma(timesteps), jnp.zeros_like(timesteps[:1])))

        # Sampling Loop
        null_cond = None
        if cf_guidance_weight is not None:
            assert other_cond is not None, "other_cond is None. Cannot do cf-guidance without any conditioning"
            jax.debug.inspect_array_sharding(other_cond, callback=print)
            null_cond = assemble_cf_guidance_conds(other_cond, cf_guidance_nulls)
            jax.debug.print("Assembed conds")

        prior = jnp.asarray(latent, jnp.float64) * timesteps[0]
        for time_idx in range(sampling_cfg.num_steps - 1):
            timestep = timesteps[sampling_cfg.num_steps - 1 - time_idx]
        step_fn = self._scannable_single_step(denoise_fn, step_indices.shape[0], timesteps, other_cond, null_cond, second_order_correct=True, cf_guidance_weight=cf_guidance_weight)
        (samples, rng), denoised = jax.lax.scan(step_fn, (prior, rng), jnp.arange(0, step_indices.shape[0] - 1))
        jax.debug.print("scanned")

        # Last step (no second_order_correct)
        (samples, _), denoised = self.single_step_sample(denoise_fn, step_indices.shape[0], timesteps, rng, step_indices.shape[0] - 1, other_cond=other_cond, \
                                                         null_cond=null_cond, x_curr=samples, second_order_correct=False, cf_guidance_weight=cf_guidance_weight)
        jax.debug.print("single final step")

        samples = (samples + 1) / 2
        
        repl_samples = jax.pjit(lambda x: x, in_shardings=None, out_shardings=None)(samples)
        return repl_samples



def assemble_cf_guidance_conds(other_cond: BatchType, 
                               guidance_nulls:Optional[Mapping[str, Optional[jnp.ndarray]]]) -> BatchType:
    null_cond = {}
    for k, v in other_cond.items():
        if guidance_nulls is None or k in guidance_nulls.keys():
            null_cond_val = None
            # If no explicit 'null' is provided, use zeros_like 
            if guidance_nulls is None or guidance_nulls[k] is None:
                null_cond_val = jnp.zeros_like(v)
            else:
                null_cond_val = guidance_nulls[k]
            null_cond[k] = null_cond_val
        else:
            null_cond[k] = v

    return null_cond

def dynamic_thresholding(denoised, p=99.5):
    s = jnp.percentile(
            jnp.abs(denoised), p,
            axis=tuple(range(1, denoised.ndim)),
            keepdims=True)
    s = jnp.max(jnp.concatenate([s, jnp.ones_like(s)]), axis=0)
    return jnp.clip(denoised, -s, s) / s

def batch_expand(scalar: jnp.ndarray, imitate: jnp.ndarray):
    """ Match batch dimension and expand rank to match 'imitate' """
    out = scalar * jnp.ones(imitate.shape[0], scalar.dtype)
    return expand_dims_like(out, imitate)