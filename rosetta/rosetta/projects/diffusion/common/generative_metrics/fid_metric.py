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
This implementation is basing on: https://github.com/mseitzer/pytorch-fid
Which is not the original implementation, but is widely used as the best PyTorch port of the original TF version.
""" 
import jax
from rosetta.projects.diffusion.common.generative_metrics.inception_v3 import load_pretrained_inception_v3
import jax.numpy as jnp
from jax.scipy import linalg
from tqdm import tqdm
import logging
import time

import numpy as np


def get_activations(params, model, batches):
    """Calculates the activations of the pool_3 layer for all images.

    Returns:
    -- A jax array of dimension (num images, dims) that contains the activations of the batches.
    """
    all_outs = []
    for batch in tqdm(batches, desc="Calculating activations"):
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = batch[0]

        batch = jnp.array(batch)

        if batch.shape[1] == 3:
            batch = batch.transpose(0, 2, 3, 1)

        logging.info("batched")
        outs = model(params, batch)
        all_outs.append(outs)
        logging.info("Completed 1 batch")
        time.sleep(1.0)

    all_outs = jnp.concatenate(all_outs, axis=0)
    return all_outs


def calculate_activation_statistics(params, model, batches, num_pads=0):
    """Calculation of the statistics used by the FID.

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of the inception model.
    """
    act = get_activations(params, model, batches)
    act = act[:act.shape[0] - num_pads]
    mu = jnp.mean(act, axis=0)
    sigma = jnp.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = jnp.atleast_1d(mu1)
    mu2 = jnp.atleast_1d(mu2)

    sigma1 = jnp.atleast_2d(sigma1)
    sigma2 = jnp.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not jnp.isfinite(covmean).all():
        msg = 'fid calculation produces singular product; adding %s to diagonal of cov estimates' % eps
        print(msg)
        offset = jnp.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if jnp.iscomplexobj(covmean):
        if not jnp.allclose(jnp.diagonal(covmean).imag, 0, atol=1e-3):
            m = jnp.max(jnp.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = jnp.trace(covmean)

    return diff.dot(diff) + jnp.trace(sigma1) + jnp.trace(sigma2) - 2 * tr_covmean


def fid(samples1, samples2, inception_weight_path, inception_batch_size=32):
    """Load pretrained Inception and calculate the FID of two set of batches"""
    params, inception = load_pretrained_inception_v3(jax_weight_restore_path=inception_weight_path)

    def pad_and_batch_array(array, divisor=inception_batch_size):
        remainder = divisor - (array.shape[0] % divisor)
        pad_instances = jnp.repeat(array[:1], remainder, axis=0)
        num_batches = (array.shape[0] + remainder) // divisor
        array = jnp.concatenate((array, pad_instances), axis=0)
        return array.reshape((num_batches, divisor, *(array.shape[1:]))), remainder

    def run(params, batch):
        return inception.apply(
            params,
            batch,
            resize_input=True,
            normalize_input=True,
            return_featuremap=True,
        )

    jitted_run = jax.jit(run)
    logging.info("Jitted run")

    m1, s1 = calculate_activation_statistics(params, jitted_run, samples1)
    m2, s2 = calculate_activation_statistics(params, jitted_run, samples2)
    cpu_device = jax.devices("cpu")[0]
    m1 = jax.device_put(m1, cpu_device)
    s1 = jax.device_put(s1, cpu_device)
    m2 = jax.device_put(m2, cpu_device)
    s2 = jax.device_put(s2, cpu_device)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    jax.debug.print(f'fid_value {fid_value}')
    return fid_value