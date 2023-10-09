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

"""Implementation of Inception v3 network in Flax."""

import os
import pickle
import io

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
import torch
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict, unflatten_dict
import requests


# DOWNLOADED FROM:
PYT_LINK = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"

# LOCAL_PATH = "FID/pt_inception-2015-12-05-6726825d.pth"


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and ReLU activation."""

    features: int
    kernel_size: int
    strides: int = 1
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, self.kernel_size, self.strides, self.padding, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=True, momentum=0.9, epsilon=0.001)(x)
        return nn.relu(x)


class FIDInceptionA(nn.Module):
    """InceptionA module."""

    pool_features: int

    # 255,904 parameters for 288 input channels and 32 pool features.

    @nn.compact
    def __call__(self, x):
        branch1x1 = ConvBlock(features=64, kernel_size=(1, 1))(x)

        branch3x3dbl = ConvBlock(features=64, kernel_size=(1, 1))(x)
        branch3x3dbl = ConvBlock(features=96, kernel_size=(3, 3))(branch3x3dbl)
        branch3x3dbl = ConvBlock(features=96, kernel_size=(3, 3))(branch3x3dbl)

        branch5x5 = ConvBlock(features=48, kernel_size=(1, 1))(x)
        branch5x5 = ConvBlock(features=64, kernel_size=(5, 5))(branch5x5)

        branch_pool = nn.avg_pool(
            x, window_shape=(3, 3), strides=(1, 1), padding="SAME", count_include_pad=False
        )
        branch_pool = ConvBlock(features=self.pool_features, kernel_size=(1, 1))(branch_pool)

        return jax.lax.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], dimension=3)


class FIDInceptionB(nn.Module):
    """InceptionB module."""

    # 1,153,280 parameters for 288 input channels.

    @nn.compact
    def __call__(self, x):
        branch3x3 = ConvBlock(features=384, kernel_size=(3, 3), strides=2, padding="VALID")(x)

        branch3x3dbl = ConvBlock(features=64, kernel_size=(1, 1))(x)
        branch3x3dbl = ConvBlock(features=96, kernel_size=(3, 3))(branch3x3dbl)

        # No padding here in PyTorch version
        branch3x3dbl = ConvBlock(features=96, kernel_size=(3, 3), strides=2, padding="VALID")(branch3x3dbl)

        branch_pool = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="VALID")

        return jax.lax.concatenate([branch3x3, branch3x3dbl, branch_pool], dimension=3)


class FIDInceptionC(nn.Module):
    """InceptionC module."""

    channels7x7: int

    # 1,297,408 parameters for 768 input channels and 128 channels7x7.

    @nn.compact
    def __call__(self, x):
        branch1x1 = ConvBlock(features=192, kernel_size=(1, 1))(x)

        branch7x7 = ConvBlock(features=self.channels7x7, kernel_size=(1, 1))(x)
        branch7x7 = ConvBlock(features=self.channels7x7, kernel_size=(1, 7))(branch7x7)
        branch7x7 = ConvBlock(features=192, kernel_size=(7, 1))(branch7x7)

        branch7x7dbl = ConvBlock(features=self.channels7x7, kernel_size=(1, 1))(x)
        branch7x7dbl = ConvBlock(features=self.channels7x7, kernel_size=(7, 1))(branch7x7dbl)
        branch7x7dbl = ConvBlock(features=self.channels7x7, kernel_size=(1, 7))(branch7x7dbl)
        branch7x7dbl = ConvBlock(features=self.channels7x7, kernel_size=(7, 1))(branch7x7dbl)
        branch7x7dbl = ConvBlock(features=192, kernel_size=(1, 7))(branch7x7dbl)

        branch_pool = nn.avg_pool(
            x, window_shape=(3, 3), strides=(1, 1), padding="SAME", count_include_pad=False
        )
        branch_pool = ConvBlock(features=192, kernel_size=(1, 1))(branch_pool)

        return jax.lax.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], dimension=3)


class FIDInceptionD(nn.Module):
    # 1,698,304 parameters for 768 input channels.

    @nn.compact
    def __call__(self, x):
        branch3x3 = ConvBlock(features=192, kernel_size=(1, 1))(x)
        branch3x3 = ConvBlock(features=320, kernel_size=(3, 3), strides=2, padding="VALID")(branch3x3)

        branch7x7x3 = ConvBlock(features=192, kernel_size=(1, 1))(x)
        branch7x7x3 = ConvBlock(features=192, kernel_size=(1, 7))(branch7x7x3)
        branch7x7x3 = ConvBlock(features=192, kernel_size=(7, 1))(branch7x7x3)
        branch7x7x3 = ConvBlock(features=192, kernel_size=(3, 3), strides=2, padding="VALID")(branch7x7x3)

        branch_pool = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="VALID")

        return jax.lax.concatenate([branch3x3, branch7x7x3, branch_pool], dimension=3)


class FIDInceptionE_1(nn.Module):
    # 5,044,608 parameters for 1024 channels

    @nn.compact
    def __call__(self, x):
        branch1x1 = ConvBlock(features=320, kernel_size=(1, 1))(x)

        branch3x3 = ConvBlock(features=384, kernel_size=(1, 1))(x)
        branch3x3_1 = ConvBlock(features=384, kernel_size=(1, 3))(branch3x3)
        branch3x3_2 = ConvBlock(features=384, kernel_size=(3, 1))(branch3x3)
        branch3x3 = jax.lax.concatenate([branch3x3_1, branch3x3_2], dimension=3)

        branch3x3dbl = ConvBlock(features=448, kernel_size=(1, 1))(x)
        branch3x3dbl = ConvBlock(features=384, kernel_size=(3, 3))(branch3x3dbl)
        branch3x3dbl_1 = ConvBlock(features=384, kernel_size=(1, 3))(branch3x3dbl)
        branch3x3dbl_2 = ConvBlock(features=384, kernel_size=(3, 1))(branch3x3dbl)
        branch3x3dbl = jax.lax.concatenate([branch3x3dbl_1, branch3x3dbl_2], dimension=3)

        branch_pool = nn.avg_pool(
            x, window_shape=(3, 3), strides=(1, 1), padding="SAME", count_include_pad=False
        )
        branch_pool = ConvBlock(features=192, kernel_size=(1, 1))(branch_pool)

        return jax.lax.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], dimension=3)


class FIDInceptionE_2(nn.Module):
    # 6,076,800 parameters for 2048 channels

    @nn.compact
    def __call__(self, x):
        branch1x1 = ConvBlock(features=320, kernel_size=(1, 1))(x)

        branch3x3 = ConvBlock(features=384, kernel_size=(1, 1))(x)
        branch3x3_1 = ConvBlock(features=384, kernel_size=(1, 3))(branch3x3)
        branch3x3_2 = ConvBlock(features=384, kernel_size=(3, 1))(branch3x3)
        branch3x3 = jax.lax.concatenate([branch3x3_1, branch3x3_2], dimension=3)

        branch3x3dbl = ConvBlock(features=448, kernel_size=(1, 1))(x)
        branch3x3dbl = ConvBlock(features=384, kernel_size=(3, 3))(branch3x3dbl)
        branch3x3dbl_1 = ConvBlock(features=384, kernel_size=(1, 3))(branch3x3dbl)
        branch3x3dbl_2 = ConvBlock(features=384, kernel_size=(3, 1))(branch3x3dbl)
        branch3x3dbl = jax.lax.concatenate([branch3x3dbl_1, branch3x3dbl_2], dimension=3)

        branch_pool = nn.max_pool(x, window_shape=(3, 3), strides=(1, 1), padding="SAME")
        branch_pool = ConvBlock(features=192, kernel_size=(1, 1))(branch_pool)

        return jax.lax.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], dimension=3)


class FIDInceptionV3(nn.Module):
    dropout: float = 0.5
    num_classes: int = 1008

    @nn.compact
    def __call__(self, x, resize_input=False, normalize_input=False, return_featuremap=True):
        if resize_input:
            x = jax.image.resize(x, shape=(x.shape[0], 299, 299, x.shape[3]), method="bilinear")

        if normalize_input:
            x = 2 * x - 1

        # N x 3 x 299 x 299
        x = ConvBlock(features=32, kernel_size=(3, 3), strides=2, padding="VALID")(x)  # N x 32 x 149 x 149
        x = ConvBlock(features=32, kernel_size=(3, 3), padding="VALID")(x)  # N x 32 x 147 x 147
        x = ConvBlock(features=64, kernel_size=(3, 3))(x)  # N x 64 x 147 x 147
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="VALID")  # N x 64 x 73 x 73

        x = ConvBlock(features=80, kernel_size=(1, 1))(x)  # N x 80 x 73 x 73
        x = ConvBlock(features=192, kernel_size=(3, 3), padding="VALID")(x)  # N x 192 x 71 x 71
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="VALID")  # N x 192 x 35 x 35

        x = FIDInceptionA(32)(x)  # N x 256 x 35 x 35
        x = FIDInceptionA(64)(x)  # N x 288 x 35 x 35
        x = FIDInceptionA(64)(x)  # N x 288 x 35 x 35

        x = FIDInceptionB()(x)  # N x 768 x 17 x 17

        x = FIDInceptionC(128)(x)  # N x 768 x 17 x 17
        x = FIDInceptionC(160)(x)  # N x 768 x 17 x 17
        x = FIDInceptionC(160)(x)  # N x 768 x 17 x 17
        x = FIDInceptionC(192)(x)  # N x 768 x 17 x 17

        x = FIDInceptionD()(x)  # N x 1280 x 8 x 8
        x = FIDInceptionE_1()(x)  # N x 2048 x 8 x 8
        x = FIDInceptionE_2()(x)  # N x 2048 x 8 x 8

        x = jnp.mean(x, axis=(1, 2))  # Global pooling: N x 2048
        # THIS x SHOULD BE RETURNED FOR THE FID CALCULATIONS
        if return_featuremap:
            return x

        x = nn.Dense(features=self.num_classes)(x)  # N x 1008
        return x


def convert_array(pyt_w):
    """Convert array from PyTorch to Jax."""
    arr = jnp.array(pyt_w)
    if len(pyt_w.shape) == 4:  # Convolution
        return jnp.transpose(arr, (2, 3, 1, 0))
    elif len(pyt_w.shape) == 2:  # Dense
        return jnp.transpose(arr, (1, 0))
    return arr


def convert_all(pyt_params, jax_params, verbose=True):
    new_jax_params = {}
    flat_jax_params = flatten_dict(jax_params, sep=".")

    if verbose:
        print("CONVERTING WEIGHTS FROM PYT TO JAX")
        print("FOLLOWING CONVERSION WILL BE APPLIED:")
        for pyt_key, jax_key in zip(pyt_params, flat_jax_params):
            pyt_key = str(pyt_key)
            jax_key = str(jax_key)
            print(f"{pyt_key.ljust(50)} ->    {jax_key.ljust(50)}")

    for (k1, v1), (k2, v2) in zip(pyt_params.items(), flat_jax_params.items()):
        new_jax_params[k2] = convert_array(v1)
        msg = f"Tried to pass weight of {k1} {v1.shape} to {k2} {v2.shape}!"
        assert new_jax_params[k2].shape == v2.shape, msg

    new_jax_params = unflatten_dict(new_jax_params, sep=".")
    return FrozenDict(new_jax_params)


def load_pretrained_inception_v3(convert_pyt_weights=None, jax_weight_restore_path=None):
    network = FIDInceptionV3()
    assert convert_pyt_weights is not None or jax_weight_restore_path is not None, "Either pytorch or jax weights must be given"

    if convert_pyt_weights is not None:
        rnd_params = network.init(jax.random.PRNGKey(0), jnp.ones((1, 299, 299, 3)))

        pyt_params = torch.load(convert_pyt_weights)
        pyt_params_batch_stats = {k: v for k, v in pyt_params.items() if "running" in k}
        jax_batch_stats = convert_all(pyt_params_batch_stats, rnd_params["batch_stats"], verbose=True)
        pyt_params = {
            k: v for k, v in pyt_params.items() if "num_batches_tracked" not in k and "running" not in k
        }

        # Every ConvBlock in PyTorch has the following order:
        # bn.bias
        # bn.weight
        # conv.weight
        # Meanwhile, in Jax the order is reversed:
        # conv.weight
        # bn.weight
        # bn.bias
        # We fix this by reversing PyTorch triplets.

        pyt_keys = list(pyt_params)
        pyt_keys_in_groups = [pyt_keys[i: i + 3][::-1] for i in range(0, len(pyt_keys), 3)]
        pyt_keys = [key for group in pyt_keys_in_groups for key in group]
        pyt_params = {key: pyt_params[key] for key in pyt_keys}

        jax_params = convert_all(pyt_params, rnd_params["params"])
        final_jax_params = FrozenDict(params=jax_params, batch_stats=jax_batch_stats)
    elif jax_weight_restore_path is not None:
        if not os.path.exists(jax_weight_restore_path):
            # If we don't have jax weights on hand, download the pytorch ones and convert
            weights = requests.get(PYT_LINK, allow_redirects=True).content
            weights_io = io.BytesIO(weights)
            params, _ = load_pretrained_inception_v3(convert_pyt_weights=weights_io)
            with open(jax_weight_restore_path, 'wb') as f:
                pickle.dump(params, f)
            
            multihost_utils.sync_global_devices("download_inception")

        final_jax_params = pickle.load(open(jax_weight_restore_path, "rb"))
    else:
        raise NotImplementedError

    return final_jax_params, network
