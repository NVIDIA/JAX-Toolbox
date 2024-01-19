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
Sample and conditioning augmentations for diffusion and multimodal
model training
"""

from typing import Callable, Tuple, Dict, Optional, Union
import typing_extensions

import jax
import jax.numpy as jnp
import jax.lax as lax

Augmentable = Union[jnp.ndarray, Dict[str, jnp.ndarray]]

class AugmentationCallable(typing_extensions.Protocol):
    """ Call signature for a sample augmentation function.
        Returns the augmented sample and fresh rng """
    def __call__(self,
                 to_aug: Augmentable, 
                 rng: jax.Array
                 ) -> Tuple[Augmentable, jax.Array]:
        ...

def text_conditioning_dropout(to_aug: Augmentable,
                              rng: jax.Array,
                              dropout_rate: float = 0.1,
                              drop_key: Optional[str] = None,
                              null_value = None,
                              ) -> Tuple[Augmentable, jax.Array]:
    """ 
    Can take either a dictionary, where it will dropout on the 'text_mask' key by default,
    or drop_key if supplied. If given just an array, it will dropout assuming shape = [b, ...]
    (setting to 0, or null_value if supplied)
    """
    if drop_key is None:
        drop_key = 'text_mask'
    if null_value is None:
        null_value = 0
    cond = to_aug
    if isinstance(to_aug, dict):
        cond = to_aug[drop_key]

    my_rng, rng = jax.random.split(rng)
    keep_prob = 1 - dropout_rate

    mask_shape = list(cond.shape)
    for i in range(1, len(mask_shape)):
        mask_shape[i] = 1

    mask = jax.random.bernoulli(my_rng, p=keep_prob, shape=mask_shape)
    mask = jnp.broadcast_to(mask, cond.shape)

    cond = lax.select(mask, cond, null_value * jnp.ones_like(cond))

    if isinstance(to_aug, dict):
        to_aug[drop_key] = cond
    else:
        to_aug = cond

    return to_aug, rng
