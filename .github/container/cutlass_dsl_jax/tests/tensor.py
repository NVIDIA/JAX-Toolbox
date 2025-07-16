# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from functools import partial
import jax
import jax.numpy as jnp


def create_tensor(shape, dtype, key, *, minval=-2.0, maxval=2.0, fill_value=None):
    if fill_value is not None:
        tensor = jnp.full(shape, fill_value, dtype=dtype)
    else:
        tensor = jax.random.uniform(key, shape, dtype=dtype, minval=minval, maxval=maxval)
    return tensor
