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

import pytest
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

import jax
import jax.numpy as jnp

from jax_cutlass import cutlass_call


@cute.jit
def launch(stream: cuda.CUstream, out: cute.Tensor):
    pass


def test_vjp_not_allowed():
    with pytest.raises(
        NotImplementedError, match=r".*cutlass_call does not support VJP.*"
    ):
        empty = jnp.zeros(tuple(), jnp.float32)
        call = cutlass_call(launch, output_shape_dtype=empty)
        jax.value_and_grad(call)(empty)


def test_transpose_not_allowed():
    with pytest.raises(
        NotImplementedError, match=r".*cutlass_call does not support transpose.*"
    ):
        empty = jnp.zeros(tuple(), jnp.float32)
        call = cutlass_call(launch, output_shape_dtype=empty)
        jax.linear_transpose(call, jax.ShapeDtypeStruct(empty.shape, empty.dtype))(
            empty
        )


def test_vmap_not_allowed():
    with pytest.raises(
        NotImplementedError,
        match=r".*cutlass_call does not support batching with jax\.vmap.*",
    ):
        empty = jnp.zeros(tuple(), jnp.float32)
        empty_b = jnp.zeros((8,), jnp.float32)
        call = cutlass_call(launch, output_shape_dtype=empty)
        jax.vmap(call)(empty_b)
