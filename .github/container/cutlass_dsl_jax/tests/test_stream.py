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

from jax_cutlass import cutlass_call, TensorMode as TM

from .tensor import create_tensor


@cute.kernel
def kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    const_a: cutlass.Constexpr,
    const_b: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    frgA = cute.make_rmem_tensor(cute.size(a, mode=[0]), a.element_type)
    frgB = cute.make_rmem_tensor(cute.size(b, mode=[0]), b.element_type)
    frgC = cute.make_rmem_tensor(cute.size(c, mode=[0]), c.element_type)

    cute.autovec_copy(a[None, tidx, bidx], frgA)
    cute.autovec_copy(b[None, tidx, bidx], frgB)
    frgC.store(frgA.load() * const_a + frgB.load() * const_b)
    cute.autovec_copy(frgC, c[None, tidx, bidx])


@cute.jit
def launch(
    stream: cuda.CUstream,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    d: cute.Tensor,
):
    # these two kernels are launched to the same stream.
    # the second call depends on the first
    kernel(a, b, c, 2.0, -3.0).launch(
        grid=[a.shape[-1], 1, 1], block=[a.shape[-2], 1, 1], stream=stream
    )
    kernel(a, c, d, -4.0, 5.0).launch(
        grid=[a.shape[-1], 1, 1], block=[a.shape[-2], 1, 1], stream=stream
    )


@pytest.mark.parametrize("n", range(3))
def test_back_to_back(n):
    def ref_call(a, b):
        c = a * 2.0 + b * -3.0
        d = a * -4.0 + c * 5.0
        return c, d

    shape = (4, 128, 128)
    dtype = jnp.float32

    for i in range(3):
        a_key, b_key = jax.random.split(jax.random.key(1123 + i), 2)

        a = create_tensor(shape, dtype, a_key)
        b = create_tensor(shape, dtype, b_key)
        c, d = cutlass_call(
            launch,
            output_shape_dtype=((a, b)),
            input_mode=[TM(static=True)] * 2,
            output_mode=[TM(static=True)] * 2,
        )(a, b)

        c_ref, d_ref = ref_call(a, b)
        assert jnp.allclose(c, c_ref, atol=1e-6), "c"
        assert jnp.allclose(d, d_ref, atol=1e-6), "d"
