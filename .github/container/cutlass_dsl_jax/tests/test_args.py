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

from typing import List, Tuple
import pytest
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

import jax
import jax.numpy as jnp

from jax_cutlass import cutlass_call

from .tensor import create_tensor


class TestConstexprArgs:
    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        y: cute.Tensor,
        z: cute.Tensor,
        const_a: cutlass.Constexpr,
        const_b: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        frgA = cute.make_fragment(cute.size(x, mode=[0]), x.element_type)
        frgB = cute.make_fragment(cute.size(y, mode=[0]), y.element_type)
        frgC = cute.make_fragment(cute.size(z, mode=[0]), z.element_type)

        cute.autovec_copy(x[None, tidx, bidx], frgA)
        cute.autovec_copy(y[None, tidx, bidx], frgB)
        frgC.store(frgA.load() * const_a + frgB.load() * const_b)
        cute.autovec_copy(frgC, z[None, tidx, bidx])

    @cute.jit
    def launch(
        self,
        stream: cuda.CUstream,
        a1: cute.Tensor,
        b1: cute.Tensor,
        c1: cute.Tensor,
        *,
        const_a: cutlass.Constexpr[float],
        const_b: cutlass.Constexpr[float]
    ):
        self.kernel(a1, b1, c1, const_a, const_b).launch(
            grid=[a1.shape[-1], 1, 1], block=[a1.shape[-2], 1, 1], stream=stream
        )

    @partial(jax.jit, static_argnums=[0, 3, 4])
    def ref_call(self, a, b, const_a, const_b):
        return a * const_a + b * const_b

    def test(self):
        shape = (4, 16, 16)
        dtype = jnp.float32
        a_key, b_key = jax.random.split(jax.random.key(1123), 2)

        a = create_tensor(shape, dtype, a_key)
        b = create_tensor(shape, dtype, b_key)

        call = partial(
            cutlass_call,
            self.launch,
            output_shape_dtype=jax.ShapeDtypeStruct(shape, dtype),
        )
        c = call(const_a=1.0, const_b=1.0)(a, b)
        c_ref = self.ref_call(a, b, 1.0, 1.0)

        c = call(const_a=4.0, const_b=-1.0)(a, b)
        c_ref = self.ref_call(a, b, 4.0, -1.0)

        # will use compile cache
        c = call(const_a=4.0, const_b=-1.0)(a, b)
        c_ref = self.ref_call(a, b, 4.0, -1.0)

        assert jnp.allclose(c, c_ref)


class TestListArgs:

    @cute.kernel
    def kernel(
        self,
        a: cute.Tensor,
        b: list[cute.Tensor],
        c: tuple[cute.Tensor, ...],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        for idx in cute.range_constexpr(len(b)):
            frgA = cute.make_fragment(cute.size(a, mode=[0]), a.element_type)
            cute.autovec_copy(a[None, tidx, bidx], frgA)
            frgB = cute.make_fragment(cute.size(b[int(idx)], mode=[0]), b[idx].element_type)
            frgC = cute.make_fragment(cute.size(c[idx], mode=[0]), c[idx].element_type)
            cute.autovec_copy(b[idx][None, tidx, bidx], frgB)
            frgC.store(frgA.load() + frgB.load())
            cute.autovec_copy(frgC, c[idx][None, tidx, bidx])

    @cute.jit
    def launch(
        self,
        stream: cuda.CUstream,
        a: cute.Tensor,
        b: list[cute.Tensor],
        c: tuple[cute.Tensor, ...],
    ):
        self.kernel(a, b, c).launch(
            grid=[a.shape[-1], 1, 1], block=[a.shape[-2], 1, 1], stream=stream
        )

    def ref_call(self, a, b):
        @partial(jax.jit)
        def _call(a, b):
            return a + b

        return [_call(a, bi) for bi in b]

    def test(self):
        key = jax.random.key(1123)
        a_key, *b_keys = jax.random.split(key, 2 + 8)

        shape = (4, 16, 16)
        dtype = jnp.bfloat16
        a = create_tensor(shape, dtype, a_key)
        b = [create_tensor(shape, dtype, k) for k in b_keys]
        c = [jax.ShapeDtypeStruct(shape, dtype) for x in b]

        call = cutlass_call(self.launch, output_shape_dtype=(c,))
        (c,) = call(a, b)

        c_ref = self.ref_call(a, b)
        for ci, ci_ref in zip(c, c_ref):
            assert jnp.allclose(ci, ci_ref)


class TestListArgsAlias:

    @cute.kernel
    def kernel(
        self,
        a: cute.Tensor,
        b: list[cute.Tensor],
        c: tuple[cute.Tensor, ...],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # Only write to the even lists
        for idx in cute.range_constexpr(0, len(b), 2):
            frgA = cute.make_fragment(cute.size(a, mode=[0]), a.element_type)
            cute.autovec_copy(a[None, tidx, bidx], frgA)
            frgB = cute.make_fragment(cute.size(b[idx], mode=[0]), b[idx].element_type)
            frgC = cute.make_fragment(cute.size(c[idx], mode=[0]), c[idx].element_type)
            cute.autovec_copy(b[idx][None, tidx, bidx], frgB)
            frgC.store(frgA.load() + frgB.load())
            cute.autovec_copy(frgC, c[idx][None, tidx, bidx])

    @cute.jit
    def launch(
        self,
        stream: cuda.CUstream,
        a: cute.Tensor,
        b: list[cute.Tensor],
        c: tuple[cute.Tensor, ...],
    ):
        self.kernel(a, b, c).launch(
            grid=[a.shape[-1], 1, 1], block=[a.shape[-2], 1, 1], stream=stream
        )

    def ref_call(self, a, b):
        @partial(jax.jit)
        def _call(a, b):
            return a + b

        results = [None] * len(b)
        for idx, bi in enumerate(b):
            if idx % 2 == 0:
                results[idx] = _call(a, bi)
            else:
                results[idx] = jnp.full(bi.shape, idx + 1, bi.dtype)
        return results

    def test(self):
        key = jax.random.key(1123)
        a_key, *b_keys = jax.random.split(key, 2 + 8)

        shape = (4, 16, 16)
        dtype = jnp.bfloat16
        a = create_tensor(shape, dtype, a_key)
        b = [create_tensor(shape, dtype, k) for k in b_keys]

        # This list of arrays will be updated by the call
        c = [jnp.full(shape, idx + 1, dtype) for idx in range(len(b))]

        call = cutlass_call(self.launch, output_shape_dtype=(c,), input_output_aliases={2: 0})
        (c,) = call(a, b, c)

        c_ref = self.ref_call(a, b)
        for ci, ci_ref in zip(c, c_ref):
            assert jnp.allclose(ci, ci_ref)
