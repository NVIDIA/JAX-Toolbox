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
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

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

    frgA = cute.make_fragment(cute.size(a, mode=[0]), a.element_type)
    frgB = cute.make_fragment(cute.size(b, mode=[0]), b.element_type)
    frgC = cute.make_fragment(cute.size(c, mode=[0]), c.element_type)

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
    *,
    const_a: cutlass.Constexpr,
    const_b: cutlass.Constexpr,
):
    # these two kernels are launched to the same stream.
    kernel(a, b, c, const_a, const_b).launch(
        grid=[a.shape[-1], 1, 1], block=[a.shape[-2], 1, 1], stream=stream
    )


@pytest.mark.parametrize("n", range(3))
def test_jit_sharding(n):
    ngpu = jax.device_count()
    mesh = jax.make_mesh((ngpu,), "b")
    sharding = P("b", None, None)

    key = jax.random.key(1123 + n)
    a_key, b_keys = jax.random.split(key, 2)

    shape = (32 * 8, 32, 32)
    dtype = jnp.float32
    a = create_tensor(shape, dtype, a_key)
    b = create_tensor(shape, dtype, b_keys)
    a = jax.device_put(a, NamedSharding(mesh, sharding))
    b = jax.device_put(b, NamedSharding(mesh, sharding))

    @partial(jax.jit, static_argnums=(2, 3))
    def compute(a, b, const_a, const_b):
        call = cutlass_call(
            launch,
            output_shape_dtype=jax.ShapeDtypeStruct(a.shape, b.dtype),
            input_mode=(TM(static=True), TM(static=True)),
            output_mode=TM(static=True),
            const_a=const_a,
            const_b=const_b,
        )
        ref_result = a * const_a + b * const_b
        return call(a, b), ref_result

    c, c_ref = compute(a, b, 1.0, 2.0)
    assert jnp.allclose(c, c_ref)

    c, c_ref = compute(a, b, 3.0, 4.0)
    assert jnp.allclose(c, c_ref)

    c, c_ref = compute(a, b, 1.0, 2.0)
    assert jnp.allclose(c, c_ref)


@pytest.mark.parametrize("n", range(3))
def test_shardmap(n):
    ngpu = jax.device_count()
    mesh = jax.make_mesh((ngpu,), "b")
    sharding = P("b", None, None)

    @partial(jax.jit, static_argnums=[0, 1])
    def compute(const_a, const_b):
        key = jax.random.key(1123 + n)
        a_key, b_keys = jax.random.split(key, 2)

        shape = (32 * 8, 32, 32)
        dtype = jnp.float32
        a = create_tensor(shape, dtype, a_key)
        b = create_tensor(shape, dtype, b_keys)
        a = jax.lax.with_sharding_constraint(a, NamedSharding(mesh, sharding))
        b = jax.lax.with_sharding_constraint(b, NamedSharding(mesh, sharding))

        @partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(sharding, sharding),
            out_specs=(sharding, sharding),
        )
        def sharded_call(a_block, b_block):
            call = cutlass_call(
                launch,
                output_shape_dtype=jax.ShapeDtypeStruct(a_block.shape, a_block.dtype),
                input_mode=(TM(static=True), TM(static=True)),
                output_mode=TM(static=True),
                const_a=const_a,
                const_b=const_b,
            )
            ref_result = a_block * const_a + b_block * const_b
            return call(a_block, b_block), ref_result

        return sharded_call(a, b)

    c, c_ref = compute(1.0, 2.0)
    assert jnp.allclose(c, c_ref)

    c, c_ref = compute(3.0, 4.0)
    assert jnp.allclose(c, c_ref)

    c, c_ref = compute(1.0, 2.0)
    assert jnp.allclose(c, c_ref)
