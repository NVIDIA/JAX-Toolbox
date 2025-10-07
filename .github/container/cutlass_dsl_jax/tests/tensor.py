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


def reorder_modes(src: str, target: str) -> tuple[int, ...]:
    """Computes the mode given a source and target order."""
    src = tuple(src)
    target = tuple(target)
    src_map = {}
    for idx, s in enumerate(src):
        src_map[s] = idx
    return tuple([src_map[d] for d in target])


def gemm_a_major(d: str):
    """Returns order for A tensor major mode."""
    return {"k": "lmk", "m": "lkm"}[d]


def gemm_a_mode(d: str) -> tuple[int, ...]:
    """Returns mode for A tensor major mode."""
    return reorder_modes(gemm_a_major(d), "mkl")


def gemm_b_major(d: str):
    """Returns order for B tensor major mode."""
    return {"k": "lnk", "n": "lkn"}[d]


def gemm_b_mode(d: str) -> tuple[int, ...]:
    """Returns mode for B tensor major mode."""
    return reorder_modes(gemm_b_major(d), "nkl")


def gemm_c_major(d: str):
    """Returns order for C tensor major mode."""
    return {"n": "lmn", "m": "lnm"}[d]


def gemm_c_mode(d: str) -> tuple[int, ...]:
    """Returns mode for C tensor major mode."""
    return reorder_modes(gemm_c_major(d), "mnl")


def gemm_a_shape(l, m, k, major) -> tuple[int, ...]:
    """Returns shape for A tensor given major mode."""
    assert major in ("k", "m")
    shape = (l, m, k) if major == "k" else (l, k, m)
    return shape


def gemm_b_shape(l, n, k, major) -> tuple[int, ...]:
    """Returns shape for B tensor given major mode."""
    assert major in ("k", "n")
    shape = (l, n, k) if major == "k" else (l, k, n)
    return shape


def gemm_c_shape(l, m, n, major) -> tuple[int, ...]:
    """Returns shape for C tensor given major mode."""
    assert major in ("m", "n")
    shape = (l, m, n) if major == "n" else (l, n, m)
    return shape


def create_tensor(
    shape, dtype, key, *, minval=-2.0, maxval=2.0, fill_value=None, fill_arange=False
):
    if fill_arange:
        tensor = jnp.ones(shape, dtype=dtype)
        tensor = tensor * jnp.arange(tensor.size, dtype=tensor.dtype).reshape(
            tensor.shape
        )
    elif fill_value is not None:
        tensor = jnp.full(shape, fill_value, dtype=dtype)
    else:
        tensor = jax.random.uniform(key, shape, dtype=jnp.float32, minval=minval, maxval=maxval)
        tensor = tensor.astype(dtype)
    return tensor


def create_a_tensor(
    l,
    m,
    k,
    major,
    dtype,
    key,
    minval=-2.0,
    maxval=2.0,
    fill_value=None,
    fill_arange=False,
):
    shape = gemm_a_shape(l, m, k, major)
    tensor = create_tensor(
        shape,
        dtype,
        key,
        minval=minval,
        maxval=maxval,
        fill_value=fill_value,
        fill_arange=fill_arange,
    )
    return tensor


def create_b_tensor(
    l,
    n,
    k,
    major,
    dtype,
    key,
    minval=-2.0,
    maxval=2.0,
    fill_value=None,
    fill_arange=False,
):
    shape = gemm_b_shape(l, n, k, major)
    tensor = create_tensor(
        shape,
        dtype,
        key,
        minval=minval,
        maxval=maxval,
        fill_value=fill_value,
        fill_arange=fill_arange,
    )
    return tensor


def create_cd_tensor(
    l,
    m,
    n,
    major,
    dtype,
    key,
    *,
    minval=-2.0,
    maxval=2.0,
    fill_value=None,
    fill_arange=False,
):
    shape = gemm_c_shape(l, m, n, major)
    tensor = create_tensor(
        shape,
        dtype,
        key,
        minval=minval,
        maxval=maxval,
        fill_value=fill_value,
        fill_arange=fill_arange,
    )
    return tensor


def gemm_reference_einsum(
    a, b, acc_dtype, c_dtype, a_major, b_major, c_major, sf_a=None, sf_b=None
):
    a_idx = gemm_a_major(a_major)
    b_idx = gemm_b_major(b_major)
    c_idx = gemm_c_major(c_major)
    spec = f"{a_idx},{b_idx}->{c_idx}"

    # If block scaled pre-scale input at higher precision
    # Assumes we only use it for fp8 and smaller.
    if sf_a is not None:
        sf_vec_size = int(a.shape[-1] // sf_a.shape[-1])
        sf_a = jnp.repeat(sf_a, sf_vec_size, axis=-1)
        a = a.astype(jnp.float16) * sf_a.astype(jnp.float16)

    if sf_b is not None:
        sf_vec_size = int(b.shape[-1] // sf_b.shape[-1])
        sf_b = jnp.repeat(sf_b, sf_vec_size, axis=-1)
        b = b.astype(jnp.float16) * sf_b.astype(jnp.float16)

    return jax.jit(
        lambda a, b: jnp.einsum(spec, a, b, preferred_element_type=acc_dtype).astype(c_dtype)
    )(a, b)


def create_attn_tensors(b, s, hq, hkv, d, dtype, key, *, minval=-2.0, maxval=2.0, fill_value=None):
    qkey, kkey, vkey = jax.random.split(key, 3)
    return (
        create_tensor(
            (b, s, hq, d), dtype, key, minval=minval, maxval=maxval, fill_value=fill_value
        ),
        create_tensor(
            (b, s, hkv, d), dtype, key, minval=minval, maxval=maxval, fill_value=fill_value
        ),
        create_tensor(
            (b, s, hkv, d), dtype, key, minval=minval, maxval=maxval, fill_value=fill_value
        ),
    )


def attn_ref(q, k, v, is_causal: bool):
    return jax.jit(
        lambda q, k, v: jax.nn.dot_product_attention(
            q, k, v, is_causal=is_causal, implementation="cudnn"
        )
    )(q, k, v)
