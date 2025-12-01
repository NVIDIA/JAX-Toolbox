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
from collections import defaultdict
from typing import List, Type, Tuple, Union, Optional
import os

import pytest
import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from jax_cutlass import cutlass_call, jax_to_cutlass_dtype, TensorMode as T

from ..tensor import (
    create_a_tensor,
    create_b_tensor,
    create_cd_tensor,
    gemm_a_mode,
    gemm_b_mode,
    gemm_c_mode,
    gemm_c_shape,
    gemm_reference_einsum,
)

from blackwell.dense_gemm_persistent import PersistentDenseGemmKernel


@pytest.mark.parametrize(
    "problem_size",
    [
        pytest.param((8 * 1024, 8 * 1024, 8 * 1024, 1), id="M8092-N8092-K8092-L1"),
        # pytest.param((8 * 1024, 4 * 1024, 4 * 1024, 1), id="M8092-N4096-K4096-L1"),
        # pytest.param((16 * 1024, 16 * 1024, 16 * 1024, 1), id="M16K-N16K-K16-L1"),
    ],
)
@pytest.mark.parametrize(
    "mma_tile_shape_mn",
    [
        pytest.param((128, 128), id="MMA_128x128"),
        # pytest.param((256, 128), id="MMA_256x128"),
        # pytest.param((256, 256), id="MMA_256x256"),
    ],
)
@pytest.mark.parametrize(
    "is_2sm, cluster_shape_mn",
    [
        pytest.param(False, (1, 1), id="1SM-1x1"),
        # pytest.param(False, (2, 1), id="1SM-2x1"),
        # pytest.param(False, (2, 2), id="1SM-2x2"),
        # pytest.param(False, (4, 1), id="1SM-4x1"),
        pytest.param(True, (2, 1), id="2SM-2x1"),
        # pytest.param(True, (2, 2), id="2SM-2x2"),
        # pytest.param(True, (4, 1), id="2SM-4x1"),
    ],
)
@pytest.mark.parametrize(
    "use_tma_store",
    [
        pytest.param(False, id="NTS"),
        pytest.param(True, id="TS"),
    ],
)
@pytest.mark.parametrize(
    "a_dtype, b_dtype, c_dtype, acc_dtype",
    [
        pytest.param(
            "float16", "float16", "float16", "float32", id="bf16xbf16xbf16xfp32"
        ),
        pytest.param(
            "float8_e4m3fn", "float8_e4m3fn", "float16", "float32", id="fp8xfp8xf16xf32"
        ),
    ],
)
@pytest.mark.parametrize(
    "a_major, b_major, c_major",
    [
        pytest.param("k", "k", "n", id="kkn_major"),
        # pytest.param("m", "n", "n", id="mnn_major"),
        # pytest.param("m", "n", "m", id="mnm_major"),
    ],
)
@pytest.mark.requires_device("B200")
def test_dense_gemm(
    benchmark,
    problem_size,
    mma_tile_shape_mn,
    is_2sm,
    cluster_shape_mn,
    use_tma_store,
    a_dtype,
    b_dtype,
    c_dtype,
    acc_dtype,
    a_major,
    b_major,
    c_major,
):
    if not is_2sm and mma_tile_shape_mn[0] not in (64, 128):
        pytest.skip(f"Skipping {is_2sm=} {mma_tile_shape_mn=}")

    m, n, k, l = problem_size

    akey, bkey = jax.random.split(jax.random.key(1337), 2)
    a = create_a_tensor(l, m, k, a_major, a_dtype, akey)
    b = create_b_tensor(l, n, k, b_major, b_dtype, bkey)

    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    gemm = PersistentDenseGemmKernel(
        jax_to_cutlass_dtype(acc_dtype),
        is_2sm,
        mma_tile_shape_mn,
        cluster_shape_mn,
        use_tma_store,
    )
    call = lambda stream, a, b, c, **kwargs: gemm(
        a, b, c, max_active_clusters, stream, **kwargs
    )

    def launch(a, b):
        return cutlass_call(
            call,
            input_mode=(gemm_a_mode(a_major), gemm_b_mode(b_major)),
            output_mode=(gemm_c_mode(c_major),),
            output_shape_dtype=jax.ShapeDtypeStruct(
                gemm_c_shape(l, m, n, c_major), c_dtype
            ),
            epilogue_op=lambda x: x,
        )(a, b)

    c = launch(a, b)
    c_ref = gemm_reference_einsum(
        a,
        b,
        acc_dtype=acc_dtype,
        c_dtype=c_dtype,
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
    )
    assert jnp.allclose(c, c_ref)

    with benchmark.runner("blackwell_dense_gemm.txt") as runner:
        runner(launch, a, b)
