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

from jax_cutlass import cutlass_call, jax_to_cutlass_dtype

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

from blackwell.dense_blockscaled_gemm_persistent import (
    Sm100BlockScaledPersistentDenseGemmKernel,
)


@pytest.mark.parametrize(
    "problem_size",
    [
        pytest.param((8 * 1024, 8 * 1024, 8 * 1024, 1), id="M8092-N8092-K8092-L1"),
        pytest.param((8 * 1024, 4 * 1024, 4 * 1024, 1), id="M8092-N4096-K4096-L1"),
        pytest.param((16 * 1024, 16 * 1024, 16 * 1024, 1), id="M16K-N16K-K16-L1"),
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
        # pytest.param(False, (1, 1), id="1SM-1x1"),
        pytest.param(False, (2, 1), id="1SM-2x1"),
        # pytest.param(False, (2, 2), id="1SM-2x2"),
        # pytest.param(False, (4, 1), id="1SM-4x1"),
        pytest.param(True, (2, 1), id="2SM-2x1"),
        # pytest.param(True, (2, 2), id="2SM-2x2"),
        # pytest.param(True, (4, 1), id="2SM-4x1"),
    ],
)
@pytest.mark.parametrize(
    "ab_dtype, c_dtype, sf_dtype, sf_vec_size",
    [
        pytest.param(
            "float4_e2m1fn", "float16", "float8_e8m0fnu", 16, id="mxfp4xmxfp4xf16"
        ),
        pytest.param(
            "float4_e2m1fn", "float16", "float8_e4m3fn", 16, id="nvfp4xnvfp4xf16"
        ),
    ],
)
@pytest.mark.parametrize(
    "a_major, b_major, c_major",
    [
        # n.b. only k major a/b is supported by this test fixture.
        pytest.param("k", "k", "n", id="kkn_major"),
    ],
)
@pytest.mark.requires_device("B200")
def test_dense_block_scaled_gemm(
    benchmark,
    problem_size,
    mma_tile_shape_mn,
    is_2sm,
    cluster_shape_mn,
    ab_dtype,
    c_dtype,
    sf_dtype,
    sf_vec_size,
    a_major,
    b_major,
    c_major,
):
    def ceil_div(a, b):
        return (a + b - 1) // b

    m, n, k, l = problem_size
    sf_k = ceil_div(k, sf_vec_size)

    if not Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
        jax_to_cutlass_dtype(ab_dtype),
        jax_to_cutlass_dtype(sf_dtype),
        sf_vec_size,
        jax_to_cutlass_dtype(c_dtype),
        mma_tile_shape_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
        a_major,
        b_major,
        c_major,
    ):
        pytest.skip(
            f"Sm100BlockScaledPersistentDenseGemmKernel does not support test config."
        )

    if not is_2sm and mma_tile_shape_mn[0] not in (64, 128):
        pytest.skip(f"Skipping {is_2sm=} {mma_tile_shape_mn=}")

    akey, asfkey, bkey, bsfkey = jax.random.split(jax.random.key(1337), 4)
    a = create_a_tensor(l, m, k, a_major, ab_dtype, akey, minval=-1.0, maxval=1.0)
    b = create_b_tensor(l, n, k, b_major, ab_dtype, bkey, minval=-2.0, maxval=2.0)

    assert a_major == "k", "a_major must be k"
    assert b_major == "k", "b_major must be k"

    # See https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
    # Scale factors are using .scale_vec::4X / .block16 config to support nvfp4 and mxfp4
    atom_mn = (32, 4)
    atom_k = 4

    sfa = create_a_tensor(l, m, sf_k, a_major, sf_dtype, asfkey, minval=1.0, maxval=3.0)
    sfa_ref = sfa
    sfa = sfa.reshape(
        l,
        ceil_div(m, atom_mn[0] * atom_mn[1]),
        atom_mn[1],
        atom_mn[0],
        ceil_div(sf_k, atom_k),
        atom_k,
    )
    # TODO: See if we can pass this layout mapping from jax primitive (it requires grouping)
    sfa = sfa.transpose(0, 1, 4, 3, 2, 5)

    sfb = create_b_tensor(l, n, sf_k, b_major, sf_dtype, bsfkey, minval=1.0, maxval=3.0)
    sfb_ref = sfb
    sfb = sfb.reshape(
        l,
        ceil_div(n, atom_mn[0] * atom_mn[1]),
        atom_mn[1],
        atom_mn[0],
        ceil_div(sf_k, atom_k),
        atom_k,
    )
    sfb = sfb.transpose(0, 1, 4, 3, 2, 5)

    gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        sf_vec_size,
        mma_tile_shape_mn,
        cluster_shape_mn,
    )

    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    def launch(a, b, sfa, sfb):
        call = (
            lambda stream, a, b, sfa, sfb, c, *, max_active_clusters, epilogue_op: gemm(
                a, b, sfa, sfb, c, max_active_clusters, stream, epilogue_op
            )
        )
        return cutlass_call(
            call,
            input_mode=(gemm_a_mode(a_major), gemm_b_mode(b_major), None, None),
            output_mode=(gemm_c_mode(c_major),),
            output_shape_dtype=jax.ShapeDtypeStruct(
                gemm_c_shape(l, m, n, c_major), c_dtype
            ),
            epilogue_op=lambda x: x,
            max_active_clusters=max_active_clusters,
        )(a, b, sfa, sfb)

    c = launch(a, b, sfa, sfb)

    c_ref = gemm_reference_einsum(
        a,
        b,
        acc_dtype=jnp.float16,
        c_dtype=c_dtype,
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        sf_a=sfa_ref,
        sf_b=sfb_ref,
    )

    assert jnp.allclose(c, c_ref)

    with benchmark.runner("blackwell_dense_block_scaled_gemm.txt") as runner:
        runner(launch, a, b, sfa, sfb)
