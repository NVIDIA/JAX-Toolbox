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

from functools import partial, reduce
from collections import defaultdict
import pytest
import jax
import jax.numpy as jnp
import os

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from jax_cutlass import cutlass_call, jax_to_cutlass_dtype, TensorMode as TM

from ..tensor import (
    create_a_tensor,
    create_b_tensor,
    create_cd_tensor,
    gemm_reference_einsum,
    gemm_a_mode,
    gemm_b_mode,
    gemm_c_mode,
)

# Import from cutlass examples
from blackwell.grouped_gemm import GroupedGemmKernel

# Needed for int64 types
jax.config.update("jax_enable_x64", True)


class JaxGroupGemmKernel:
    """A Jax wrapper around GroupGemmKernel.

    The jax flavor of group gemm takes as input a single unified tensor and runs an aux
    kernel to extract the addresses of the groups. This allows the use of the existing
    group gemm kernel from cutlass w/o modification.
    """

    def __init__(
        self,
        a_mode,
        b_mode,
        c_mode,
        group_count,
        acc_dtype,
        is_2sm,
        mma_tile_shape_mn,
        cluster_shape_mn,
        tensormap_update_mode,
        num_tensormap_buffers,
        max_active_clusters,
        total_num_clusters,
    ):
        self._gemm = GroupedGemmKernel(
            jax_to_cutlass_dtype(acc_dtype),
            is_2sm,
            mma_tile_shape_mn,
            cluster_shape_mn,
            tensormap_update_mode,
        )
        self._a_mode = a_mode
        self._b_mode = b_mode
        self._c_mode = c_mode
        self._group_count = group_count
        self._num_tensormap_buffers = num_tensormap_buffers
        self._max_active_clusters = max_active_clusters
        self._total_num_clusters = total_num_clusters

    @partial(jax.jit, static_argnums=[0], donate_argnums=[3])
    def __call__(
        self,
        tensor_a,
        tensor_b,
        tensor_c,
        group_offsets,
        problem_sizes_mnkl,
        strides_abc,
    ):

        # Storage for tensormap in gmem
        tensormap = jnp.zeros(
            (
                self._num_tensormap_buffers,
                GroupedGemmKernel.num_tensormaps,
                GroupedGemmKernel.bytes_per_tensormap // 8,
            ),
            dtype=jnp.int64,
        )

        # Storage for pointer offsets to each tensor.
        ptrs_abc = jnp.zeros((group_offsets.shape[0], 3), jnp.int64)

        c, tmap, ptrs = cutlass_call(
            fn=self.launch,
            output_shape_dtype=(tensor_c, tensormap, ptrs_abc),
            input_output_aliases={2: 0, 7: 1, 6: 2},
            group_count=self._group_count,
            total_num_clusters=self._total_num_clusters,
            max_active_clusters=self._max_active_clusters,
            input_mode=(
                self._a_mode,
                self._b_mode,
                self._c_mode,
                None,
                None,
                None,
                None,
                None,
            ),
            output_mode=(self._c_mode, None, None),
            use_static_tensors=True,
        )(
            tensor_a,
            tensor_b,
            tensor_c,
            group_offsets,
            problem_sizes_mnkl,
            strides_abc,
            ptrs_abc,
            tensormap,
        )
        return c

    @cute.jit
    def launch(
        self,
        stream: cuda.CUstream,
        initial_a: cute.Tensor,
        initial_b: cute.Tensor,
        initial_c: cute.Tensor,
        group_offsets: cute.Tensor,
        problem_shape_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        tensormap_cute_tensor: cute.Tensor,
        *,
        group_count: cutlass.Constexpr[int],
        total_num_clusters: cutlass.Constexpr[int],
        max_active_clusters: cutlass.Constexpr[int],
    ):
        extract_tensor_address_kernel(
            group_offsets, initial_a, initial_b, initial_c, tensor_address_abc
        ).launch(
            stream=stream, grid=[tensor_address_abc.shape[0], 1, 1], block=[1, 1, 1]
        )

        self._gemm(
            initial_a,
            initial_b,
            initial_c,
            group_count,
            problem_shape_mnkl,
            strides_abc,
            tensor_address_abc,
            total_num_clusters,
            tensormap_cute_tensor,
            max_active_clusters,
            stream,
        )

    @cute.kernel
    def extract_tensor_address_kernel(
        group_offsets: cute.Tensor,
        tensor_a: cute.Tensor,
        tensor_b: cute.Tensor,
        tensor_c: cute.Tensor,
        dst: cute.Tensor,
    ):
        # mkl, nkl, mnl
        bidx, _, _ = cute.arch.block_idx()

        num_groups = group_offsets.shape[0]
        group_offset = group_offsets[bidx]
        per_expert_size = tensor_b.shape[0] // num_groups

        a_offset = (
            cute.Int64(group_offset)
            * tensor_a.stride[0]
            * tensor_a.element_type.width
            // 8
        )
        a_ptr = tensor_a.iterator.toint() + a_offset
        dst[bidx, 0] = a_ptr

        b_offset = (
            cute.Int64(bidx)
            * per_expert_size
            * tensor_b.stride[0]
            * tensor_b.element_type.width
            // 8
        )
        b_ptr = tensor_b.iterator.toint() + b_offset
        dst[bidx, 1] = b_ptr

        c_offset = (
            cute.Int64(group_offset)
            * tensor_c.stride[0]
            * tensor_c.element_type.width
            // 8
        )
        c_ptr = tensor_c.iterator.toint() + c_offset
        dst[bidx, 2] = c_ptr


@partial(jax.jit, static_argnums=[0, 1, 3])
def generate_group_sizes(
    expert_count, token_count, key, uniform_group_size=False, round_group_sizes=8
):
    if uniform_group_size:
        return jnp.array([token_count // expert_count] * expert_count)
    round_group_sizes = float(round_group_sizes)
    key1, key2 = jax.random.split(key, 2)
    v = jax.random.truncated_normal(key1, -2.0, 2.0, expert_count) + 2.0
    expert_probs = v / jnp.sum(v)
    expert_assignment = jax.random.choice(
        key2, expert_count, (token_count,), p=expert_probs
    )
    group_sizes = jnp.bincount(expert_assignment, length=expert_count)
    group_sizes = round_group_sizes * jnp.floor(
        group_sizes.astype(jnp.float32) / round_group_sizes
    )
    group_sizes = group_sizes.at[0].add(token_count - group_sizes.sum())
    return group_sizes.astype(jnp.int32)


@pytest.mark.parametrize(
    "uniform_groups",
    [pytest.param(True, id="UNIFORM"), pytest.param(False, id="RANDOM")],
)
@pytest.mark.parametrize(
    "problem_size",
    [
        pytest.param(
            (16, 8 * 1024, int(1.5 * 1024), 3 * 1024, 1), id="E16-M8192-N1536-K3072-L1"
        ),
        pytest.param(
            (128, 32 * 1024, int(1.5 * 1024), 2048, 1), id="E128-M32768-N1536-K2048-L1"
        ),
    ],
)
@pytest.mark.parametrize(
    "tensormap_update_mode",
    [
        # pytest.param(utils.TensorMapUpdateMode.GMEM, id="GMEM"),
        pytest.param(utils.TensorMapUpdateMode.SMEM, id="SMEM"),
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
    "a_dtype, b_dtype, c_dtype, acc_dtype",
    [
        pytest.param(
            jnp.float16,
            jnp.float16,
            jnp.float16,
            jnp.float32,
            id="bf16xbf16xbf16xfp32",
        ),
        pytest.param(
            jnp.float8_e4m3fn,
            jnp.float8_e4m3fn,
            jnp.float16,
            jnp.float32,
            id="fp8xfp8xf16xf32",
        ),
    ],
)
@pytest.mark.parametrize(
    "a_major, b_major, c_major",
    [
        pytest.param("k", "k", "n", id="kkn_major"),
        # pytest.param("k", "n", "n", id="knn_major"),
    ],
)
@pytest.mark.requires_device("B200")
def test_grouped_gemm(
    benchmark,
    problem_size,
    uniform_groups,
    mma_tile_shape_mn,
    cluster_shape_mn,
    tensormap_update_mode,
    is_2sm,
    a_dtype,
    b_dtype,
    c_dtype,
    acc_dtype,
    a_major,
    b_major,
    c_major,
):
    key = jax.random.key(1337)

    num_groups, m, n, k, l = problem_size

    # Skip invalid mma tile shape
    if not (
        (not is_2sm and mma_tile_shape_mn[0] in [64, 128])
        or (is_2sm and mma_tile_shape_mn[0] in [128, 256])
    ):
        raise pytest.skip(f"Skip invalid mma tiler M {mma_tile_shape_mn[0]}")

    if mma_tile_shape_mn[1] not in range(32, 257, 32):
        raise pytest.skip(f"Skip invalid mma tiler N {mma_tile_shape_mn[1]}")

    if m % (mma_tile_shape_mn[0] * cluster_shape_mn[0]) != 0:
        pytest.skip(f"Problem too small for M tiling.")

    if n % (mma_tile_shape_mn[1] * cluster_shape_mn[1]) != 0:
        pytest.skip(f"Problem too small for N tiling.")

    # Skip illegal cluster shape
    if cluster_shape_mn[0] % (2 if is_2sm else 1) != 0:
        raise pytest.skip(
            f"cluster_shape_m need align with is_2sm config {cluster_shape_mn}"
        )

    tensors_abc = []
    problem_sizes_mnkl = []
    strides_abc = []

    gkey, key = jax.random.split(key)
    group_sizes = generate_group_sizes(num_groups, m, gkey, uniform_groups)
    assert group_sizes.sum() == m, "unexpected group sizes"

    # Build separate tensors for each expert. It is expected that the total tokens will
    # sum to m. n is uniform across all experts.
    for idx in range(num_groups):
        sub_m = int(group_sizes[idx])
        akey, bkey, ckey, key = jax.random.split(key, 4)

        tensor_a = create_a_tensor(l, sub_m, k, a_major, a_dtype, akey)
        tensor_b = create_b_tensor(l, n, k, b_major, b_dtype, bkey)
        tensor_c = create_cd_tensor(l, sub_m, n, c_major, c_dtype, ckey, fill_value=0.0)
        tensors_abc.append((tensor_a, tensor_b, tensor_c))

        stride_mk_a = (k, 1) if a_major == "k" else (1, m)  # mkl
        stride_nk_b = (k, 1) if b_major == "k" else (1, n * num_groups)  # nkl
        stride_mn_c = (n, 1) if c_major == "n" else (1, m)  # mnl

        strides_abc.append([stride_mk_a, stride_nk_b, stride_mn_c])
        problem_sizes_mnkl.append(((sub_m, n, k, l)))

    # layout (num_groups, 3, 2):(6, 2, 1)
    strides_abc_tensor = jnp.array(strides_abc, dtype=jnp.int32)
    problem_sizes_mnkl_tensor = jnp.array(problem_sizes_mnkl, dtype=jnp.int32)
    group_offsets = jnp.cumsum(group_sizes) - group_sizes

    # get number of SMs by querying max active clusters with 1x1 cluster shape
    hardware_info = cutlass.utils.HardwareInfo()
    num_sms = hardware_info.get_device_multiprocessor_count()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    num_tensormap_buffers = num_sms

    def compute_total_num_clusters(problem_sizes_mnkl, cga_tile_shape_mn):
        total_num_clusters = 0
        for m, n, _, _ in problem_sizes_mnkl:
            num_clusters_mn = tuple(
                (x + y - 1) // y for x, y in zip((m, n), cga_tile_shape_mn)
            )
            total_num_clusters += reduce(lambda x, y: x * y, num_clusters_mn)
        return total_num_clusters

    def compute_cga_tile_shape(mma_tile_shape_mn, cluster_shape_mn, is_2sm):
        cta_tile_shape_mn = list(mma_tile_shape_mn)
        if is_2sm:
            cta_tile_shape_mn[0] = cta_tile_shape_mn[0] // 2
        return tuple(x * y for x, y in zip(cta_tile_shape_mn, cluster_shape_mn))

    cga_tile_shape_mn = compute_cga_tile_shape(
        mma_tile_shape_mn, cluster_shape_mn, is_2sm
    )
    total_num_clusters = compute_total_num_clusters(
        problem_sizes_mnkl, cga_tile_shape_mn
    )

    gemm = JaxGroupGemmKernel(
        gemm_a_mode(a_major),
        gemm_b_mode(b_major),
        gemm_c_mode(c_major),
        num_groups,
        acc_dtype,
        is_2sm,
        mma_tile_shape_mn,
        cluster_shape_mn,
        tensormap_update_mode,
        num_tensormap_buffers,
        max_active_clusters,
        total_num_clusters,
    )

    # Create the combined tensors by concatenating along the appropriate axis
    am_axis = gemm_a_mode(a_major)[0]  # mkl
    bn_axis = gemm_b_mode(b_major)[0]  # nkl
    cm_axis = gemm_c_mode(c_major)[0]  # mnl
    tensor_a_device = jnp.concatenate([x[0] for x in tensors_abc], axis=am_axis)
    tensor_b_device = jnp.concatenate([x[1] for x in tensors_abc], axis=bn_axis)
    tensor_c_device = jnp.concatenate([x[2] for x in tensors_abc], axis=cm_axis)

    # Note: this call setup is a bit tricky because we need to extract addresses
    # from tensor_c. To do this we donate tensor_c so we can treat it as both an
    # input and output ensuring it has a stable allocation.
    tensor_c_device = gemm(
        tensor_a_device,
        tensor_b_device,
        tensor_c_device,
        group_offsets,
        problem_sizes_mnkl_tensor,
        strides_abc_tensor,
    )

    c_ref = []
    for idx in range(num_groups):
        c_ref.append(
            gemm_reference_einsum(
                tensors_abc[idx][0],
                tensors_abc[idx][1],
                acc_dtype=acc_dtype,
                c_dtype=c_dtype,
                a_major=a_major,
                b_major=b_major,
                c_major=c_major,
            )
        )
    c_ref = jnp.concatenate(c_ref, axis=cm_axis).astype(jnp.float32)
    tensor_c_device = tensor_c_device.astype(jnp.float32)

    # Tolerance from cutedsl tests.
    assert jnp.allclose(c_ref, tensor_c_device, atol=0.1)

    with benchmark.runner("blackwell_grouped_gemm.txt") as runner:
        for _ in runner:
            tensor_c_device = gemm(
                tensor_a_device,
                tensor_b_device,
                tensor_c_device,
                group_offsets,
                problem_sizes_mnkl_tensor,
                strides_abc_tensor,
            )
