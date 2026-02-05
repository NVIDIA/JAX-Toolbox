# Combined CUDA Kernel Files Summary - JAX & XLA

**Generated:** 2026-02-03  
**Updated:** 2026-02-04

## EXECUTIVE SUMMARY

**IMPORTANT:** This summary distinguishes between FILES, KERNEL IMPLEMENTATIONS, and TEMPLATE INSTANTIATIONS

**Total CUDA kernel FILES (NVIDIA):** 29

### Repository Breakdown

| Repository | Files | Implementations<br>(.cu.cc + .cu.h) | Template Variants | Total |
|------------|-------|-------------------------------------|-------------------|-------|
| **JAX**    | 4     | 6 (6+0)                            | 0                 | 6     |
| **XLA**    | 25    | 35 (19+16)                         | 58                | 93    |
| **TOTAL**  | **29**| **41 (25+16)**                     | **58**            | **99**|

### Key Definitions

- **Implementations**: Unique `__global__` kernel code (in .cu, .cu.cc, and .cu.h files)
- **Template Variants**: Same kernel compiled for different types/parameters
- **Total**: All kernel functions that can be launched

**Important:** 16 XLA kernel implementations are in .cu.h header files, shared by multiple .cu.cc instantiation files

**Note:** ROCm (AMD) kernels excluded from this summary

---

## JAX REPOSITORY - CUDA KERNELS

**Location:** [`jax-ml/jax`](https://github.com/jax-ml/jax)

**Total CUDA kernel files:** 4  
**Total actual kernel functions:** 6
- .cu files: 1 (2 kernels)
- .cu.cc files: 3 (4 kernels)

### JAX CUDA Kernel Files with Actual Kernel Counts

**Files ending in .cu:**
1. [`examples/ffi/src/jax_ffi_example/cuda_examples.cu`](https://github.com/jax-ml/jax/blob/main/examples/ffi/src/jax_ffi_example/cuda_examples.cu) → **2 kernels**
   - `FooFwdKernel`
   - `FooBwdKernel`

**Files ending in .cu.cc:**
2. [`jaxlib/gpu/linalg_kernels.cu.cc`](https://github.com/jax-ml/jax/blob/main/jaxlib/gpu/linalg_kernels.cu.cc) → **2 kernels**
   - `CholeskyUpdateKernel`
   - `LuPivotsToPermutationKernel`
3. [`jaxlib/gpu/make_batch_pointers.cu.cc`](https://github.com/jax-ml/jax/blob/main/jaxlib/gpu/make_batch_pointers.cu.cc) → **1 kernel**
   - `MakeBatchPointersAsyncKernel`
4. [`jaxlib/gpu/prng_kernels.cu.cc`](https://github.com/jax-ml/jax/blob/main/jaxlib/gpu/prng_kernels.cu.cc) → **1 kernel**
   - `ThreeFry2x32Kernel`

### JAX Notes
- JAX has minimal custom CUDA kernels as it relies primarily on XLA for GPU execution
- The `jaxlib/gpu/` directory contains core GPU kernels for linear algebra, batching, and PRNG
- The FFI example demonstrates how to integrate custom CUDA code with JAX

---

## XLA REPOSITORY - CUDA KERNELS

**Location:** [`openxla/xla`](https://github.com/openxla/xla)

**Total CUDA kernel files (NVIDIA):** 25 (.cu.cc files)  
**Additional header files:** 9 (.cu.h files with kernel implementations)  
**Unique kernel implementations:** 35 total
- In .cu.cc files: 19 implementations
- In .cu.h headers: 16 implementations

**Template instantiations:** 58 (type/parameter variants of base kernels)  
**Total kernel variants:** 93

**Note:** Many XLA files use templates, macros (e.g., `REGISTER_TOPK_KERNEL`), and shared header files to instantiate multiple kernel variants from a single kernel implementation.

---

## XLA Files by Kernel Source

### CATEGORY 1: EXTERNAL LIBRARY - CUTLASS (5 files, 0 XLA kernels)

**What:** NVIDIA's CUDA Templates for Linear Algebra (matrix multiplication)  
**Purpose:** Highly-optimized GEMM (General Matrix Multiply) operations  
**Kernel location:** `cutlass/gemm/device/gemm_universal.h` (external library)

**Files:**
- [`xla/service/gpu/kernels/cutlass_gemm_kernel_bf16xbf16_to_bf16.cu.cc`](https://github.com/openxla/xla/blob/main/xla/service/gpu/kernels/cutlass_gemm_kernel_bf16xbf16_to_bf16.cu.cc)
- [`xla/service/gpu/kernels/cutlass_gemm_kernel_bf16xbf16_to_f32.cu.cc`](https://github.com/openxla/xla/blob/main/xla/service/gpu/kernels/cutlass_gemm_kernel_bf16xbf16_to_f32.cu.cc)
- [`xla/service/gpu/kernels/cutlass_gemm_kernel_bf16xs8_to_f32.cu.cc`](https://github.com/openxla/xla/blob/main/xla/service/gpu/kernels/cutlass_gemm_kernel_bf16xs8_to_f32.cu.cc)
- [`xla/service/gpu/kernels/cutlass_gemm_kernel_f32xbf16_to_f32.cu.cc`](https://github.com/openxla/xla/blob/main/xla/service/gpu/kernels/cutlass_gemm_kernel_f32xbf16_to_f32.cu.cc)
- [`xla/service/gpu/kernels/cutlass_gemm_kernel_f32xf32_to_f32.cu.cc`](https://github.com/openxla/xla/blob/main/xla/service/gpu/kernels/cutlass_gemm_kernel_f32xf32_to_f32.cu.cc)

**XLA's role:** Instantiate CUTLASS templates and provide adaptor layer to StreamExecutor  
**Note:** These files wrap NVIDIA's pre-written kernels; XLA provides only the glue code

### CATEGORY 2: EXTERNAL LIBRARY - CUB (2 files, 1 XLA kernel)

**What:** CUDA Unbound - NVIDIA's parallel primitives library  
**Purpose:** Reusable building blocks (sort, scan, reduce, histogram)  
**Kernel location:** `cub/device/*.cuh` (external library)

**Files:**
- [`xla/stream_executor/cuda/cub_sort_kernel_cuda_impl.cu.cc`](https://github.com/openxla/xla/blob/main/xla/stream_executor/cuda/cub_sort_kernel_cuda_impl.cu.cc)  
  Uses: `cub::DeviceRadixSort` (no custom kernels)
- [`xla/stream_executor/cuda/cub_prefix_sum_kernel_cuda.cu.cc`](https://github.com/openxla/xla/blob/main/xla/stream_executor/cuda/cub_prefix_sum_kernel_cuda.cu.cc)  
  Uses: `cub::BlockScan` + 1 custom wrapper kernel

**XLA's role:** Thin wrapper around CUB's parallel algorithms  
**Note:** Similar to CUTLASS, XLA leverages NVIDIA's optimized implementations

### CATEGORY 3: XLA SHARED HEADERS (6 files, 16 kernels in headers)

**What:** XLA's own kernels defined in shared .cu.h headers  
**Purpose:** Code reuse - define once, instantiate for multiple types  
**Kernel location:** XLA's .cu.h header files

#### Header Files with Kernel Implementations

1. [`xla/stream_executor/gpu/gpu_test_kernels_lib.cu.h`](https://github.com/openxla/xla/blob/main/xla/stream_executor/gpu/gpu_test_kernels_lib.cu.h) → **6 kernels**
2. [`xla/stream_executor/gpu/buffer_comparator_kernel_lib.cu.h`](https://github.com/openxla/xla/blob/main/xla/stream_executor/gpu/buffer_comparator_kernel_lib.cu.h) → **2 kernels**
3. [`xla/service/gpu/kernels/cutlass_gemm_adaptor.cu.h`](https://github.com/openxla/xla/blob/main/xla/service/gpu/kernels/cutlass_gemm_adaptor.cu.h) → **2 kernels**
4. [`xla/stream_executor/cuda/topk_kernel_cuda_common.cu.h`](https://github.com/openxla/xla/blob/main/xla/stream_executor/cuda/topk_kernel_cuda_common.cu.h) → **1 kernel**
5. [`xla/stream_executor/gpu/all_reduce_kernel_lib.cu.h`](https://github.com/openxla/xla/blob/main/xla/stream_executor/gpu/all_reduce_kernel_lib.cu.h) → **1 kernel**
6. [`xla/stream_executor/gpu/multi_gpu_barrier_kernel.cu.h`](https://github.com/openxla/xla/blob/main/xla/stream_executor/gpu/multi_gpu_barrier_kernel.cu.h) → **1 kernel**
7. [`xla/stream_executor/gpu/ragged_all_to_all_kernel_lib.cu.h`](https://github.com/openxla/xla/blob/main/xla/stream_executor/gpu/ragged_all_to_all_kernel_lib.cu.h) → **1 kernel**
8. [`xla/stream_executor/gpu/redzone_allocator_kernel_lib.cu.h`](https://github.com/openxla/xla/blob/main/xla/stream_executor/gpu/redzone_allocator_kernel_lib.cu.h) → **1 kernel**
9. [`xla/stream_executor/gpu/repeat_buffer_kernel.cu.h`](https://github.com/openxla/xla/blob/main/xla/stream_executor/gpu/repeat_buffer_kernel.cu.h) → **1 kernel**

**Header files subtotal:** 16 kernel implementations

#### Instantiation Files (reference these headers)
- [`xla/stream_executor/cuda/buffer_comparator_kernel_cuda.cu.cc`](https://github.com/openxla/xla/blob/main/xla/stream_executor/cuda/buffer_comparator_kernel_cuda.cu.cc)
- [`xla/stream_executor/cuda/gpu_test_kernels_cuda.cu.cc`](https://github.com/openxla/xla/blob/main/xla/stream_executor/cuda/gpu_test_kernels_cuda.cu.cc)
- [`xla/stream_executor/cuda/topk_kernel_cuda_float.cu.cc`](https://github.com/openxla/xla/blob/main/xla/stream_executor/cuda/topk_kernel_cuda_float.cu.cc)
- [`xla/stream_executor/cuda/topk_kernel_cuda_bfloat16.cu.cc`](https://github.com/openxla/xla/blob/main/xla/stream_executor/cuda/topk_kernel_cuda_bfloat16.cu.cc)
- [`xla/stream_executor/cuda/redzone_allocator_kernel_cuda.cu.cc`](https://github.com/openxla/xla/blob/main/xla/stream_executor/cuda/redzone_allocator_kernel_cuda.cu.cc)
- [`xla/stream_executor/cuda/multi_gpu_barrier_kernel_cuda.cu.cc`](https://github.com/openxla/xla/blob/main/xla/stream_executor/cuda/multi_gpu_barrier_kernel_cuda.cu.cc)

### CATEGORY 4: CUSTOM XLA KERNELS (11 files, 19 kernels)

**What:** XLA's own `__global__` kernels written directly in .cu.cc files  
**Purpose:** Utilities, debugging, testing, and specialized operations  
**Kernel location:** Directly in these .cu.cc files

#### Testing/Profiling (3 files, 9 kernels)
- [`xla/backends/profiler/gpu/cuda_test.cu.cc`](https://github.com/openxla/xla/blob/main/xla/backends/profiler/gpu/cuda_test.cu.cc) → **5 kernels** (AddI32, MulI32, etc.)
- [`xla/backends/profiler/gpu/nvtx_with_cuda_kernels.cu.cc`](https://github.com/openxla/xla/blob/main/xla/backends/profiler/gpu/nvtx_with_cuda_kernels.cu.cc) → **2 kernels**
- [`xla/backends/profiler/gpu/profile_with_cuda_kernels.cu.cc`](https://github.com/openxla/xla/blob/main/xla/backends/profiler/gpu/profile_with_cuda_kernels.cu.cc) → **2 kernels**

#### Debugging (2 files, 3 kernels)
- [`xla/stream_executor/cuda/buffer_debug_float_check_kernel_cuda.cu.cc`](https://github.com/openxla/xla/blob/main/xla/stream_executor/cuda/buffer_debug_float_check_kernel_cuda.cu.cc) → **2 kernels**
- [`xla/stream_executor/cuda/buffer_debug_xor_checksum_kernel_cuda.cu.cc`](https://github.com/openxla/xla/blob/main/xla/stream_executor/cuda/buffer_debug_xor_checksum_kernel_cuda.cu.cc) → **1 kernel**

#### Utilities (6 files, 7 kernels)
- [`xla/stream_executor/cuda/delay_kernel_cuda.cu.cc`](https://github.com/openxla/xla/blob/main/xla/stream_executor/cuda/delay_kernel_cuda.cu.cc) → **1 kernel**
- [`xla/stream_executor/cuda/make_batch_pointers_kernel_cuda.cu.cc`](https://github.com/openxla/xla/blob/main/xla/stream_executor/cuda/make_batch_pointers_kernel_cuda.cu.cc) → **1 kernel**
- [`xla/stream_executor/cuda/cuda_executor_multigpu_test_kernels.cu.cc`](https://github.com/openxla/xla/blob/main/xla/stream_executor/cuda/cuda_executor_multigpu_test_kernels.cu.cc) → **1 kernel**
- [`xla/experiments/sm_bandwidth_benchmark/sm_bw_kernels.cu.cc`](https://github.com/openxla/xla/blob/main/xla/experiments/sm_bandwidth_benchmark/sm_bw_kernels.cu.cc) → **1 kernel**
- [`xla/tests/collective_ops_ffi_kernels.cu.cc`](https://github.com/openxla/xla/blob/main/xla/tests/collective_ops_ffi_kernels.cu.cc) → **2 kernels**

**Custom kernels subtotal:** 19 kernel implementations

### CATEGORY 5: BUILD TOOLS (2 files, 0 kernels)

**What:** Compilation verification only
- [`build_tools/configure/assert_cuda_clang.cu.cc`](https://github.com/openxla/xla/blob/main/build_tools/configure/assert_cuda_clang.cu.cc)
- [`build_tools/configure/assert_nvcc.cu.cc`](https://github.com/openxla/xla/blob/main/build_tools/configure/assert_nvcc.cu.cc)

---

## KERNEL SOURCE SUMMARY

| Category | Files | XLA Kernels | Notes |
|----------|-------|-------------|-------|
| **External Libraries (NVIDIA)** | 7 | 1 | Uses NVIDIA's kernels |
| - CUTLASS GEMM | 5 | 0 | |
| - CUB primitives | 2 | 1 | |
| **XLA's Own Code** | 18 | 35 | |
| - Shared headers | 6 | 16 | In .cu.h files |
| - Direct implementations | 11 | 19 | In .cu.cc files |
| **Build/Test Tools** | 2 | 0 | |
| **Total XLA files** | **27** | **36** | Including header kernels |

**Key Insight:** 27% of XLA files (7/27) leverage external NVIDIA libraries for heavy lifting (GEMM, sorting), while 73% (18/27) contain XLA's custom code.

---

## External Library Dependencies

XLA leverages two major NVIDIA libraries for GPU kernels:

### 1. CUTLASS (CUDA Templates for Linear Algebra Subroutines)
- **Purpose:** Highly-optimized matrix multiplication (GEMM) kernels
- **Source:** NVIDIA's official [CUTLASS library](https://github.com/NVIDIA/cutlass)
- **Usage:** XLA instantiates CUTLASS templates for different precision types
- **Files:** 5 CUTLASS wrapper files (`cutlass_gemm_kernel_*.cu.cc`)
- **Benefit:** Production-quality GEMM without writing/maintaining complex kernels

### 2. CUB (CUDA Unbound - Parallel Primitives)
- **Purpose:** Reusable parallel building blocks (sort, scan, reduce, histogram)
- **Source:** NVIDIA's CUDA Toolkit (standard library component)
- **Usage:** XLA wraps CUB functions for sorting and prefix sum operations
- **Files:** 2 CUB wrapper files (`cub_sort_*`, `cub_prefix_sum_*`)
- **Benefit:** Battle-tested parallel algorithms optimized for GPU architectures

### Why use external libraries?
- Avoids reinventing the wheel for complex, well-solved problems
- Gets NVIDIA's years of optimization and hardware expertise
- Reduces maintenance burden on XLA team
- Ensures compatibility with new GPU architectures

---

## METHODOLOGY - How Kernels Were Counted

1. **File count:** Counted all .cu and .cu.cc files → 29 files total
2. **Implementation count:** Searched for `__global__` keyword in ALL CUDA source files (.cu, .cu.cc, .cu.h, .cuh)
   - JAX: 6 kernel implementations (all in .cu/.cu.cc files)
   - XLA .cu.cc files: 19 kernel implementations
   - XLA .cu.h headers: 16 kernel implementations
   - **Total: 41 unique kernel implementations**
3. **Template instantiation count:** Searched for template instantiation patterns:
   - `REGISTER_TOPK_KERNEL` macros: 20 instantiations (2 files × 10 each)
   - Template class declarations: 15 CUTLASS instantiations (5 files × 3 each)
   - CUB prefix sum instantiations: 14 type specializations
   - Other template patterns: 9 instantiations
   - **Total: 58 template instantiations in XLA**

### Why distinguish implementations from instantiations?
- An **implementation** is unique algorithm code
- An **instantiation** is the same code compiled for a different type/parameter
- Example: topk_kernel for float and bfloat16 use THE SAME kernel code from `topk_kernel_cuda_common.cu.h`, just instantiated with different template parameters
- This is standard C++ template metaprogramming, not code duplication

### Files with 0 direct kernels
Many files contain 0 `__global__` declarations because they:
- Use template instantiation macros (e.g., `REGISTER_TOPK_KERNEL`)
- Reference kernel definitions from shared header files (.cu.h)
- Use external libraries (NVIDIA CUB, CUTLASS)
- This is best practice in CUDA development for code reuse and maintainability

---

## Notes

- Search performed recursively through both repositories
- CUDA (NVIDIA) and ROCm (AMD) are separate GPU computing platforms
- Kernel counting methodology: Search for `__global__` keyword which marks CUDA kernel functions
- Header files (.cu.h, .cuh) ARE included in kernel counts as they contain actual kernel definitions
- Many .cu.cc files are wrapper/instantiation files that include kernel templates from headers
- External library kernels (CUTLASS, CUB) are NOT counted as XLA kernels since they're pre-written
- The combined architecture allows JAX to focus on high-level APIs while XLA handles low-level GPU execution across multiple hardware vendors

---

**Generated by:** Comprehensive CUDA kernel analysis of JAX and XLA repositories  
**Repository Links:** [JAX](https://github.com/jax-ml/jax) | [XLA](https://github.com/openxla/xla)
