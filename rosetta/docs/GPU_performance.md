# Tips for High-Performance LLMs with JAX and XLA 

This page documents the various flags in XLA and JAX to improve performance for LLMs on GPUs. The XLA flags are defined with their default values in [xla/debug_options_flags.cc](https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc)

The flags can be set via the environment variable `XLA_FLAGS="--xla-flag1=true --xla-flag2=false"` on command line or your script.

Please note that some of these flags are experimental. All combinations of flags have not been tested, yet. If you see any unexpected behaviors, please let us know.


## Quick lookup by symptom

**Poor compute/communication overlap**
- Use Optimization level (O1) → see [Optimization level](#optimization-level)
- Consider manual async overrides (disable specific collectives, enable memcpy local P2P) → see [Asynchronous collectives](#xla-flags-for-asynchronous-collective-communication)
- Force pipelined AR/AG/RS → see [Optimization level](#optimization-level)
- Enable while-loop double buffering → see [Optimization level](#optimization-level)

**Many small collectives or low bandwidth**
- Tune combine thresholds (AR/AG/RS) → see [FSDP optimization flags](#flags-to-enable-optimizations-for-fsdp-communication)

**Slow collectives within a node (NVLINK/NVSwitch)**
- Provide perf table or use built-in default → see [Optimization level](#optimization-level)
- Enable memcpy-based local P2P → see [Asynchronous collectives](#xla-flags-for-asynchronous-collective-communication)

**Slow cross-node collectives (DCN)**
- Review SOL latency estimator notes and adjust as needed → see [Optimization level](#optimization-level)

**High CPU launch overhead**
- Use CUDA Graphs (command buffer) → see [CUDA graphs](#cuda-graphs)

**OOMs or fragmentation**
- Set XLA_PYTHON_CLIENT_MEM_FRACTION → see [Memory management](#flags-to-manage-memory-used-in-jaxxla)
- Tune xla_gpu_memory_limit_slop_factor → see [Memory management](#flags-to-manage-memory-used-in-jaxxla)

**Multi-device single-process hangs**
- Consider `NCCL_LAUNCH_MODE=GROUP` or one process per device → see [Blackwell tips](#tips-for-good-llm-training-performance-on-blackwell-b200)

**Blackwell-specific performance questions**
- Prefer command buffer for FUSION,CUSTOM_CALL; do not set `CUDA_DEVICE_MAX_CONNECTIONS=1` → see [Blackwell tips](#tips-for-good-llm-training-performance-on-blackwell-b200)


## Flags to manage memory used in JAX/XLA

- XLA_PYTHON_CLIENT_MEM_FRACTION is a XLA environment variable that allocates a fraction of GPU memory for JAX/XLA.
--  Ideally, should be 1, but in practice less because some memory is used by NVIDIA Libraries, and the JAX framework.
--  We typically set it to 0.9 or 0.8. At 0.9, XLA gets 90% of GPU memory.

- The `xla_gpu_memory_limit_slop_factor` flag controls the memory used by XLA for determining its default heuristics for scheduling, and rematerialization. Default is recommended.


## General CUDA/NCCL flags 

### CUDA configuration

The following environment variable restricts CUDA queues to 1 and is useful when a strict ordering of operations is required to achieve best performance. This is recommended to achieve good performance with latency hiding optimizations with asynchronous collectives.
- CUDA_DEVICE_MAX_CONNECTIONS=1
  
### NCCL configuration 

See [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) for more details.
- NCCL_PROTO: SIMPLE,LL,LL128

The following variable accelerates all-reduce collective on NVLink4/H100. It requires additional GPU memory and may need one to reduce `XLA_PYTHON_CLIENT_MEM_FRACTION` to avoid OOMs if enabled.
- NCCL_NVLS_ENABLE:1 


## XLA flags for asynchronous collective communication

To enable more efficient P2P transfers utilizing Copy Engine, turn on the following flag. Note that this will enable Copy Engine transfers only for devices managed within a single process.

- --xla_gpu_use_memcpy_local_p2p=true

For more fine-grained control over which collectives should be asynchronous or not, please use: 

- --xla_gpu_disable_async_collectives=allreduce,allgather,reducescatter,collectivebroadcast,alltoall,collectivepermute

## Optimization level

Optimization level is a high-level knob that enables a bundle of XLA GPU optimizations that generally improve runtime perfomance at the cost of increasing compile time. Using optimization level reduces manual flag tuning, increases compute/communication overlap, and improves out-of-the-box performance while keeping behavior predictable.

- Set via JAX (not an XLA_FLAGS switch):
```python
import jax
jax.config.update("jax_optimization_level", "O1")
```

- Or via environment variable:
```
JAX_OPTIMIZATION_LEVEL=O1 python your_script.py
```

### Effect details (O1):
  - Latency Hiding Scheduler (LHS): previously required `--xla_gpu_enable_latency_hiding_scheduler=true`; now enabled by O1. To force explicitly:

    ```
    --xla_gpu_enable_latency_hiding_scheduler=true
    ```
  - Collective pipelining: previously set via flags; now enabled by O1. To force explicitly:

    ```
    --xla_gpu_enable_pipelined_all_reduce=true
    --xla_gpu_enable_pipelined_all_gather=true
    --xla_gpu_enable_pipelined_reduce_scatter=true
    ```
  - Unified SOL latency estimator:
    - Enabled on Hopper/Blackwell at O1 when supported. Collectives within a single NVLink domain use a perf table; collectives spanning multiple NVLink domains use the analytical SOL model.
    - To explicitly enable or tune the SOL latency estimator:
      ```
      --xla_gpu_enable_analytical_sol_latency_estimator=true
      --xla_gpu_experimental_collective_perf_table_path=/path/to/collective_profiles.pbtxt  # optional for NVLink
      --xla_gpu_analytical_latency_estimator_options="nccl_op_launch_us=45,nic_speed_gbps=25.0,chunk_prep_us=6,rtt_us=31,gpus_per_node=8,chunk_size_bytes=4194304"  # H100 defaults from XLA SoL; optional for InfiniBand etc.
      ```
      If no perf table is provided or found for the device, SOL falls back to the approximate estimator for NVLink collectives.
  - While-loop unrolling: with the default enum `AUTO`, O1 enables unrolling that can realize double-buffering. To force it regardless of O1, set:
    
    ```
    --xla_gpu_enable_while_loop_double_buffering=true
    ```
    
    or set the enum to DOUBLE_BUFFER/FULL_UNROLL.

### Flags to enable optimizations for FSDP communication 

With FSDP in JAX/XLA, the following knobs can further tune behavior beyond what optimization level O1 enables:

- Combine threshold values in XLA that determine when an all-gather (AG) or reduce-scatter (RS) is triggered. We want to set these values to be at least as large as the size of weights (AG) or gradients (RS) in a single transformer layer since large communication buffers achieve higher link bandwidth utilization. For example, LLAMA2-7B with BF16 weights and gradients, we have 32 transformer layers => each layer has ~218M weights => one would want to set these thresholds to at least 436MB.
    - --xla_gpu_all_gather_combine_threshold_bytes=8589934592
    - --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
      
- Combine threshold values in XLA that determine when an all-reduce (AR) is triggered. Typically, used to overlap AR of gradients with back-prop of compute. We want to set this to be at least as large as possible to achieve high efficiency, but as small as possible to achieve maximum overlap. Depending on the interconnect of your system, one might want to try several threshold values in steps of 2 from say 16MB to total gradient size.
    - --xla_gpu_all_reduce_combine_threshold_bytes=8589934592


### Flags to enable async collective permute 

The following flags enable overlap of pipeline parallel communication of send/recv with computation. 
- --xla_gpu_enable_pipelined_p2p=true  (false by default)
- --xla_gpu_collective_permute_decomposer_threshold=1024
- --xla_gpu_lhs_enable_gpu_async_tracker=true

### Flags to enable collective matmul

The following flags enable overlap of tensor parallel communication with GEMMs/matmul by splicing GEMMs into smaller chunks and triggering each chunks' collective right after the chunk's GEMM is done. The threshold determines the size of output buffer of GEMM when this optimization becomes active (0 enables collective matmul for all GEMM-collective patterns)
- --xla_gpu_multi_streamed_windowed_einsum=true
- --xla_gpu_threshold_for_windowed_einsum_mib=0

### Profile Guided Latency Estimator (PGLE)

The following flag enables use of PGLE with JAX/XLA. Please see [PGLE notes](https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/PGLE.md) for more details.
- --xla_gpu_pgle_profile_file_or_directory_path=filename

## Other XLA Flags 

### CUDA graphs

The below enables CUDA Graph suppport for JAX/XLA workloads, and is enabled by default.
- --xla_gpu_enable_command_buffer (Set to "" to disable)



### Dynamic-Update Slice Fusion

The following flag removes extra copies introduced by DUS (dynamic update slice) when used in conjunction with custom NVIDIA kernels (like cuBLAS for GEMMs). This happens in particular when used with scan operations.
- --xla_gpu_enable_custom_fusions=true
- --xla_gpu_enable_address_computation_fusion=true

### NCCL Optimizations

Enable user-buffers in NCCL for zero-copy collectives and send/recv. Needs NCCL_NVLS_ENABLE=1 for AG, AR, RS.
- --xla_gpu_enable_nccl_user_buffers=true

When user-buffers is enabled, a separate memory pool is created for user-buffer registered memory. Environment variable `XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB` can be used to configure this memory pool. It may also be necessary to reduce `XLA_PYTHON_CLIENT_MEM_FRACTION` to ensure there is enough memory for the user buffer pool.
- `XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB=0` (default value) - The user buffer pool will start empty, but will grow during execution as more collective memory is required. This setting can result in extra fragmentation and inefficient memory use.
- `XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB=<amount of MiB to preallocate>` - The user buffer pool will preallocate this amount of memory at the begining. The number should be high enough to cover peak collective memory usage.


Flags to reduce memory consumed by NCCL.
- --xla_gpu_enable_nccl_comm_splitting=true  
- --xla_gpu_enable_nccl_per_stream_comms=false [https://github.com/openxla/xla/pull/9845](https://github.com/openxla/xla/pull/9845)

Fine-grain control to improve performance by initializing a NCCL communicator to use only max_nchannels (SMs). Default value of 0 gets the default values from NCCL for SMs used per collective.
- --xla_gpu_nccl_collective_max_nchannels
- --xla_gpu_nccl_p2p_max_nchannels

### Debug flags 
- --xla_dump_to=some/path
- --xla_dump_latency_hiding_schedule=true

### Miscellaneous flags 
- --xla_gpu_cudnn_gemm_fusion=true (enables GEMM/bias fusion via cuDNN)
- --xla_gpu_enable_cudnn_fmha=false (enables XLA pattern matcher to detect multi-headed attention pattern in JAX)
- --xla_disable_hlo_passes=<> (turns off specific HLO passes; can be used for debugging)

## Previously used XLA Flags

The following flags were used previously used but no longer required.
- --xla_gpu_enable_async_reduce_scatter, --xla_gpu_enable_async_all_reduce, --xla_gpu_enable_async_all_gather ; Turned on by default, no longer needed
- --xla_gpu_enable_highest_priority_async_stream ; Turned on by default
- --xla_gpu_enable_triton_softmax_fusion ; Deprecated, no longer used

## Tips for Good LLM Training Performance on Blackwell (B200)

### **Support for Attention Mask Type**
MaxText uses the `padding_causal` mask type for [cuDNN Flash Attention](https://github.com/AI-Hypercomputer/maxtext/blob/6ec3368af31fff6e6d735ac9d5fb77f91fc0f784/MaxText/layers/attentions.py#L411). However, this mask type is not yet supported on Blackwell systems through TransformerEngine. Using `padding_causal` will default to the `unfused_attention` backend, which may reduce performance. As a temporary workaround, you can use the `causal` mask type for attention to maintain performance.

### **No Need to Set `CUDA_DEVICE_MAX_CONNECTIONS=1`**
Hopper was requiring CUDA_DEVICE_MAX_CONNECTIONS=1 to achieve better communication-compute overlap. This isn't needed for Blackwell and is in fact slower. On Blackwell systems, kernels assigned to higher-priority streams can utilize SM (Streaming Multiprocessor) resources without waiting for lower-priority kernels to release them. Therefore, it is better to leave `CUDA_DEVICE_MAX_CONNECTIONS` at its default value.

### **Additional XLA Flags**
Enabling CUDA Graphs only for Fusions and Custom Calls reduces CPU launch latency overheads on B200, ensure that you set the following XLA flags: `--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL`

This configuration improves performance on Blackwell systems by leveraging efficient command buffer execution in all the models tested on B200.

### **Better Utilizing Additional Memory in Blackwell**
Blackwell (B200) GPUs have a memory capacity of 180GB, significantly more than H100 GPUs. To take full advantage of this additional memory and enhance performance:

- Adjust model parallelism configurations: can use less model parallelism to fit the same model in memory.
- Increase batch sizes where possible: larger batch sizes can improve GeMM kernel efficiency.
- Optimize activation checkpointing policies: fewer activation tensors need to be recomputed in the backward pass on B200.

Careful tuning of these parameters is essential when transitioning from H100 to B200 systems to fully utilize the available resources.

### **Debugging Hangs in 1-process-multiple-device set-up**
If using 1 process to manage multiple devices in a node, hangs can happen when a process-wide synchronizing CUDA API such as cudaFree is called at the same time a collective is running across multiple devices within the same process.
The following remedy steps can be taken to workaround such cases:

- Set [NCCL_LAUNCH_MODE=GROUP](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-launch-mode) in the environment.
- Change the JAX program to bind 1 process to a single device instead of managing multiple devices.
