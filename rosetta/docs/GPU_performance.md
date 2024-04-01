This page documents the various flags in XLA and JAX to improve performance for LLMs on GPUs. The XLA flags are defined with their default values in [xla/debug_options_flags.cc](https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc)


## Flags to manage memory used in XLA

- XLA_PYTHON_CLIENT_MEM_FRACTION is a JAX flag that allocates a fraction of GPU memory for JAX/XLA.
--  Ideally, should be 1, but in practice less because some memory is used by NVIDIA Libraries, and the JAX framework.
--  We typically set it to 0.9 or 0.8. At 0.9, XLA gets 90% of GPU memory.

- xla_gpu_memory_limit_slop_factor controls the memory used by XLA for determining its default heuristics for scheduling, and rematerialization. Default is recommended.


## Enable Latency Hiding Scheduler, and Asynchronous collective communication

To achieve communication computation overlap for models in JAX/XLA, we must enable Latency Hiding Scheduler and enable asynchronous communications. 

To enable latency hiding optimizations with XLA, turn on the following flag: 

- --xla_gpu_enable_latency_hiding_scheduler=true 

To enable asynchronous communication for all collectives, the following is recommended, and is set by default in XLA :

- --xla_gpu_enable_async_collectives=true
- --xla_gpu_enable_highest_priority_async_stream=true

For more fine-grained control over which collectives should be asynchronous or not, please use: 

- --xla_gpu_enable_async_all_reduce=<>
- --xla_gpu_enable_async_all_gather=<>
- --xla_gpu_enable_async_reduce_scatter=<> 
- --xla_gpu_enable_async_collective_permute=<>


### Enable Optimizations for FSDP communication 

With FSDP in JAX/XLA, there are additional optimizations of 

- scan loop unrolling and loop double buffering 
    - --xla_gpu_enable_while_loop_double_buffering=true
- optimized pipelining of all-gather and reduce-scatter for latency hiding in FSDP
    - --xla_gpu_enable_pipelined_all_gather=true
    - --xla_gpu_enable_pipelined_reduce_scatter=true
    - --xla_gpu_enable_pipelined_all_reduce=true 
    - --xla_gpu_enable_pipelined_collectives=false // overrides the above
- combining tensors that are sharded along different dimensions. Within a transformer layer, tensors can be sharded row-wise or column-wise and by default XLA will generate multiple collective calls for tensors sharded along different dimensions. The following optimization flags combine all tensors shardings, and map them to a group NCCL call that has a large commulative size and achieves high communication efficiency. 
    - --xla_gpu_enable_all_gather_combine_by_dim=false
    - --xla_gpu_enable_reduce_scatter_combine_by_dim=false
- Combine threshold values in XLA that determine when an all-gather (AG) or reduce-scatter (RS) is triggered. We want to set these values to be atleast as large as the size of weights (AG) or gradients (RS). For example, LLAMA2-7B with BF16 weights and gradients, we have 32 transformer layers => each layer has ~218M weights => one would want to set these thresholds to atleast 436MB. 
    - --xla_gpu_all_gather_combine_threshold_bytes=8589934592
    - --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
- Combine threshold values in XLA that determine when an all-reduce (AR) is triggered. Typically, used to overlap AR of gradients with back-prop of compute. We want to set this to be atleast as large as possible to achieve high efficiency, but as small as possible to achieve maximum overlap. Depending on the interconnect of your system, one might want to try several threshold values in steps of 2 from say 16MB to total gradient size.
    - --xla_gpu_all_reduce_combine_threshold_bytes=8589934592


### Flags for Collective permute 

The following flags enable overlap of pipeline parallel communication of send/recv with computation. 
- --xla_gpu_enable_pipelined_p2p=true  (false by default)
- --xla_gpu_collective_permute_decomposer_threshold=33521664

### Flags to enable Collective Matmul

The following flags enable overlap of tensor parallel communication with GEMMs/matmul by splicing GEMMs into smaller chunks and triggering each chunks' collective right after the chunk's GEMM is done. The threshold determines the size of output buffer of GEMM when this optimization becomes active (0 enables collective matmul for all GEMM-collective patterns)
- --xla_gpu_multi_streamed_windowed_einsum=true
- --xla_gpu_threshold_for_windowed_einsum_mib=0

### Profile Guided Latency Estimator (PGLE)

The following flag enables use of PGLE with JAX/XLA. Please see [PGLE notes](https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/PGLE.md) for more details.
- --xla_gpu_pgle_profile_file_or_directory_path=filename

## Other Flags 

### CUDA graph via Command Buffers

The below enables CUDA Graph suppport for JAX/XLA workloads
- --xla_gpu_enable_command_buffer (?? by default)

### Dynamic-Update Slice Fusion
- --xla_gpu_enable_custom_fusions=true

### Other flags ????
- --xla_gpu_enable_triton_softmax_fusion=false 
- CUDA_DEVICE_MAX_CONNECTIONS=1
- --xla_gpu_cudnn_gemm_fusion

### NCCL Optimizations
- --xla_gpu_enable_nccl_user_buffers=true (enables user-buffers in NCCL, for zero-copy collectives)
- --xla_gpu_enable_nccl_comm_splitting 
- --xla_gpu_enable_nccl_per_stream_comms
- NCCL_LL128_BUFFSIZE: -2
- NCCL_LL_BUFFSIZE: -2
- NCCL_PROTO: SIMPLE,LL,LL128

### Debug flags 
- --xla_dump_to=some/path
- --xla_dump_latency_hiding_schedule=true
- --xla_gpu_lhs_enable_gpu_async_tracker=false




