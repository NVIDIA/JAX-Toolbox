This page documents the various flags in XLA and JAX to improve performance for LLMs on GPUs. The XLA flags are defined with their default values in [xla/debug_options_flags.cc](https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc)


### Flags to manage memory used in XLA

- XLA_PYTHON_CLIENT_MEM_FRACTION is a JAX flag that allocates a fraction of GPU memory for JAX/XLA.
--  Ideally, should be 1, but in practice less because some memory is used by NVIDIA Libraries, and the JAX framework.
--  We typically set it to 0.9 or 0.8. At 0.9, XLA gets 90% of GPU memory.

- xla_gpu_memory_limit_slop_factor controls the memory used by XLA for determining its default heuristics for scheduling, and rematerialization.


### Enable Asynchronous collective communication

To enable asynchronous communication for all collectives, the following is recommended:

- --xla_gpu_enable_async_collectives=true
- --xla_gpu_enable_highest_priority_async_stream=true

For more fine-grained control over which collectives should be asynchronous, please use: 

- --xla_gpu_enable_async_all_reduce=<>
- --xla_gpu_enable_async_all_gather=<>
- --xla_gpu_enable_async_reduce_scatter=<> 
- --xla_gpu_enable_async_collective_permute=<>


### Enable Optimizations for FSDP communication 

- --xla_gpu_all_reduce_combine_threshold_bytes=51200	
- --xla_gpu_all_reduce_combine_threshold_bytes=8589934592
- --xla_gpu_all_gather_combine_threshold_bytes=8589934592
- --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
- --xla_gpu_enable_all_gather_combine_by_dim=false
- --xla_gpu_enable_reduce_scatter_combine_by_dim=false

### Flags for Collective permute 
- --xla_gpu_enable_pipelined_p2p=false
- --xla_gpu_collective_permute_decomposer_threshold=33521664

### CUDA graph via Command Buffers
- --xla_gpu_enable_command_buffer

### Nccl User Buffers and Optimizations
- --xla_gpu_enable_nccl_user_buffers
- --xla_gpu_enable_nccl_comm_splitting
- --xla_gpu_enable_nccl_per_stream_comms

### Dynamic-Update Slice Fusion
- --xla_gpu_enable_custom_fusions=true

### Profile Guided Latency Estimator
- --xla_gpu_pgle_profile_file_or_directory_path

### Pipelined comm/compute for FSDP flags 
- --xla_gpu_enable_pipelined_all_gather=true
- --xla_gpu_enable_pipelined_reduce_scatter=true
- --xla_gpu_enable_pipelined_all_reduce=true 
- --xla_gpu_enable_pipelined_collectives=false // overrides the above
- --xla_gpu_enable_while_loop_double_buffering=true

### Flags to enable Collective Matmul
- --xla_gpu_multi_streamed_windowed_einsum=true
- --xla_gpu_threshold_for_windowed_einsum_mib=0

### Other flags
- --xla_gpu_enable_xla_runtime_executable=true
- --xla_gpu_enable_latency_hiding_scheduler=true 
- --xla_gpu_enable_triton_gemm=false 
- --xla_gpu_enable_triton_softmax_fusion=false 
- CUDA_DEVICE_MAX_CONNECTIONS=1
- --xla_gpu_cudnn_gemm_fusion

### Environemnt variables for best performance with NCCL
- NCCL_LL128_BUFFSIZE: -2
- NCCL_LL_BUFFSIZE: -2
- NCCL_PROTO: SIMPLE,LL,LL128

### Debug flags 
- --xla_dump_to=some/path
- --xla_dump_latency_hiding_schedule=true
- --xla_gpu_lhs_enable_gpu_async_tracker=false




