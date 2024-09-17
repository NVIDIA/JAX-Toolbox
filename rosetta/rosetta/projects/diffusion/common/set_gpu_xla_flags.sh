# These XLA flags are meant to be used with the JAX version in the imagen container
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=false --xla_gpu_enable_async_all_gather=false --xla_gpu_enable_async_reduce_scatter=false --xla_gpu_enable_triton_gemm=false --xla_gpu_cuda_graph_level=0 --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_enable_async_all_reduce=false ${XLA_FLAGS}"
