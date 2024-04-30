export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=false --xla_gpu_enable_triton_gemm=false --xla_gpu_cuda_graph_level=0 --xla_gpu_enable_triton_softmax_fusion=false ${XLA_FLAGS}"
