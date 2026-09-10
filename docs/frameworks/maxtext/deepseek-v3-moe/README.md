# DeepSeek-v3 MoE Configuration with MaxText

Reference configuration for training DeepSeek-v3 671B with MaxText on GPUs, using the TransformerEngine MoEBlock with MXFP8 grouped GEMMs.

This configuration uses FSDP combined with expert parallelism across 128 GPUs. Tensor parallelism is not used. Individual parameters are documented in the [MaxText repository](https://github.com/AI-Hypercomputer/maxtext).

## Parallelism

The mesh is FSDP 16 (ICI 8 × DCN 2) × expert parallelism 8, for 128 GPUs in total.

| Axis | ICI | DCN |
|------|-----|-----|
| Data | 1 | 1 |
| FSDP | 8 | 2 |
| Tensor | 1 | 1 |
| Expert | 8 | 1 |

## MaxText configuration

```yaml
# Model parameters
model_name: "deepseek3-671b"
max_target_length: 4096
hardware: "gpu_multiprocess"

# Training settings
per_device_batch_size: 6
gradient_accumulation_steps: 1
steps: 15
attention: "cudnn_flash_te"
remat_policy: "custom"

# Transformer Engine MoEBlock with MXFP8 grouped GEMMs
quantization: "te_fp8_currentscaling"
te_moe_block: true
te_gmm_quantization: "te_mxfp8"
ragged_buffer_factor: 2.0
te_ep_overflow_check_every_n_steps: 20
prefuse_moe_weights: true

weight_dtype: "bfloat16"
mu_dtype: "bfloat16"

# Features
pgle: true
profiler: "xplane"
scan_layers: true
zero_one: false
shardy: true
use_segment: false

skip_first_n_steps_for_profiler: 4

custom_remat_enabled: true
logits_dot_in_fp32: false
use_iota_embed: false

custom_remat_config:
  mlpwi: device
  mlpwi_0: device
  mlpwi_1: device
  mlpwo: device
  moe_mlpwi_0: offload #remat
  moe_mlpwi_1: offload #remat
  moe_mlpwo: device
  query_proj: remat #offload
  key_proj: remat
  value_proj: remat #offload
  query_wa_proj: device
  kv_wa_proj: device
  out_proj: device
  context: device

# MoE routing parameters
n_routing_groups: -1
topk_routing_group: -1
capacity_factor: 1.0
megablox: false

# 128 GPUs: total FSDP=16 (ICI 8 × DCN 2) × EP=8.
nodes: 32
ici_data_parallelism: 1
ici_fsdp_parallelism: 8
ici_tensor_parallelism: 1
ici_expert_parallelism: 8
dcn_data_parallelism: 1
dcn_fsdp_parallelism: 2
dcn_tensor_parallelism: 1
dcn_expert_parallelism: 1
shard_optimizer_over_data: false

shard_exp_on_fsdp: false
```

Note that `quantization` and `te_gmm_quantization` are set independently: the dense GEMMs use FP8 with current scaling, while the MoE grouped GEMMs use MXFP8 block scaling.

## XLA flags

See [Tips for High-Performance LLMs with JAX and XLA](../../../GPU_performance.md) for what these flags do and how to tune them.

```bash
xla_gpu_all_reduce_combine_threshold_bytes: 33554432
xla_gpu_all_gather_combine_threshold_bytes: 6442450944
xla_gpu_reduce_scatter_combine_threshold_bytes: 201326592
xla_gpu_experimental_enable_nccl_symmetric_buffers: false
xla_gpu_enable_command_buffer: "'FUSION,CUBLAS,CUDNN,DYNAMIC_SLICE_FUSION'"
xla_gpu_experimental_max_unroll_factor: 8
xla_gpu_memory_limit_slop_factor: 99
```

## Environment variables

See [Environment Variables](../../../getting-started/environment-variables.md) for the full set of JAX and XLA variables available.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION: 0.88
CUDA_DEVICE_MAX_CONNECTIONS: 16
XLA_PJRT_GPU_HOST_MEMORY_PREALLOCATE: false
XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB: 180
```
