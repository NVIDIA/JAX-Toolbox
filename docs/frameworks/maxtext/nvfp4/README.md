# NVFP4 Training Example with MaxText

This directory contains an example script for running MaxText training with NVFP4 (FP4) quantization on NVIDIA GPUs.

For details on the NVFP4 methodology, see the paper: [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149).

## Overview

`nvfp4_example.sh` demonstrates how to train a Llama3-8B model using Transformer Engine's NVFP4 quantization with MaxText. It uses synthetic data, making it easy to benchmark performance without requiring a dataset.

## Enabling NVFP4

### Quantization Mode

Set the `quantization` flag in MaxText to enable NVFP4:

- **`quantization=te_nvfp4_no_rht`** — NVFP4 without RHT. Lower overhead but may degrade convergence quality.
- **`quantization=te_nvfp4`** — NVFP4 with Random Hadamard Transform (RHT). Recommended if the convergence using te_nvfp4_no_rht is not satisfactory.

### What is RHT?

Random Hadamard Transform (RHT) applies orthogonal rotations to tensors before quantization, redistributing outlier values into an approximately Gaussian distribution. This makes the tensor values easier to represent in the narrow FP4 format. Specifically, a 16x16 Hadamard matrix with a random sign vector is applied to weight-gradient (Wgrad) inputs during training. Enabling RHT improves training stability and convergence at the cost of a small amount of extra computation.

### Rematerialization Policy

Use `remat_policy='minimal_with_context_and_quantization'` to prevent rematerialization of quantization ops. This avoids redundant re-execution of quantization kernels during the backward pass, improving training performance.

## Configuration

### Parallelism

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ici_data_parallelism` | 1 | Intra-chip data parallelism |
| `dcn_data_parallelism` | 1 | Data-center data parallelism |
| `ici_fsdp_parallelism` | 4 | Intra-chip FSDP parallelism |
| `dcn_fsdp_parallelism` | 1 | Data-center FSDP parallelism |

### Model / Training Settings

| Parameter | Value |
|-----------|-------|
| Model | `llama3-8b` |
| Quantization | `te_nvfp4_no_rht` |
| Attention | `cudnn_flash_te` |
| Dtype | `bfloat16` |
| Sequence length | 8192 |
| Per-device batch size | 4 |
| Steps | 50 |
| Dataset | Synthetic |
| Remat policy | `minimal_with_context_and_quantization` |
| Profiler | `nsys` |

## Usage

```bash
bash nvfp4_example.sh
```

The script should be run from the MaxText repository root inside a container that has JAX, Transformer Engine, and the required CUDA/cuDNN libraries installed. The Maxtext container from https://github.com/NVIDIA/JAX-Toolbox is recommended: ghcr.io/nvidia/jax:maxtext.

