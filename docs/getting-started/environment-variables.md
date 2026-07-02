---
title: Environment Variables
subtitle: XLA and NCCL flags baked into the JAX images for performance tuning
slug: environment-variables
---

The [JAX image](https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax) is
embedded with the following flags and environment variables for performance
tuning of XLA and NCCL:

| XLA Flags | Value | Explanation |
| --------- | ----- | ----------- |
| `--xla_gpu_enable_latency_hiding_scheduler` | `true` | allows XLA to move communication collectives to increase overlap with compute kernels |

There are various other XLA flags users can set to improve performance. XLA flags
can also be tuned per workload based on specific performance needs.

For a detailed description of the XLA flags that can be set to optimize
performance, see [GPU Performance](../GPU_performance.md).
