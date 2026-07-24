---
title: JAX Toolbox
description: Public CI, Docker images, and optimized examples for JAX development on NVIDIA GPUs.
---

# JAX Toolbox

JAX Toolbox provides a public CI pipeline, Docker images for popular JAX libraries, and optimized JAX examples to simplify and enhance your JAX development experience on NVIDIA GPUs.

## Supported Frameworks

| Framework | Models | Use cases | Container |
| :--- | :--- | :--- | :--- |
| [MaxText](https://github.com/google/maxtext) | DeepSeek V3.2, Gemma 4, Llama 4, Qwen3, Qwen 3.5, Mixtral, Kimi K2.6 | Pre-training | `ghcr.io/nvidia/jax:maxtext` |
| [AXLearn](frameworks/axlearn) | Fuji | Pre-training | `ghcr.io/nvidia/jax:axlearn` |
| [Equinox](https://github.com/patrick-kidger/equinox) |  | Pre-training | `ghcr.io/nvidia/jax:equinox` |
| [TorchAX](https://github.com/google/torchax) |  | Running PyTorch models on JAX | `ghcr.io/nvidia/jax:torchax` |
| [AlphaFold3](https://github.com/google-deepmind/alphafold3) | EvoFormer | Inference | `ghcr.io/nvidia/jax:alphafold` |

## What's Included

- **Performance guides** — XLA and JAX flags for high-performance LLMs, FP8 quantization, and profile-guided latency estimation (PGLE)
- **Profiling tools** — `nsys-jax`, a wrapper around Nsight Systems for collecting and analyzing JAX program profiles
- **Framework integration** — Pre-configured containers and example configs for MaxText and AXLearn
- **Resiliency** — Ray-based fault-tolerant JAX training patterns
- **Triage tool** — Automates regression attribution to specific commits or container versions

## Getting Started

Pull a container for your target framework and run:

```bash
docker pull ghcr.io/nvidia/jax:maxtext
```

See the [MaxText](frameworks/maxtext) or [AXLearn](frameworks/axlearn) pages for hardware requirements, example configs, and performance numbers.

## Source

[github.com/NVIDIA/JAX-Toolbox](https://github.com/NVIDIA/JAX-Toolbox)
