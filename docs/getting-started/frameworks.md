---
title: Frameworks & Supported Models
subtitle: JAX frameworks and model architectures we support and test
slug: frameworks
---

We support and test the following JAX frameworks and model architectures. More
details about each model and available containers can be found in their
respective guides.

| Framework | Models | Use cases | Container |
| :--- | :---: | :---: | :---: |
| [MaxText](../frameworks/maxtext/README.md) | DeepSeek V3.2, Gemma 4, Llama 4, Qwen3, Qwen 3.5, Mixtral, Kimi K2.6 | pre-training | `ghcr.io/nvidia/jax:maxtext` |
| [AXLearn](../frameworks/axlearn/README.md) | Fuji | pre-training | `ghcr.io/nvidia/jax:axlearn` |
| [Equinox](https://github.com/patrick-kidger/equinox) |  | pre-training | `ghcr.io/nvidia/jax:equinox` |
| [TorchAX](https://github.com/google/torchax) |  | running PyTorch models on JAX | `ghcr.io/nvidia/jax:torchax` |
| [AlphaFold3](https://github.com/google-deepmind/alphafold3) | EvoFormer | inference | `ghcr.io/nvidia/jax:alphafold` |

## Framework guides

- [MaxText](../frameworks/maxtext/README.md) — high-performance, scalable LLM
  framework by Google, including an [NVFP4 training example](../frameworks/maxtext/nvfp4/README.md).
- [AXLearn](../frameworks/axlearn/README.md) — Apple's deep-learning design framework built on
  JAX and XLA.

For the full list of published containers and their build/test status, see
[Build Pipeline Status](./build-status.md).
