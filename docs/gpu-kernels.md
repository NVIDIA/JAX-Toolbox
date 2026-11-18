---
title: Customize JAX with GPU kernels
subtitle: Writing custom kernels for JAX on GPUs.
slug: gpu-kernels
---

Developers need customization of the JAX stack to customize their models and to optimize beyond what's natively available in JAX/XLA stack.

NVIDIA offers a suite of tools, libraries and kernel DSLs, helping customization.

- [**Writing High-Performance CuTe DSL kernels in JAX**](https://docs.jax.dev/en/latest/notebooks/cute_dsl_jax.html)

If you work with an AI coding agent, see [Agent Skills](agent-skills/README.md) —
starting with [`jax-cudnn-frontend`](agent-skills/jax-cudnn-frontend/README.md),
a skill for integrating cuDNN Frontend / CuTe DSL kernels into pure JAX
workflows.
