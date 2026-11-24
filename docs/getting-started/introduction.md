---
title: Introduction
subtitle: A public CI, Docker images, and optimized examples for JAX on NVIDIA GPUs
slug: introduction
---

JAX Toolbox provides a public CI, Docker images for popular JAX libraries, and
optimized JAX examples to simplify and enhance your JAX development experience on
NVIDIA GPUs. It supports JAX libraries such as
[MaxText](https://github.com/google/maxtext) and
[Pallas](https://jax.readthedocs.io/en/latest/pallas/quickstart.html).

## Where to go next

- [Frameworks & Supported Models](./frameworks.md) — the frameworks and
  model architectures we support and test, and their containers.
- [Build Pipeline Status](./build-status.md) — live build and test
  status for every published container.
- [Environment Variables](./environment-variables.md) — the XLA and
  NCCL flags baked into the images for performance tuning.
- [Profiling](../profiling.md) — how to profile JAX programs on GPU.
- [Container Versions](../reference/container-versions.md) — base-container history and
  staging releases.

## Containers

In all cases, `ghcr.io/nvidia/jax:XXX` points to the latest nightly build of the
container for `XXX`. For a stable reference, use `ghcr.io/nvidia/jax:XXX-YYYY-MM-DD`.

In addition to the public CI, we also run internal CI nightlies on GB300, B300,
GB200, B200, DGX Spark, RTX PRO 6000 Blackwell, Jetson AGX Thor, H100 SXM 80GB,
and A100 SXM 80GB.

## License

JAX Toolbox is dual-licensed. Source code is released under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), and
documentation, skills and other assets are released under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license. See [LICENSE.md](https://github.com/NVIDIA/JAX-Toolbox/blob/main/LICENSE.md)
for the full text of both licenses.
