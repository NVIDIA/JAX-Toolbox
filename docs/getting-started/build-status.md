---
title: Build Pipeline Status
subtitle: Live build and test status for the published JAX containers
slug: build-status
---

[![Workflow status](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-workflow-metadata.json&logo=github-actions&logoColor=white)](https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/ci.yaml?query=event%3Aschedule+branch%3Amain)

The table below lists the published containers and their Dockerfiles. Each
container is built nightly for `amd64` and `arm64`. For the full, always-current
matrix of per-test status badges (single-GPU, multi-GPU, distributed, etc.), see
the [build-pipeline table in the repository README](https://github.com/NVIDIA/JAX-Toolbox#build-pipeline-status)
and the [CI workflow runs](https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/ci.yaml?query=event%3Aschedule+branch%3Amain).

| Component | Container | Dockerfile |
| --- | --- | --- |
| base = {CUDA, cuDNN, NCCL, OFED, EFA} | `ghcr.io/nvidia/jax:base` | [Dockerfile.base](https://github.com/NVIDIA/JAX-Toolbox/blob/main/.github/container/Dockerfile.base) |
| core = {base, JAX, Flax, TE} | `ghcr.io/nvidia/jax:jax` | [Dockerfile.jax](https://github.com/NVIDIA/JAX-Toolbox/blob/main/.github/container/Dockerfile.jax) |
| Equinox = {core, Equinox} | `ghcr.io/nvidia/jax:equinox` | [Dockerfile.equinox](https://github.com/NVIDIA/JAX-Toolbox/blob/main/.github/container/Dockerfile.equinox) |
| MaxText = {core, MaxText} | `ghcr.io/nvidia/jax:maxtext` | [Dockerfile.maxtext](https://github.com/NVIDIA/JAX-Toolbox/blob/main/.github/container/Dockerfile.maxtext) |
| AXLearn = {core, AXLearn} | `ghcr.io/nvidia/jax:axlearn` | [Dockerfile.axlearn](https://github.com/NVIDIA/JAX-Toolbox/blob/main/.github/container/Dockerfile.axlearn) |
| AlphaFold = {core, AlphaFold} | `ghcr.io/nvidia/jax:alphafold` | [Dockerfile.alphafold](https://github.com/NVIDIA/JAX-Toolbox/blob/main/.github/container/Dockerfile.alphafold) |

In all cases, `ghcr.io/nvidia/jax:XXX` points to the latest nightly build of the
container for `XXX`. For a stable reference, use `ghcr.io/nvidia/jax:XXX-YYYY-MM-DD`.

In addition to the public CI, we also run internal CI nightlies on GB300, B300,
GB200, B200, DGX Spark, RTX PRO 6000 Blackwell, Jetson AGX Thor, H100 SXM 80GB,
and A100 SXM 80GB.
