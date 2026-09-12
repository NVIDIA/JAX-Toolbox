---
title: Staging Containers
subtitle: Pending NVIDIA-authored XLA and JAX enhancements
slug: staging-containers
---

JAX-Toolbox staging containers host pending NVIDIA-authored XLA enhancements for
NVIDIA GPUs. These are pending PRs that are awaiting upstream review and merge in
the OSS OpenXLA repository. These are exposed as tags in this repository:
https://github.com/NVIDIA/xla_staging/tags and containers based on those commits
are published from JAX-Toolbox on the GitHub Container registry each Saturday.

## Staging releases

Older containers were named `-scale-training`, the newer ones are named `-staging`:
```
ghcr.io/nvidia/jax:jax-staging[-YYYY-MM-DD]
ghcr.io/nvidia/jax:maxtext-staging[-YYYY-MM-DD]
```

The underlying CUDA container used for building `-scale-training` containers can
be seen from the corresponding dated version in [Container Versions](./container-versions.md).

To check the versions of libraries in the container, run:

```bash
docker run --rm --gpus all ghcr.io/nvidia/jax:jax-staging -c '
echo "=== CUDA Toolkit ===" && echo ${CUDA_VERSION}
echo "=== cuDNN ===" && echo ${CUDNN_VERSION}
echo "=== NCCL ===" &&  echo ${NCCL_VERSION}
echo "=== Python Packages ===" && pip list | grep -iE "jax|flax|equinox|optax|chex|orbax|numpy|scipy|nvidia-"
'
```
