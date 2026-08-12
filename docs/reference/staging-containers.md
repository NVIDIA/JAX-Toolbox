---
title: Staging Containers
subtitle: Historical NVIDIA-authored XLA and JAX staging releases
slug: staging-containers
---

JAX-Toolbox staging containers hosted pending NVIDIA-authored XLA and JAX
enhancements for NVIDIA GPU — PRs awaiting upstream review and merge in the OSS
OpenXLA and JAX repositories — published as `jax-scale-training` tagged
containers.

**The staging-container program has concluded**: the `scale-training.yaml`
pipeline was removed from CI in June 2026 and no new `jax-scale-training`
containers are being published. The dated images below remain available from
`ghcr.io/nvidia/jax`.

## Staging releases

| Release date | Container | XLA branch (includes pending PRs) |
| --- | --- | --- |
| 2026-05-30 | [jax-scale-training-2026-05-30](https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax/906936355?tag=jax-scale-training-2026-05-30) | [00de6b34](https://github.com/openxla/xla/blob/00de6b342f7f02f1b306121dab98706ae1d180b4) |
| 2026-05-11 | [jax-scale-training-2026-05-11](https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax/856082965?tag=jax-scale-training-2026-05-11) | [552b0a3e](https://github.com/openxla/xla/blob/552b0a3ef06453c74de9f62f778a0f3a960d7e6d/STAGING.md) |
| 2026-04-24 | [jax-scale-training-2026-04-24](https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax/820220988?tag=jax-scale-training-2026-04-24) | [5dfe2147](https://github.com/openxla/xla/blob/5dfe2147cbdd54b2fa1d76da817c64a1847373ca/STAGING.md) |
| 2026-04-18 | [jax-scale-training-2026-04-18](https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax/805313885?tag=jax-scale-training-2026-04-18) | [8147118](https://github.com/sfvaroglu/xla/tree/8147118a7b9707d26dcb747767a9c0dd9081325f) |

The underlying CUDA container used for building `-scale-training` containers can
be seen from the corresponding dated version in [Container Versions](./container-versions.md).

To check the versions of libraries in the container, run:

```bash
docker run --rm --gpus all ghcr.io/nvidia/jax:jax-scale-training-2026-04-24 -c '
echo "=== CUDA Toolkit ===" && echo ${CUDA_VERSION}
echo "=== cuDNN ===" && echo ${CUDNN_VERSION}
echo "=== NCCL ===" &&  echo ${NCCL_VERSION}
echo "=== Python Packages ===" && pip list | grep -iE "jax|flax|equinox|optax|chex|orbax|numpy|scipy|nvidia-"
'
```
