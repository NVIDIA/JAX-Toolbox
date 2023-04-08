# JAX Toolbox

| App                               | CUDA container              | unit tests  | parallel tests |
| --------------------------------- | --------------------------- | ----------- | -------------- |
| Base                              | [![base-build]][base-image] |             |                |
| JAX                               | [![jax-build]][jax-image]   | ![jax-test] |                |
| T5X                               | ![t5x-build]                |             |                |
| Paxml                             |                             |             |                |

[base-build]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/weekly-base.yaml?label=weekly&logo=docker
[jax-build]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/nightly-jax.yaml?label=nightly&logo=docker
[t5x-build]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/nightly-t5x.yaml?label=nightly&logo=docker

[jax-image]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax
[base-image]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax-toolbox

[jax-test]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/jax-test.yaml?label=V100&logo=nvidia

## Note
This is not an officially supported NVIDIA product
