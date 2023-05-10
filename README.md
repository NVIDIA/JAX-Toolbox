# JAX Toolbox

| Image                                                | Build                                      | Test                                   |
| ---------------------------------------------------- | ------------------------------------------ | -------------------------------------- |
| [![container-badge-base]][container-link-base]       | [![build-badge-base]][workflow-base]       |                                        |
| [![container-badge-jax]][container-link-jax]         | [![build-badge-jax]][workflow-jax]         | [![test-badge-jax]][workflow-jax-unit] |
| [![container-badge-t5x]][container-link-t5x]         | [![build-badge-t5x]][workflow-t5x]         | [![test-badge-t5x]][workflow-t5x-perf] |
| [![container-badge-pax]][container-link-pax]         | [![build-badge-pax]][workflow-pax]         |                                        |
| [![container-badge-te]][container-link-te]           | [![build-badge-te]][workflow-te]           | [![unit-test-badge-te]][workflow-te-test] |
| [![container-badge-rosetta]][container-link-rosetta] | [![build-badge-rosetta]][workflow-rosetta] | [![test-badge-rosetta]][workflow-rosetta-test] (dummy) |

[container-badge-base]: https://img.shields.io/static/v1?label=&message=.base&color=gray&logo=docker
[container-badge-jax]: https://img.shields.io/static/v1?label=&message=JAX&color=gray&logo=docker
[container-badge-t5x]: https://img.shields.io/static/v1?label=&message=T5X&color=gray&logo=docker
[container-badge-pax]: https://img.shields.io/static/v1?label=&message=PAX&color=gray&logo=docker
[container-badge-rosetta]: https://img.shields.io/static/v1?label=&message=ROSETTA&color=gray&logo=docker
[container-badge-te]: https://img.shields.io/static/v1?label=&message=TE&color=gray&logo=docker

[container-link-base]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax-toolbox
[container-link-jax]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax
[container-link-t5x]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/t5x
[container-link-pax]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/pax
[container-link-te]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax-te
[container-link-rosetta]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/rosetta

[build-badge-base]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/weekly-base-build.yaml?branch=main&label=weekly&logo=github-actions&logoColor=dddddd
[build-badge-jax]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/nightly-jax-build.yaml?branch=main&label=nightly&logo=github-actions&logoColor=dddddd
[build-badge-t5x]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/nightly-t5x-build.yaml?branch=main&label=nightly&logo=github-actions&logoColor=dddddd
[build-badge-pax]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/nightly-pax-build.yaml?branch=main&label=nightly&logo=github-actions&logoColor=dddddd
[build-badge-rosetta]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/nightly-rosetta-build.yaml?branch=main&label=nightly&logo=github-actions&logoColor=dddddd
[build-badge-te]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/nightly-te-build.yaml?branch=main&label=nightly&logo=github-actions&logoColor=dddddd

[workflow-base]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/weekly-base-build.yaml
[workflow-jax]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-jax-build.yaml
[workflow-t5x]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-t5x-build.yaml
[workflow-pax]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-pax-build.yaml
[workflow-rosetta]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-rosetta-build.yaml
[workflow-te]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-te-build.yaml

[test-badge-jax]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fjax-unit-test-status.json&logo=nvidia
[test-badge-t5x]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/nightly-t5x-test-mgmn.yaml?branch=main&label=A100%20MGMN&logo=nvidia
[unit-test-badge-te]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fte-unit-test-status.json&logo=nvidia
[test-badge-rosetta]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/nightly-rosetta-test.yaml?branch=main&label=A100%20MGMN&logo=nvidia

[workflow-jax-unit]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-jax-test-unit.yaml
[workflow-t5x-perf]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-t5x-test-mgmn.yaml
[workflow-te-test]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-te-test.yaml
[workflow-rosetta-test]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-rosetta-test.yaml


## Note
This repo currently hosts a public CI for JAX on NVIDIA GPUs and covers some JAX libraries like: [T5x](https://github.com/google-research/t5x), [PAXML](https://github.com/google/paxml), [Transformer Engine](https://github.com/NVIDIA/TransformerEngine), and others to come soon.
