# JAX Toolbox

<table>
    <thead>
        <tr>
            <th>Image</th>
            <th>Build</th>
            <th>Tests</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>

[![container-badge-base]][container-link-base]
            </td>
            <td>
[![build-badge-base]][workflow-base]
            </td>
            <td> n/a </td>
        </tr>
        <tr style="border-bottom-style:hidden">
            <td colspan=3> Frameworks </td>
        </tr>
        <!-- JAX -->
        <tr>
            <td>
[![container-badge-jax]][container-link-jax]
            </td>
            <td>
[![build-badge-jax]][workflow-jax]
            </td>
            <td>
[![test-badge-jax-V100]][workflow-jax-unit]
<br>
[![test-badge-jax-A100]][workflow-jax-unit]
            </td>
        </tr>
        <!-- te -->
        <tr>
            <td>
[![container-badge-te]][container-link-te]
            </td>
            <td>
Included in JAX build
            </td>
            <td>
[![unit-test-badge-te]][workflow-te-test] <br> [![integration-test-badge-te]][workflow-te-test]
            </td>
        </tr>
        <!-- rosetta-t5x -->
        <tr>
            <td rowspan=3>
[![container-badge-rosetta-t5x]][container-link-rosetta-t5x]
            </td>
            <td rowspan=3>
[![build-badge-rosetta-t5x]][workflow-rosetta-t5x]
            </td>
        </tr>
        <tr>
            <td>
[![test-badge-t5x]][workflow-t5x-perf]
            </td>
        </tr>
        <tr>
            <td>
[![test-badge-rosetta-t5x]][workflow-rosetta-t5x]
            </td>
        </tr>
        <!-- rosetta pax -->
        <tr>
            <td rowspan=3>
[![container-badge-rosetta-pax]][container-link-rosetta-pax]
            </td>
            <td rowspan=3>
[![build-badge-rosetta-pax]][workflow-rosetta-pax]
            </td>
        </tr>
        <tr>
            <td>
[![test-badge-pax]][workflow-pax-perf]
            </td>
        </tr>
        <tr>
            <td>
[![test-badge-rosetta-pax]][workflow-rosetta-pax]
            </td>
        </tr>
        <!-- Pallas -->
        <tr>
            <td>
[![container-badge-pallas]][container-link-pallas]
            </td>
            <td>
[![build-badge-pallas]][workflow-pallas]
            </td>
            <td>
[![test-badge-pallas-V100]][workflow-pallas-unit]
<br>
[![test-badge-pallas-A100]][workflow-pallas-unit]
            </td>
        </tr>
    </tbody>
</table>


[container-badge-base]: https://img.shields.io/static/v1?label=&message=.base&color=gray&logo=docker
[container-badge-jax]: https://img.shields.io/static/v1?label=&message=JAX&color=gray&logo=docker
[container-badge-te]: https://img.shields.io/static/v1?label=&message=TE&color=gray&logo=docker
[container-badge-rosetta-t5x]: https://img.shields.io/static/v1?label=&message=T5X&color=gray&logo=docker
[container-badge-rosetta-pax]: https://img.shields.io/static/v1?label=&message=PAX&color=gray&logo=docker
[container-badge-pallas]: https://img.shields.io/static/v1?label=&message=Pallas&color=gray&logo=docker

[container-link-base]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax-toolbox
[container-link-jax]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax
[container-link-te]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax-te
[container-link-rosetta-t5x]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/t5x
[container-link-rosetta-pax]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/pax
[container-link-pallas]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax

[build-badge-base]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/nightly-base-build.yaml?branch=main&label=nightly&logo=github-actions&logoColor=dddddd
[build-badge-jax]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/nightly-jax-build.yaml?branch=main&label=nightly&logo=github-actions&logoColor=dddddd
[build-badge-rosetta-t5x]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Frosetta-t5x-build-status.json&logo=github-actions&logoColor=dddddd
[build-badge-rosetta-pax]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Frosetta-pax-build-status.json&logo=github-actions&logoColor=dddddd
[build-badge-pallas]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/nightly-pallas-build.yaml?branch=main&label=nightly&logo=github-actions&logoColor=dddddd

[workflow-base]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-base-build.yaml
[workflow-jax]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-jax-build.yaml
[workflow-rosetta-t5x]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-rosetta-t5x-build-test.yaml
[workflow-rosetta-pax]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-rosetta-pax-build.yaml
[workflow-pallas]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-pallas-build.yaml

[test-badge-jax-V100]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-jax-unit-test-V100.json&logo=nvidia
[test-badge-jax-A100]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-jax-unit-test-A100.json&logo=nvidia
[test-badge-t5x]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-t5x-mgmn-test.json&logo=nvidia
[test-badge-pax]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-pax-mgmn-test.json&logo=nvidia
[unit-test-badge-te]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fte-unit-test-status.json&logo=nvidia
[integration-test-badge-te]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fte-integration-test-status.json&logo=nvidia
[test-badge-rosetta-t5x]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Frosetta-t5x-overall-test-status.json&logo=nvidia
[test-badge-rosetta-pax]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Frosetta-pax-overall-test-status.json&logo=nvidia
[test-badge-pallas-V100]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-pallas-unit-test-V100.json&logo=nvidia
[test-badge-pallas-A100]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-pallas-unit-test-A100.json&logo=nvidia

[workflow-jax-unit]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-jax-test-unit.yaml
[workflow-te-test]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-te-test.yaml
[workflow-t5x-perf]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-t5x-test-mgmn.yaml
[workflow-pax-perf]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-pax-test-mgmn.yaml
[workflow-pallas-unit]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-pallas-test-unit.yaml


## Note
This repo currently hosts a public CI for JAX on NVIDIA GPUs and covers some JAX libraries like: [T5x](https://github.com/google-research/t5x), [PAXML](https://github.com/google/paxml), [Transformer Engine](https://github.com/NVIDIA/TransformerEngine), [Pallas](https://jax.readthedocs.io/en/latest/pallas/quickstart.html) and others to come soon.

## Supported Models
We currently enable training and evaluation for the following models:
| Model Name | Pretraining | Fine-tuning | Evaluation |
| :--- | :---: | :---: | :---: |
| [GPT-3(paxml)](./rosetta/rosetta/projects/pax) | ✔️ |   | ✔️ |
| [t5(t5x)](./rosetta/rosetta/projects/t5x) | ✔️ | ✔️ | ✔️ |
| [ViT](./rosetta/rosetta/projects/vit) | ✔️ | ✔️ | ✔️ |
| [Imagen](./rosetta/rosetta/projects/imagen) | ✔️ |   | ✔️ |

We will update this table as new models become available, so stay tuned.

## Environment Variables

The [JAX image](https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax) is embedded with the following flags and environment variables for performance tuning:

| XLA Flags | Value | Explanation |
| --------- | ----- | ----------- |
| `--xla_gpu_enable_latency_hiding_scheduler` | `true`  | allows XLA to move communication collectives to increase overlap with compute kernels |
| `--xla_gpu_enable_async_all_gather` | `true` | allows XLA to run NCCL [AllGather](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html#allgather) kernels on a separate CUDA stream to allow overlap with compute kernels |
| `--xla_gpu_enable_async_reduce_scatter` | `true` | allows XLA to run NCCL [ReduceScatter](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html#reducescatter) kernels on a separate CUDA stream to allow overlap with compute kernels |
| `--xla_gpu_enable_triton_gemm` | `false` | use cuBLAS instead of Trition GeMM kernels |

| Environment Variable | Value | Explanation |
| -------------------- | ----- | ----------- |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | use a single queue for GPU work to lower latency of stream operations; OK since XLA already orders launches |
| `NCCL_IB_SL` | `1` | defines the InfiniBand Service Level ([1](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-sl)) |
| `NCCL_NVLS_ENABLE` | `0` | Disables NVLink SHARP ([1](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nvls-enable)). Future releases will re-enable this feature. |
| `CUDA_MODULE_LOADING` | `EAGER` | Disables lazy-loading ([1](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-environment-variables)) which uses slightly more GPU memory. |

## FAQ (Frequently Asked Questions)

<details>
    <summary>`bus error` when running JAX in a docker container</summary>

**Solution:**
```bash
docker run -it --shm-size=1g ...
```

**Explanation:**
The `bus error` might occur due to the size limitation of `/dev/shm`. You can address this by increasing the shared memory size using
the `--shm-size` option when launching your container.
</details>

<details>

<summary>enroot/pyxis reports error code 404 when importing multi-arch images</summary>

**Problem description:**
```
slurmstepd: error: pyxis:     [INFO] Authentication succeeded
slurmstepd: error: pyxis:     [INFO] Fetching image manifest list
slurmstepd: error: pyxis:     [INFO] Fetching image manifest
slurmstepd: error: pyxis:     [ERROR] URL https://ghcr.io/v2/nvidia/jax/manifests/<TAG> returned error code: 404 Not Found
```

**Solution:**
Upgrade [enroot](https://github.com/NVIDIA/enroot) or [apply a single-file patch](https://github.com/NVIDIA/enroot/releases/tag/v3.4.0) as mentioned in the enroot v3.4.0 release note.

**Explanation:**
Docker has traditionally used Docker Schema V2.2 for multi-arch manifest lists but has switched to using the Open Container Initiative (OCI) format since 20.10. Enroot added support for OCI format in version 3.4.0.
</details>

## JAX on Public Clouds

* AWS
    * [Add EFA integration](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-efa.html)
    * [SageMaker code sample](https://github.com/aws-samples/aws-samples-for-ray/tree/main/sagemaker/jax_alpa_language_model)
* GCP
    * [Getting started with JAX multi-node applications with NVIDIA GPUs on Google Kubernetes Engine](https://cloud.google.com/blog/products/containers-kubernetes/machine-learning-with-jax-on-kubernetes-with-nvidia-gpus)
* Azure
    * [Accelerating AI applications using the JAX framework on Azure’s NDm A100 v4 Virtual Machines](https://techcommunity.microsoft.com/t5/azure-high-performance-computing/accelerating-ai-applications-using-the-jax-framework-on-azure-s/ba-p/3735314)
* OCI
    * [Running a deep learning workload with JAX on multinode multi-GPU clusters on OCI](https://blogs.oracle.com/cloud-infrastructure/post/running-multinode-jax-clusters-on-oci-gpu-cloud)

## Resources
* [What's New in JAX | GTC Spring 2023](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51956/)
* [Slurm and OpenMPI zero config integration](https://jax.readthedocs.io/en/latest/_autosummary/jax.distributed.initialize.html)
* [Adding custom GPU ops](https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html)
