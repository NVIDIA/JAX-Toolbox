# **JAX Toolbox**

[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](https://github.com/NVIDIA/JAX-Toolbox/blob/main/LICENSE.md)
[![Build](https://badgen.net/badge/build/check-status/blue)](#build-pipeline-status)

JAX Toolbox provides a public CI, Docker images for popular JAX libraries, and optimized JAX examples to simplify and enhance your JAX development experience on NVIDIA GPUs. It supports JAX libraries such as [MaxText](https://github.com/google/maxtext) and [Pallas](https://jax.readthedocs.io/en/latest/pallas/quickstart.html).

## Frameworks and Supported Models
We support and test the following JAX frameworks and model architectures. More details about each model and available containers can be found in their respective READMEs.

| Framework | Models | Use cases | Container |
| :--- | :---: | :---: | :---: |
| [maxtext](./rosetta/rosetta/projects/maxtext)| GPT, LLaMA, Gemma, Mistral, Mixtral | pre-training | `ghcr.io/nvidia/jax:maxtext` |
| [t5x](./rosetta/rosetta/projects/t5x) | T5, ViT | pre-training, fine-tuning | `ghcr.io/nvidia/jax:t5x` |
| [t5x](./rosetta/rosetta/projects/imagen) | Imagen | pre-training | `ghcr.io/nvidia/t5x:imagen-2023-10-02.v3` |
| [axlearn](./docs/frameworks/axlearn/README.md) | Fuji | pre-training | `ghcr.io/nvidia/jax:axlearn` |

# Build Pipeline Status
<table>
  <thead>
    <tr>
      <th colspan=4 style="text-align:center;">
        <a href="https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/ci.yaml?query=event%3Aschedule+branch%3Amain">
        <img
          style="height: 1.5em;"
          src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-workflow-metadata.json&logo=github-actions&logoColor=white"
        />
        </a>
      </th>
    </tr>
    <tr>
      <th>Components</th>
      <th>Container</th>
      <th>Build</th>
      <th>Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <a href="https://github.com/NVIDIA/JAX-Toolbox/blob/main/.github/container/Dockerfile.base">
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=base%3D%7BCUDA%2CcuDNN%2CNCCL%2COFED%2CEFA%7D">
        </a>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:base</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-base-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-base-build-amd64.json&logo=docker&label=amd64"></a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-base-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-base-build-arm64.json&logo=docker&label=arm64"></a>
      </td>
      <td>
        [no tests]
      </td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/NVIDIA/JAX-Toolbox/blob/main/.github/container/Dockerfile.jax">
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=core%3D%7Bbase%2CJAX%2CFlax%2CTE%7D">
        </a>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:jax</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-jax-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-jax-build-amd64.json&logo=docker&label=amd64"></a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-jax-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-jax-build-arm64.json&logo=docker&label=arm64"></a>
      </td>
      <td>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae#file-badge-jax-unit-test-a100-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-jax-unit-test-A100.json&logo=nvidia&label=A100">
        </a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae#file-badge-te-unit-test-A100-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-te-unit-test-A100.json&logo=nvidia&label=TransformerEngine%20A100">
        </a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae#file-badge-transformer-engine-test.json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-transformer-engine-test.json&logo=nvidia&label=TransformerEngine%20H100">
        </a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae#file-badge-nsys-jax-unit-test-a100-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-nsys-jax-unit-test-A100.json&logo=nvidia&label=nsys-jax A100">
        </a>
      </td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/NVIDIA/JAX-Toolbox/blob/main/.github/container/Dockerfile.equinox">
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=Equinox%3D%7Bcore%2CEquinox%7D">
        </a>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:equinox</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-equinox-md">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-equinox-build-amd64.json&logo=docker&label=amd64">
        </a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-equinox-md">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-equinox-build-arm64.json&logo=docker&label=arm64">
        </a>
      </td>
      <td>
        [tests disabled]
      </td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/NVIDIA/JAX-Toolbox/blob/main/.github/container/Dockerfile.t5x">
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=Upstream%20T5X%3D%7Bcore%2CT5X%7D">
        </a>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:upstream-t5x</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-upstream-t5x-md">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-t5x-build-amd64.json&logo=docker&label=amd64">
        </a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-upstream-t5x-md">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-t5x-build-arm64.json&logo=docker&label=arm64">
        </a>
      </td>
      <td>
        [tests disabled]
      </td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/Dockerfile.t5x">
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=Rosetta%20T5X%3D%7Bcore%2CT5X%7D">
        </a>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:t5x</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-t5x-md">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-rosetta-build-t5x-amd64.json&logo=docker&label=amd64">
        </a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-t5x-md">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-rosetta-build-t5x-arm64.json&logo=docker&label=arm64">
        </a>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae#file-badge-rosetta-t5x-mgmn-test-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-rosetta-t5x-mgmn-test.json&logo=nvidia&label=A100%20distributed">
        </a>
      </td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/NVIDIA/JAX-Toolbox/blob/main/.github/container/Dockerfile.maxtext">
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=MaxText%3D%7Bcore%2CMaxText%7D">
        </a>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:maxtext</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-maxtext-md">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-maxtext-build-amd64.json&logo=docker&label=amd64">
        </a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-maxtext-md">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-maxtext-build-arm64.json&logo=docker&label=arm64">
        </a>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae#file-badge-maxtext-test-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-maxtext-test.json&logo=nvidia&label=A100%20distributed">
        </a>
      </td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/NVIDIA/JAX-Toolbox/blob/main/.github/container/Dockerfile.axlearn">
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=AXLearn%3D%7Bcore%2CAXLearn%7D">
        </a>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:axlearn</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-axlearn-md">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-axlearn-build-amd64.json&logo=docker&label=amd64">
        </a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-maxtext-md">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-axlearn-build-arm64.json&logo=docker&label=arm64">
        </a>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae#file-badge-maxtext-test-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-axlearn-test.json&logo=nvidia&label=H100%20distributed">
        </a>
      </td>
    </tr>
  </tbody>
</table>

In all cases, `ghcr.io/nvidia/jax:XXX` points to latest nightly build of the container for `XXX`. For a stable reference, use `ghcr.io/nvidia/jax:XXX-YYYY-MM-DD`.

In addition to the public CI, we also run internal CI tests on H100 SXM 80GB and A100 SXM 80GB.

## Environment Variables

The [JAX image](https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax) is embedded with the following flags and environment variables for performance tuning of XLA and NCCL:

| XLA Flags | Value | Explanation |
| --------- | ----- | ----------- |
| `--xla_gpu_enable_latency_hiding_scheduler` | `true`  | allows XLA to move communication collectives to increase overlap with compute kernels |


| Environment Variable | Value | Explanation |
| -------------------- | ----- | ----------- |
| `NCCL_NVLS_ENABLE` | `0` | Disables NVLink SHARP ([1](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nvls-enable)). Future releases will re-enable this feature. |

There are various other XLA flags users can set to improve performance. For a detailed explanation of these flags, please refer to the [GPU performance](./rosetta/docs/GPU_performance.md) doc. XLA flags can also be tuned per workload. For example, each script includes a directory [xla_flags](./rosetta/rosetta/projects/maxtext/xla_flags).

For a list of previously used XLA flags that are no longer needed, please also refer to the [GPU performance](./rosetta/docs/GPU_performance.md#previously-used-xla-flags) page.

## Versions

| First nightly with new base container | Base container |
| ------------------------------------- | -------------- |
| 2025-04-11 | nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-devel-ubuntu24.04 |
| 2025-03-04 | nvcr.io/nvidia/cuda-dl-base:25.02-cuda12.8-devel-ubuntu24.04 |
| 2025-01-31 | nvcr.io/nvidia/cuda-dl-base:25.01-cuda12.8-devel-ubuntu24.04 |
| 2025-01-28 | nvcr.io/nvidia/cuda-dl-base:24.11-cuda12.6-devel-ubuntu24.04 |
| 2024-12-07 | nvidia/cuda:12.6.3-devel-ubuntu22.04 |
| 2024-11-06 | nvidia/cuda:12.6.2-devel-ubuntu22.04 |
| 2024-09-25 | nvidia/cuda:12.6.1-devel-ubuntu22.04 |
| 2024-07-24 | nvidia/cuda:12.5.0-devel-ubuntu22.04 |


## Profiling
See [this page](./docs/profiling.md) for more information about how to profile JAX programs on GPU.

## Frequently asked questions (FAQ)

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
* [JAX | NVIDIA NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax)
* [Slurm and OpenMPI zero config integration](https://jax.readthedocs.io/en/latest/_autosummary/jax.distributed.initialize.html)
* [Adding custom GPU ops](https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html)
* [Triaging regressions](docs/triage-tool.md)

## Videos
* [Equinox for JAX: The Foundation of an Ecosystem for Science and Machine Learning](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62668/)
* [Scaling Grok with JAX and H100](https://www.nvidia.com/en-us/on-demand/session/gtc24-s63257/)
* [JAX Supercharged on GPUs: High Performance LLMs with JAX and OpenXLA](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62246/)
* [What's New in JAX | GTC Spring 2024](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62659/)
* [What's New in JAX | GTC Spring 2023](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51956/)
