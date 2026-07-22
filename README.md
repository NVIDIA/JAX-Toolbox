<div align="center">

JAX Toolbox
===========================

**JAX Toolbox** is NVIDIA's home for [**JAX and XLA on GPUs**](https://docs.nvidia.com/jax-toolbox/getting-started/jax-on-nvidia-gpu-stack), including the latest updates, containers and educational resources.

[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](https://github.com/NVIDIA/JAX-Toolbox/blob/main/LICENSE.md)
[![Build](https://badgen.net/badge/build/check-status/blue)](#build-pipeline-status)

[**Latest news**](https://github.com/NVIDIA/JAX-Toolbox/#latest-news)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[**Tech blogs**](https://github.com/NVIDIA/JAX-Toolbox/#tech-blogs)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[**Documentation**](https://docs.nvidia.com/jax-toolbox/introduction)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[**Resources**](https://github.com/NVIDIA/JAX-Toolbox/#resources)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[**Container images**](https://github.com/NVIDIA/JAX-Toolbox/#container-images)

---
<div align="left">

## What's included

- **Latest news**: New NGC JAX releases, new hardware support, major JAX and XLA optimizations and updates on the JAX on GPU stack.

- **Tech blogs**: Blogs highlighting key technical challenges, practical guidance and observed results

- **Documentation**: Overview of JAX on GPUs, customizing JAX with NVIDIA CUDA Toolkit and going beyond what’s possible in native JAX, practical guides for profiling, optimizing, scaling and third-party tool integrations in JAX.

- **Resources**: Talks, presentations and ecosystem references

- **Container images**: Nightly builds, monthly NGC releases, and staging containers for JAX on GPU projects

- **Experimental projects**: Experimental projects for developers to evaluate and provide feedback on

## Latest news
- [**NGC JAX 26.06**](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax/tags?version=26.06-py3) container image released
  - [`CUDA 13.3.0`](https://docs.nvidia.com/deeplearning/frameworks/cuda-dl-release-notes/rel-26-06.html)
  - Improved GPU dot fusion cost model and introduced tileable GemmFusion.
- [**NGC JAX 26.05**](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax/tags?version=26.05-py3) container image released
  - performant `jax.ragged_dot`; XLA’s `nvfp4` kernel codegen improvements; multi-stream collective support in XLA; PDL (programmatic dependent launch) enablement in XLA and improved D2H & H2D copy overlap with compute
- [**NGC JAX 26.04**](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax/tags?version=26.04-py3) container image released
  - Thor support in JAX/XLA with full Triton; FP8 reduction support in XLA; and improved collective cost model of GB200/GB300

Detailed release notes for NGC JAX containers are published in the [**NGC JAX release notes**](https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/index.html).

## Tech blogs
- [Jul 2026] [**Reducing High-Bandwidth Memory Bottlenecks in JAX-Based LLM Training with Host Offloading**](https://developer.nvidia.com/blog/reducing-high-bandwidth-memory-bottlenecks-in-jax-based-llm-training-with-host-offloading/)
- [Jun 2026] [**Train Models Faster with JAX and MaxText Using NVFP4 on NVIDIA Blackwell**](https://developer.nvidia.com/blog/train-models-faster-with-jax-and-maxtext-using-nvfp4-on-nvidia-blackwell/)
- [Feb 2026] [**Accelerating Long-Context Model Training in JAX and XLA**](https://developer.nvidia.com/blog/accelerating-long-context-model-training-in-jax-and-xla/)
- [Jul 2025] [**Optimizing for Low-Latency Communication in Inference Workloads with JAX and XLA**](https://developer.nvidia.com/blog/optimizing-for-low-latency-communication-in-inference-workloads-with-jax-and-xla/)

## Resources
### Talks and presentations
#### 2026
- [GTC 2026] [**JAX on Blackwell customer success story**](https://www.nvidia.com/en-us/on-demand/session/gtc26-s82225/)
- [GTC 2026] [**Large-Scale JAX Training With Pipeline Parallelism**](https://www.nvidia.com/en-us/on-demand/session/gtc26-s82141/)
- [NVIDIA Developer] [**Two Ways to Fine-Tune JAX on NVIDIA GPUs: PEFT and SFT with Tunix and MaxText**](https://youtu.be/Zlh49mWVydo?si=cqfEbN2xeQx49LvK)
- [NVIDIA Developer] [**CuTe DSL for JAX Developers: Writing Custom GPU Kernels in Python**](https://youtu.be/4c8qFBbsDb0?si=n7TxMbhEU81cL_PS)

#### 2025
- [JAX/OpenXLA DevLab Fall 2025] [**Accelerating LLM models on Blackwell**](https://youtu.be/QDhSk-MegUg?si=AtciF3lBY0-1dV1Q)
- [JAX/OpenXLA DevLab Fall 2025] [**Leveraging CuTeDSL in JAX**](https://youtu.be/v_DV61ErjAY?si=2COf8AIh8xBIxKPx)
- [JAX/OpenXLA DevLab Fall 2025] [**JaxPP: A library for MPMD training in JAX**](https://www.youtube.com/watch?v=ln51xYyEsRY)
- [GTC 2025] [**Horizontal Scaling of LLM Training with JAX**](https://www.nvidia.com/en-us/on-demand/session/gtc25-s73266/)
- [GTC 2025] [**Scaling Transformers: Navigating Challenges and Innovations in Long-Context Modeling**](https://www.nvidia.com/gtc/session-catalog/?search=&tab.catalogallsessionstab=16566177511100015Kus&search.sessiontype=option_1614028602338#/session/1727994825598001BAtq)

## Container images
### Frameworks and supported models
We support and test the following JAX-based frameworks and model architectures. More details about each model and available containers can be found in their respective READMEs.

Nightly image builds use the latest JAX and XLA and are published to the `ghcr.io/nvidia/jax` image registry. The plain image tag `ghcr.io/nvidia/jax:jax` or `ghcr.io/nvidia/jax:maxtext` is simply the latest image. Append a date suffix to any image tag to specify a particular nightly build, e.g `ghcr.io/nvidia/jax:jax-2026-06-01`.

Stable builds are released at the end of each month to NVIDIA's [**NGC Catalog**](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax) at the `nvcr.io/nvidia/jax` image registry. NGC images pin specific [**JAX releases**](https://github.com/jax-ml/jax/releases). See [**NGC JAX release notes**](https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/index.html) for more details.

| Framework | Models | Use cases | Container image (nightly build) | Container image (NGC release) |
| :--- | :---: | :---: | :---: | :---: |
| [jax](https://github.com/jax-ml/jax)|  | pre-training | `ghcr.io/nvidia/jax:jax` | [`nvcr.io/nvidia/jax:YY.MM-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax/tags?version=26.05-py3) |
| [maxtext](https://github.com/google/maxtext)| [**DeepSeek V3.2**](https://maxtext.readthedocs.io/en/latest/reference/models/supported_models_and_architectures.html#deepseek), [**Gemma 4**](https://maxtext.readthedocs.io/en/latest/reference/models/supported_models_and_architectures.html#gemma), [**Llama 4**](https://maxtext.readthedocs.io/en/latest/reference/models/supported_models_and_architectures.html#llama), [**Mixtral**](https://maxtext.readthedocs.io/en/latest/reference/models/supported_models_and_architectures.html#mistral-mixtral) | pre-training | `ghcr.io/nvidia/jax:maxtext` | [`nvcr.io/nvidia/jax:YY.MM-maxtext-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax/tags?version=26.05-maxtext-py3) |
| [axlearn](./docs/frameworks/axlearn/README.md) | Fuji | pre-training | `ghcr.io/nvidia/jax:axlearn` | |
| [alphafold3](https://github.com/google-deepmind/alphafold3) | Evoformer | inference | `ghcr.io/nvidia/jax:alphafold` | |


### Nightly build pipeline status
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
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-badge-jax-unit-backend-independent-h100-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-jax-unit-backend-independent-H100.json&logo=nvidia&label=JAX%20-%20backend%20independent%20(H100)">
        </a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-badge-jax-unit-single-gpu-h100-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-jax-unit-single-gpu-H100.json&logo=nvidia&label=JAX%20-%20single%20GPU%20(H100)">
        </a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-badge-jax-unit-multi-gpu-h100-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-jax-unit-multi-gpu-H100.json&logo=nvidia&label=JAX%20multi-GPU%20(H100)">
        </a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-badge-te-l0_jax_unittest-h100-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-te-L0_jax_unittest-H100.json&logo=nvidia&label=TransformerEngine%20-%20unit%20(H100)">
        </a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-badge-te-l0_jax_distributed_unittest-h100-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-te-L0_jax_distributed_unittest-H100.json&logo=nvidia&label=TransformerEngine%20-%20distributed%20(H100)">
        </a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-badge-nsys-jax-unit-h100-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-nsys-jax-unit-H100.json&logo=nvidia&label=nsys-jax-unit%20(H100)">
        </a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-badge-nsys-jax-nccl-h100-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-nsys-jax-nccl-H100.json&logo=nvidia&label=nsys-jax-nccl%20(H100)">
        </a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-badge-jax-cutlass-h100-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-jax-cutlass-H100.json&logo=nvidia&label=CUTLASS%20(H100)">
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
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-badge-maxtext-single-node-h100-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-maxtext-single-node-H100.json&logo=nvidia&label=MaxText%20-%20single-node%20(H100)">
        </a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-badge-maxtext-multi-node-h100-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-maxtext-multi-node-H100.json&logo=nvidia&label=MaxText%20-%20multi-node%20(H100)">
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
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-axlearn-md">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-axlearn-build-arm64.json&logo=docker&label=arm64">
        </a>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae#file-badge-axlearn-test-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-axlearn-test.json&logo=nvidia&label=H100%20distributed">
        </a>
      </td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/NVIDIA/JAX-Toolbox/blob/main/.github/container/Dockerfile.alphafold">
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=AlphaFold%3D%7Bcore%2CAlphaFold%7D">
        </a>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:alphafold</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-alphafold-md">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-alphafold-build-amd64.json&logo=docker&label=amd64">
        </a>
        <br>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-alphafold-md">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-alphafold-build-arm64.json&logo=docker&label=arm64">
        </a>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-badge-alphafold-inference-h100-json">
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-alphafold-inference-H100.json&logo=nvidia&label=AlphaFold%20inference%20(H100)">
        </a>
      </td>
    </tr>
  </tbody>
</table>

In addition to the public CI, we also run internal CI nightlies on [**GB300**](https://www.nvidia.com/en-us/data-center/gb300-nvl72/), [**B300 SXM6**](https://www.nvidia.com/en-us/data-center/dgx-b300/), [**GB200**](https://www.nvidia.com/en-us/data-center/gb200-nvl72/), [**B200**](https://www.nvidia.com/en-us/data-center/dgx-b200/), [**DGX Spark**](https://www.nvidia.com/en-us/products/workstations/dgx-spark/), [**RTX PRO 6000 Blackwell**](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/), [**Jetson AGX Thor**](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/), [**H100 SXM 80GB**](https://www.nvidia.com/en-us/data-center/h100/), A100 SXM 80GB.

## Experimental projects
- [**JAX-vLLM Rollout Offloading Bridge**](https://github.com/NVIDIA/JAX-Toolbox/tree/main/jax-inference-offloading#readme)
  - This project couples JAX training with vLLM inference to accelerate reinforcement-learning (RL) post-training
