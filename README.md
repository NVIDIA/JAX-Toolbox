# JAX Toolbox

<table>
  <thead>
    <tr>
      <th colspan=4 style="text-align:center;">
        <img
          src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-workflow-metadata.json&logo=github-actions&logoColor=white"
          href="https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/ci.yaml"
        />
      </th>
    </tr>
    <tr>
      <th colspan=2>Components</th>
      <th>Build</th>
      <th>Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <img src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=base%3D%7BCUDA%2CcuDNN%2CNCCL%2COFED%2CEFA%7D">
      </td>
      <td></td>
      <td>
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-base-build-amd64.json&logo=docker&label=amd64">
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-base-build-arm64.json&logo=docker&label=arm64">
      </td>
      <td>
        n/a
      </td>
    </tr>
    <tr>
      <td>
        <img src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=core%3D%7Bbase%2CJAX%2CFlax%2CTE%7D">
      </td>
      <td></td>
      <td>
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-jax-build-amd64.json&logo=docker&label=amd64">
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-jax-build-arm64.json&logo=docker&label=arm64">
      </td>
      <td>
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-jax-unit-test-V100.json&logo=nvidia">
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-jax-unit-test-A100.json&logo=nvidia">
      </td>
    </tr>
    <tr>
      <td>
        <img src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=Levanter%3D%7Bcore%2CLevanter%7D">
      </td>
      <td></td>
      <td>
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-levanter-build-amd64.json&logo=docker&label=amd64">
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-levanter-build-arm64.json&logo=docker&label=arm64">
      </td>
      <td>
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-levanter-unit-test-V100.json&logo=nvidia">
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-levanter-unit-test-A100.json&logo=nvidia">
      </td>
    </tr>
    <tr>
      <td>
        <img src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=Pallas%3D%7Bcore%2CTriton%2CPallas%7D">
      </td>
      <td></td>
      <td>
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-pallas-build-amd64.json&logo=docker&label=amd64">
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-pallas-build-arm64.json&logo=docker&label=arm64">
      </td>
      <td>
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-pallas-unit-test-V100.json&logo=nvidia">
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-pallas-unit-test-A100.json&logo=nvidia">
      </td>
    </tr>
    <tr>
      <td rowspan=2>
        <img src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=T5X%3D%7Bcore%2CT5X%7D">
      </td>
      <td>upstream</td>
      <td>
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-t5x-build-amd64.json&logo=docker&label=amd64">
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-t5x-build-arm64.json&logo=docker&label=arm64">
      </td>
      <td>
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-t5x-mgmn-test.json&logo=nvidia">
      </td>
    </tr>
    <tr>
      <td>rosetta</td>
      <td>
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-rosetta-build-t5x-amd64.json&logo=docker&label=amd64">
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-rosetta-build-t5x-arm64.json&logo=docker&label=arm64">
      </td>
      <td>
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Frosetta-t5x-overall-test-status.json&logo=nvidia">
      </td>
    </tr>
    <tr>
      <td rowspan=2>
        <img src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=PAX%3D%7Bcore%2Cpaxml%2Cpraxis%7D">
      </td>
      <td>upstream</td>
      <td>
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-pax-build-amd64.json&logo=docker&label=amd64">
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-pax-build-arm64.json&logo=docker&label=arm64">
      </td>
      <td>
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-pax-mgmn-test.json&logo=nvidia">
      </td>
    </tr>
    <tr>
      <td>rosetta</td>
      <td>
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-rosetta-build-pax-amd64.json&logo=docker&label=amd64">
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-rosetta-build-pax-arm64.json&logo=docker&label=arm64">
      </td>
      <td>
        <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Frosetta-pax-overall-test-status.json&logo=nvidia">
      </td>
    </tr>
  </tbody>
</table>

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
