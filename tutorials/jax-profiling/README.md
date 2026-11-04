# JAX Profiling Tutorials

Tutorial notebooks for profiling JAX programs on NVIDIA GPUs.

- [Diagnose Compute- and Memory-Bound Custom JAX Kernels](https://github.com/NVIDIA/JAX-Toolbox/blob/main/tutorials/jax-profiling/diagnose_compute_and_memory_bound_custom_jax_kernels.ipynb)

## Prerequisites

- A Linux GPU host with an NVIDIA Ampere or newer GPU (compute capability 8.0+) and a driver compatible with the CUDA version you use.
- CUDA 12.9+, JAX 0.8.1+, and CUTLASS DSL 4.4+ for the [CuTe DSL integration](https://docs.jax.dev/en/latest/401/cute-dsl.html).
- NVIDIA Nsight Systems (`nsys`), Nsight Compute 2026.2.1+ (`ncu`), and Nsight Python 1.0+. The `nsys` and `ncu` executables must be on `PATH`.
- Permission to access GPU performance counters. If profiling reports `ERR_NVGPUCTRPERM`, see [NVIDIA's permissions guide](https://developer.nvidia.com/ERR_NVGPUCTRPERM).

## Run in a container

The [NVIDIA NGC JAX container](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/jax) is recommended, but not required. The example below uses `26.10-py3`; see its [release notes](https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/rel-26-10.html) for driver requirements. CUDA and JAX are provided inside the container.

For this workflow, install Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the GPU host. From the repository root, run:

```bash
docker run --rm -it \
  --gpus all \
  --cap-add=SYS_ADMIN \
  -p 127.0.0.1:8888:8888 \
  -v "$PWD/tutorials/jax-profiling:/tutorial" \
  -w /tutorial \
  nvcr.io/nvidia/jax:26.10-py3 bash
```

`--cap-add=SYS_ADMIN` grants profiling access inside the container. The directory mount keeps notebooks and generated reports on the host.

Inside the container, install the Python dependencies and start JupyterLab:

```bash
python -m pip install jupyterlab ipykernel "nsight-python>=1.0" "nvidia-cutlass-dsl>=4.4" nvtx
python -m jupyterlab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Open `http://localhost:8888/lab` using the token printed in the terminal. For a remote GPU host, forward the port from your local machine with `ssh -L 8888:localhost:8888 user@gpu-host`.

Generated scripts and reports are written to `profile_artifacts/`.

## Use another environment

You can use another compatible container or a Linux Python environment that meets the prerequisites above. For a local installation, follow the [JAX GPU installation instructions](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu), install the Python dependencies listed above, and make `nsys` and `ncu` available on `PATH`.

Start JupyterLab from `tutorials/jax-profiling/` with `python -m jupyterlab --no-browser`. Docker and the NVIDIA Container Toolkit are only needed for the container workflow.
