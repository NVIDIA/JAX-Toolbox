# MaxText
[MaxText](https://github.com/google/maxtext) is high performance scalable LLM framework by Google written in Python and JAX. We support the upstream maxtext and have containers that can support the MaxText main branch out-of-the-box. While training, we strongly recommend to use propoer XLA flags pointed below.

## Hardware and Software Specifications
Functionality and performance have been validated on NVIDIA DGX H100 (8x H100 80G) nodes; for details, please refer to the `results and config`section below. We provide both singlenode and multinode pre-training support. If running on a machine with less than 80G memory, some of the default configurations may run out of memory; if you run out of memory and have more GPUs available, increase your GPU count and decrease your batch size per GPU.

The [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) is required to run the subsequent commands with GPU support. Ensure the NVIDIA Container Toolkit is installed before proceeding.

## Containers
We provide a fully built and ready-to-use multi-arch container which includes the latest optimizations, experimental features, and examples benchmarked for multi-node, multi-GPU training: `nvcr.io/nvidia/jax:24.04-maxtext-py3` (amd64 support). Verified containers will be updated periodically, but if you wish to use the bleeding edge (which may come with unexpected behavior), please use `ghcr.io/nvidia/jax:maxtext`. We also provide nightly dated images with the naming pattern `ghcr.io/nvidia/jax:maxtext-YYYY-MM-DD`, but we encourage you to use the latest ones for the best performance.

*Note*: All paths mentioned in subsequent sections are relative to the top-level directory of the MaxText repository. When working interactively with containers, make sure you navigate to `/opt/maxtext` before running any commmands.

## Downloading the C4 dataset
You can use `download_dataset.sh` script to download the C4 dataset. For details regarding the dataset download please see these [guidelines](https://github.com/google/maxtext/blob/main/getting_started/First_run.md). Alternatively, you can pass set this argument `dataset_type=synthetic` while launching the training script to use a synthetic dataset which is quite helpful to debug the initial performance. For more datasets, please see the [Paxml readme](https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/rosetta/projects/pax/README.md#downloading-the-pile-and-lambada-datasets).

## Launching a container
Use the following command to launch a container:
```
docker run -ti --gpus=all --net=host --ipc=host -v <WORKSPACE_PATH>:/opt/maxtext/workspace -w /opt/maxtext <CONTAINER> /bin/bash
```
where `WORKSPACE_PATH` is the path to the directory where you would like to store any persistent files and `container` is the name of the maxtext container. You can additionally add dataset and vocab paths with the `-v` flag.

## Running a job
Once the container is up and running, you can quickly launch a job with the following command
```
python3 MaxText/train.py MaxText/configs/base.yml hardware-gpu run_name=$YOUR_JOB_NAME
```
### Running a multinode job
Please see the (example_slurm.sub)[./scripts/example_slurm.sub] for a multinode multiprocess job.

... more details to be included

## XLA Flags
The [GPU Performance document](../../../docs/GPU_performance.md) provides a detailed description of the XLA flags that can be set to optimize performance. These are the recommended XLA flags to get good performance for MaxText.

```
XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true 
            --xla_gpu_enable_async_all_gather=true 
            --xla_gpu_enable_async_reduce_scatter=true 
            --xla_gpu_enable_triton_gemm=false
            --xla_gpu_simplify_all_fp_conversions 
            --xla_gpu_graph_level=0 
            --xla_gpu_enable_async_all_reduce=true 
            --xla_gpu_enable_highest_priority_async_stream=true
            --xla_gpu_all_reduce_combine_threshold_bytes=1073741824 
            --xla_gpu_all_gather_combine_threshold_bytes=1073741824 
            --xla_gpu_reduce_scatter_combine_threshold_bytes=134217728
            --xla_gpu_enable_pipelined_all_gather=true 
            --xla_gpu_enable_pipelined_reduce_scatter=true 
            --xla_gpu_enable_pipelined_all_reduce=true 
            --xla_gpu_enable_while_loop_double_buffering=true
            --xla_gpu_enable_triton_softmax_fusion=false 
            --xla_gpu_enable_all_gather_combine_by_dim=false 
            --xla_gpu_enable_reduce_scatter_combine_by_dim=false 
            --xla_disable_hlo_passes=rematerialization"
```

# Notes
1. The only changes we need to support multiprocessing is to pin tensorflow and tensorflow-text to 2.13.0 version.
2. In order to remove extra copies introduced by DUS (dynamic update slice) when used in conjunction with custom NVIDIA kernels (like cuBLAS for GEMMs), the `--xla_gpu_enable_custom_fusions` and `--xla_gpu_enable_address_computation_fusion` flags were introduced. However, the current XLA has some limitation and sometimes using these flags lead to error. So, in this release, it is advised to turn off these two flags:
    - --xla_gpu_enable_custom_fusions=false
    - --xla_gpu_enable_address_computation_fusion=false

