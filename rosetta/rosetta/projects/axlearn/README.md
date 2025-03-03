# AXLearn
[AXLearn](https://github.com/apple/axlearn) is a deep learning design framework, built on top of JAX and XLA, to support the development of large-scale models. 


## Hardware and Software Specifications

Functionality have been validated on AWS p5.48xlarge EKS cluster (8x H100 80G); please refer to the [Configs](#configs) section below for some initial configs and performance numbers. We will continue to populate it with more models and configs. We provide both singlenode and multinode pre-training support. If running on a machine with less than 80G memory, some of the default configurations may run out of memory; if you run out of memory and have more GPUs available, increase your GPU count and decrease your batch size per GPU.


## Containers
We provide a fully built and ready-to-use multi-arch container, bleeding edge: `ghcr.io/nvidia/jax:axlearn`. We also provide nightly dated images with the naming pattern `ghcr.io/nvidia/jax:axlearn-YYYY-MM-DD`, but we encourage you to use the latest ones for the best performance.

*Note*: All paths mentioned in subsequent sections are relative to the top-level directory of the AXLearn repository. When working interactively with containers, make sure you navigate to `/opt/axlearn` before running any commmands.

## Launching a container
Use the following command to launch a container:
```
docker run -ti --gpus=all --net=host --ipc=host -v <WORKSPACE_PATH>:/opt/axlearn/workspace -w /opt/axlearn <CONTAINER> /bin/bash
```
where `WORKSPACE_PATH` is the path to the directory where you would like to store any persistent files and `container` is the name of the maxtext container. You can additionally add dataset and vocab paths with the `-v` flag.

## Running a Fuji model
### Quick Runs

#### EKS Single node: `fuji-3B-v3-flash-single-host`
Fuji models are defined with 1B, 3B, 7B or 70B parameters. In this example, we deploy the training for a Fuji-3B model, that uses flash attention, and runs on a single host. [Here](scripts/eks-fuji.yaml) we provide an example deployment file. The core point of the deployment is: 
```bash 
python3 -m axlearn.common.launch_trainer_main \
        --module=text.gpt.c4_trainer \
        --config=${CONFIG} \
        --trainer_dir=${TRAINER_DIR} \
        --data_dir=gs://axlearn-public/tensorflow_datasets \
        --jax_backend=gpu             
```
Where `CONFIG="fuji-3B-v3-flash-single-host`. The input dataset is the public tensorflow [C4 dataset](https://www.tensorflow.org/datasets/catalog/c4). 

#### Running a multinode job for `fuji-XB-v2-flash` 

For running a multinode job  we provide a [custom example](scripts/multinode.py). The code access AXLearn directly, it allows to specify a custom dataset, the number of GPUs to use, the global batch size, as well as the `max_sequence_length`. 


## XLA Flags
The [GPU Performance document](../../../docs/GPU_performance.md) provides a detailed description of the XLA flags that can be set to optimize performance. These are the recommended XLA flags to get good performance for AXLearn.

```
XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
            --xla_gpu_enable_triton_gemm=false
            --xla_gpu_enable_command_buffer=
            --xla_gpu_all_reduce_combine_threshold_bytes=1073741824 
            --xla_gpu_all_gather_combine_threshold_bytes=1073741824 
            --xla_gpu_reduce_scatter_combine_threshold_bytes=1073741824
            --xla_gpu_enable_pipelined_all_gather=true 
            --xla_gpu_enable_pipelined_reduce_scatter=true 
            --xla_gpu_enable_pipelined_all_reduce=true 
            --xla_gpu_enable_while_loop_double_buffering=true
            --xla_gpu_enable_all_gather_combine_by_dim=false 
            --xla_gpu_enable_reduce_scatter_combine_by_dim=false 
            --xla_disable_hlo_passes=rematerialization"
```