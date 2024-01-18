# Pax
[Pax](https://github.com/google/paxml/tree/main) is a framework developed by Google optimized for running machine learning experiments using JAX. Pax consists of the Paxml and [Praxis](https://github.com/google/praxis/tree/main) repositories and is maintained as a [distribution](../../../docs/DEVELOPMENT.md) within Rosetta. This means that we cherry-pick the necessary changes to optimize Pax for GPUs on top of upstream Paxml and Praxis' `main` branches. We also provide support for FP8 training via both [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) and native [XLA-FP8](https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/NATIVE_FP8.md).

Any `paxml/*` or `praxis/*` relative directory/file can be found in [google/paxml](https://github.com/google/paxml/tree/main) or [google/praxis](https://github.com/google/praxis/tree/main), respectively, but to
view the most up-to-date version of that directory/file with any GPU-specific patches, please see [Inspecting the Source Code](#inspecting-the-source-code).

## Hardware and Software Specifications
Convergence and performance has been validated on NVIDIA DGX H100 (8x H100 80G) and A100 (8x A100 80G) nodes; for details, please refer to the [Configs](#configs) section below. We provide both singlenode and multinode pre-training support. If running on a machine with less than 80G memory, some of the default configurations may run out of memory; if you run out of memory and have more GPUs available, increase your GPU count and decrease your batch size per GPU.

The [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) is required to run the subsequent commands with GPU support. Ensure the NVIDIA Container Toolkit is installed before proceeding.

## Containers
We provide a fully built and ready-to-use multi-arch container which includes the latest optimizations, experimental features, and examples benchmarked for multi-node, multi-GPU training: `nvcr.io/nvidia/jax:23.10-paxml-py3` (amd64 and arm64 support). This container also provides FP8 support via [Transformer Engine](https://github.com/NVIDIA/TransformerEngine). Verified containers will be updated periodically, but if you wish to use the bleeding edge (which may come with unexpected behavior), please use `ghcr.io/nvidia/pax:latest`. We also provide nightly dated images with the naming pattern `ghcr.io/nvidia/pax:nightly-YYYY-MM-DD`, but we encourage you to use the latest ones for the best performance.

For more information on the Pax build and for details on how to manually build the Pax distribution, please refer to [DEVELOPMENT.md](../../../docs/DEVELOPMENT.md). 

*Note*: All paths mentioned in subsequent sections are relative to the top-level directory of the Paxml repository. When working interactively with containers, make sure you navigate to `/opt/paxml` before running any commmands.

## Downloading the SentencePiece Model
Pax models require a pretrained SentencePiece model to tokenize the datasets. The SentencePiece model used in the following experiments is `gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model`. This model was trained using [these instructions](https://github.com/sgpyc/training/blob/paxml-llm-draft/large_language_model/paxml/utils/generate_spm.md). Use the following commands to download the tokenizer locally. This should be done _prior_ to launching the container.
```
wget -P c4_sentencepiece https://github.com/nvjax-svc-0/assets/raw/main/sentencepiece_c4/c4_en_301_5Mexp2_spm.model
```
You can then use the following mount to attach the tokenizer to your container:
```
docker run -v ${PWD}/c4_sentencepiece/c4_en_301_5Mexp2_spm.model:/opt/paxml/vocab ...
```

## Launching a container
Use the following command to launch a container:
```
docker run -ti --gpus=all --net=host --ipc=host -v <DATASET_PATH>:/opt/paxml/datasets -v <WORKSPACE_PATH>:/opt/paxml/workspace -v <VOCAB_PATH>:/opt/paxml/vocab -w /opt/paxml <CONTAINER> /bin/bash
```
where `DATASET_PATH` is the path to the Pile or Lambada dataset. If these datasets have not yet been downloaded, they can be downloaded from inside of the container (see [Downloading The Pile and Lambada Datasets](#Downloading-the-pile-and-lambada-datasets) for more). `WORKSPACE_PATH` is the path to the directory where you would like to store any persistent files, and `VOCAB_PATH` is the path to the pretrained SentencePiece model to use during tokenization (see [Downloading the SentencePiece Model](#Downloading-the-sentencepiece-model) for more). 

## Downloading The Pile and Lambada Datasets
__IMPORTANT UPDATE__: Please be aware that as of October 2023, 'the_pile' dataset is no longer accessible. The team is actively updating our instructions and configurations to incorporate a more recent large language model (LLM) dataset. Additionally, we will shortly provide updated instructions that include methods for using synthetic data, ensuring that our users can continue their work without interruption.

The GPT model configs we provide are trained using The Pile dataset and evaluated using the Lambada dataset. The scripts [download_the_pile.py](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/download_the_pile.py) and [download_lambada.py](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/download_lambada.py) will download The Pile and Lambada datasets to the `TFDS_DATA_DIR` enviroment variable. To control the location of the downloaded datasets, use the following command prior to running the download scripts: `export TFDS_DATA_DIR=<path_to_dowload_data_to>`. For example, the following commands download the Pile dataset to `/opt/paxml/datasets/`:
```
export TFDS_DATA_DIR=/opt/paxml/datasets/
python3 paxml/contrib/gpu/scripts_gpu/download_the_pile.py
```

After the data has been successfully downloaded, use the same `TFDS_DATA_DIR` when running experiments.

## Inspecting the Source Code
If you would like to inspect Pax's source code (`paxml/*` and `praxis/*`) to learn more about what is being run, you can do so by inspecting
the source within the container. Here are some examples:

```bash
# (Interactive = already in container): navigate to paxml/contrib/gpu/scripts_gpu/
cd $(python -c 'import paxml; print(paxml.__path__[0])')/../paxml/contrib/gpu/scripts_gpu

# (Non-interactive): View paxml/contrib/gpu/scripts_gpu/configs.py
FILE=paxml/contrib/gpu/scripts_gpu/configs.py
docker run --entrypoint="" --rm <CONTAINER> sh -c 'cat $(python -c "import paxml; print(*paxml.__path__)" 2>/dev/null)/../'$FILE
```

## Running a Job
Note that when training with The Pile dataset, you must provide the `TFDS_DATA_DIR` as a command-line argument and a `VOCAB_PATH` (the path to a pretrained SentencePiece model) as an environment variable. See the bash scripts below for examples. 

### Quick Runs
#### Interactive: Single Node
See [run_pile_singlenode.sh](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/run_pile_singlenode.sh) for an example of training a 126M parameter model on a single node using The Pile. Once inside of your container, this script can be run interactively using the following command:
``` 
bash paxml/contrib/gpu/scripts_gpu/run_pile_singlenode.sh <TFDS_DATA_DIR> <VOCAB_PATH> <PRECISION> <NUM_GPUS> <PERCORE_BATCH_SIZE> <LOGDIR>
```
where `TFDS_DATA_DIR` is the path to The Pile dataset, `VOCAB_PATH` is the path to the pretrained SentencePiece `.model` file, and `LOGDIR` is the relative path of the directory to which to write checkpoints and logging information. `PERCORE_BATCH_SIZE` is the batch size per GPU _prior_ to sharding according to the parallel strategy. See [Customized Runs](#Customized-runs) for more information about this hyperparameter. 

For example, to train the 126M model using a percore batch size of 4 on 8 H100 gpus, you can use the following command:
```
ENABLE_FP8=1 bash paxml/contrib/gpu/scripts_gpu/run_pile_singlenode.sh /opt/paxml/datasets /opt/paxml/vocab bfloat16 8 4 log_dir
```

See [run_lambada_singlenode.sh](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/run_lambada_singlenode.sh) for an example of running zero-shot evaluation on the 126M model using the Lambada dataset. Use the following command to run this script:
``` 
bash paxml/contrib/gpu/scripts_gpu/run_lambada_singlenode.sh <TFDS_DATA_DIR> <VOCAB_PATH> <PRECISION> <NUM_GPUS> <PERCORE_BATCH_SIZE> <LOGDIR>
```
`TFDS_DATA_DIR` should contain the path to the Lambada dataset and `LOGDIR` should match the `LOGDIR` from the pretraining run. Note that a pre-trained checkpoint is required in order for this script to run successfully.

#### Multi Node
See [example_slurm_pile.sub](https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/rosetta/projects/pax/scripts/example_slurm_pile.sub) for an example slurm submit file that launches an 8-node training run with a 126 million parameter GPT model.

To launch `example_slurm_pile.sub`, run the following command:
```
CONTAINER=<CONTAINER> BASE_WORKSPACE_DIR=<PATH_TO_WORKSPACE> BASE_TFDS_DATA_DIR=<PATH_TO_THE_PILE> BASE_VOCAB_PATH=<PATH_TO_SENTENCEPIECE_MODEL> LOG_DIR_LOCAL=<LOG_DIR_LOCAL> OUTPUT_DIR=<OUTPUT_DIR_LOCAL> PREC=bfloat16 GPUS_PER_NODE=8 PERCORE_BATCH_SIZE=4 ENABLE_FP8=<ENABLE_FP8> sbatch -N 8 -A <ACCOUNT> -p <PARTITION> -J <JOBNAME> scripts/example_slurm_pile.sub
```
where `BASE_WORKSPACE_DIR`, `BASE_TFDS_DATA_DIR`, and `BASE_VOCAB_PATH` are absolute paths and `LOG_DIR` and `OUTPUT_DIR` are relative to `BASE_WORKSPACE_DIR`.
    
### Customized Runs
Paxml's [main.py](https://github.com/google/paxml/blob/main/paxml/main.py) takes an experiment config as a command-line argument via the `--fdl_config` flag. To control which model to run, swap out the experiment config passed to `main.py`. For example, in [run_pile_multinode.sh](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/run_pile_multinode.sh), we run the experiment [Pile126M](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/configs.py#L177-L181):
```
    ...
    --fdl_config=paxml.contrib.gpu.scripts_gpu.configs.Pile126M \
    ...
```

Paxml uses [Fiddle](https://github.com/google/fiddle/tree/main) for configuring hyperparameters. To overwrite an existing hyperparameter from the command line, use the following syntax: 
```
--fdl.<PARAM_NAME>=<NEW_VALUE>
```
For example, in our `*.sh` scripts, we override the default values of `FPROP_DTYPE`, `ICI_MESH_SHAPE`, and `PERCORE_BATCH_SIZE`. 

We provide a list of some of the frequently overridden hyperparameters, and an explanation of each, below:
- `ICI_MESH_SHAPE`: This refers to the parallelism strategy used on chips connected by a fast network (e.g. NVLink). `ICI_MESH_SHAPE` typically has 3 dimensions, `[data, fsdp, tensor]`, corresponding to data parallelism (DP), fully-sharded data parallelism (FSDP/ZeRO-3), and tensor parallelism (TP), respectively. For example,to use pure data parallelism, you should set `ICI_MESH_SHAPE` to `[NUM_GPUS, 1, 1]`.
- `DCN_MESH_SHAPE`: This refers to the parallelism strategy for machines connected by a datacenter network. In our case, this refers to the parallel strategy used _across_ nodes. It has the same dimensions as `ICI_MESH_SHAPE`.
- `PERCORE_BATCH_SIZE`: This is the batch size loaded by each worker _prior_ to sharding the data according to the parallel strategy. We should always have that `GLOBAL_BATCH_SIZE = PERCORE_BATCH_SIZE * NUM_GPUS`, regardless of the parallel strategy. Note that a consequence of this is that `PERCORE_BATCH_SIZE` will not always equal `MICROBATCH_SIZE`, particularly when using tensor parallelism (TP). If using 2-way TP, for example, `MICROBATCH_SIZE` will be twice the `PERCORE_BATCH_SIZE`. If using tensor or pipeline parallelism, `PERCORE_BATCH_SIZE` may be fractional. For example, when using 2-way TP, setting `PERCORE_BATCH_SIZE` to 0.5 will result in a microbatch size of `PERCORE_BATCH_SIZE * TP = 1`.
- `NUM_LAYERS`, `NUM_HEADS`, `MODEL_DIMS`, `HIDDEN_DIMS`: These are hyperparameters of the transformer model. `MODEL_DIMS` refers to the hidden dimension of the transformer and `HIDDEN_DIMS` refers to the hidden dimension of the transformer feed-forward network.

We provide three "base" configurations in `paxml/contrib/gpu/scripts_gpu/configs.py`. For more information about these configurations and how to run experiments using them, please refer to the [Configs](#Configs) section below.

### Transformer Engine
Training using Transformer Engine (TE) with bfloat16 precision is controlled via the environment variable `ENABLE_TE`. TE is enabled by default in the prebuilt container, but if you would like to disable TE, you can do so by flipping the value of `ENABLE_TE` in the container:
```
export ENABLE_TE=0
```

FP8 training is controlled via the `ENABLE_FP8` environment variable. To enable FP8 training, set `ENABLE_FP8=1`. For example, the following command trains a 126M model on a single node using FP8:
```
ENABLE_FP8=1 bash paxml/contrib/gpu/scripts_gpu/run_pile_singlenode.sh /opt/paxml/datasets /opt/paxml/vocab bfloat16 8 4 log_dir
```

Note that packing is currently not supported when using TE. All configs disable packing by default, but beware that if packing is manually enabled, training with TE will error.

### Native FP8
Rosetta Pax containers also provide support for native FP8 through XLA. Enabling FP8 can be done by adding the following command-line flag to `paxml/contrib/gpu/scripts_gpu/run_pile_singlenode.sh`: `--fdl.USE_FP8=True`. When using native FP8, TE must be disabled. For a detailed explanation of native FP8 support in Pax, as well as a comparison between native FP8 and TE FP8, please refer to the [NATIVE_FP8](https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/NATIVE_FP8.md) documentation.

## XLA Flags
We recommend setting the following XLA flags when running experiments: 

1. `--xla_gpu_simplify_all_fp_conversions`: Allows all floating-point `f32 -> bf16 -> f32` conversion pairs to be simplified.
2. `--xla_gpu_enable_latency_hiding_scheduler=true`: Allows XLA:GPU to move communication collectives to increase overlap with compute kernels
3. `--xla_gpu_enable_async_all_gather=true`: Allows XLA:GPU to run All Gather NCCL kernels on a separate CUDA stream to allow overlap with compute kernels
4. `--xla_gpu_enable_async_reduce_scatter=true`: Allows XLA:GPU to run Reduce Scatter NCCL kernels on a separate CUDA stream to allow overlap with compute kernels
5. `--xla_gpu_enable_async_all_reduce=true`: Allows XLA:GPU to run All Reduce NCCL kernels on a separate CUDA stream to allow overlap with compute kernels.
6. `--xla_gpu_enable_highest_priority_async_stream=true`: Allows XLA to prioritize the launch of NCCL kernels before GeMMs to ensure enough SMs are available for async communication kernels.
7. `--xla_gpu_all_reduce_combine_threshold_bytes=<BYTES>`: Combines NCCL All Reduce kernels until threshold size is reached. For 126M, we recommend setting this value to 33554432. For 5B and 175B, we recommend 51200.
8. `--xla_gpu_enable_triton_gemm=false`: Disallows Triton GeMM kernels; uses CUBLAS GeMM kernels instead. CUBLAS kernels are currently better tuned for GPUs and thus provide better performance
9. `--xla_gpu_cuda_graph_level=0`: Disallows XLA from using CUDA graphs.

These flags are enabled by default in `paxml/contrib/gpu/scripts_gpu/run_pile_multinode.sh`. The `XLA_FLAGS` environment variable controls these flags; to configure XLA flags explicitly, you can use the following command.
```
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true --xla_gpu_enable_triton_gemm=false"
```
For the the 126M model, we recommend setting `--xla_gpu_all_reduce_combine_threshold_bytes=33554432`, which is different from the default value in `paxml/contrib/gpu/scripts_gpu/run_pile_multinode.sh`. To overwrite the default XLA flags set in the script, set the `BASE_XLA_FLAGS` environment variable prior to calling `run_pile_multinode` as follows:

```
BASE_XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false
                --xla_gpu_simplify_all_fp_conversions --xla_gpu_enable_async_all_gather=true
                --xla_gpu_enable_async_reduce_scatter=true  --xla_gpu_enable_highest_priority_async_stream=true
                --xla_gpu_enable_triton_softmax_fusion=false  --xla_gpu_all_reduce_combine_threshold_bytes=33554432
                --xla_gpu_graph_level=0 --xla_gpu_enable_async_all_reduce=true" bash run_pile_multinode.sh ...
```

## Configs
We provide three "base" model configurations in `paxml/contrib/gpu/scripts_gpu/configs.py`. The first is a 126 million parameter GPT model. Convergence using The Pile dataset has been verified with this model. The remaining configs are 5 billion and 175 billion parameter models. Both 5B and 175B are provided primarily for benchmarking purposes and been less thoroughly tested for convergence.

The tables below describe current performance of the given configs. Experiments were run using NVIDIA DGX A100 80G and H100 80G nodes. Note that Lambada accuracy reported corresponds to the best accuracy seen across the run. Estimated walltime denotes the aproximate time to train each model to completion (i.e. number of days to reach `MAX_STEPS` number of steps as described in `configs.py`).

### A100 Results

| Size | GPU | Precision | #GPUs | DP | FSDP | TP | BS / GPU | Sequences/Sec | Est. Walltime (days) | Lambada Accuracy (± standard deviation) | Convergence Log |
| ---- | ----- |----- |----- | -- | ---- | -- | ---------| ---------------| ------------------------- | ---------------- |---------------- |
| 126M | A100 80G SXM | BF16 |  64    |64    |1    |1    | 4     |   1877.20  |         0.95        |   0.397 (± 0.012)     | [log](https://tensorboard.dev/experiment/RCroDLAUQzGUoudzqD1NmQ/) |
| 5B   | A100 80G SXM | BF16 | 256    | 1    |256    |1    | 8       |  465.45     |       3.82           |       N/A        |            |
| 175B | A100 80G SXM | BF16 | 256    |1    |256    |1    | 6       |   18.29     |        72.92         |        N/A       |    |
| 126M | A100 80G SXM | TE BF16 |  64    |64    |1    |1    | 4     |  2512.2   |     0.71            |   N/A | |
| 5B   | A100 80G SXM | TE BF16 | 256    | 1    |256    |1    | 8       | 586.82    |    3.02    |       N/A        |            |
| 175B | A100 80G SXM | TE BF16 | 256    |1    |256    |1    | 6       |   19.47    |      68.49     |        N/A       |    |

## H100 Results

| Size | GPU | Precision | #GPUs | DP | FSDP | TP | BS / GPU | Sequences/Sec | Est. Walltime (days) | Lambada Accuracy (± standard deviation) | Convergence Log |
| ---- | ----- |----- |----- | -- | ---- | -- | ---------| ---------------| ------------------------- | ---------------- |---------------- |
| 126M | H100 80G SXM | TE BF16 |  64    |64    |1    |1    | 4        |  4143.21  |     0.43           |    0.425 (± 0.018)        | [log](https://tensorboard.dev/experiment/GgDMwODzQjm9kVc9H6259A/) |
| 5B   | H100 80G SXM | TE BF16 | 256    | 1    |256    |1    | 8       |  1066.67  |   1.67   |       N/A        |   |
| 175B | H100 80G SXM | TE BF16 | 256    |1    |256    |1    | 6       |  44.01  |   30.35   |       N/A        |    |
| 5B   | H100 80G SXM | TE FP8 | 256    | 1    |256    |1    | 8       |  1288.05   |     1.38       |       N/A        |    [log](https://tensorboard.dev/experiment/i5kiGeQpRRapswa68RkYHQ/)      |
| 175B | H100 80G SXM | TE FP8 | 256    |1    |256    |1    | 6       |   65.64   |     20.33      |       N/A        |   [log](https://tensorboard.dev/experiment/HvpU324wQYarwgvd9P3Uew/)     |


*Note*: Estimated walltime is computed assuming full throughput continuously. In practice, true walltime may be greater due to compilation overheads, interleaved evaluation, and checkpointing. A number of the linked convergence runs were completed using older software; thus, throughput reported in the linked logs may not match current results. The most up-to-date throughput numbers are reported in the table. 

5B FP8 was trained for 75,000 steps at a global batch size of 2048 and a sequence length of 2048, amounting to around 300 billion consumed tokens. 175B FP8 was trained for a total of around 1,000 steps at a global batch size of 1536 and a sequence length of 2048, amounting to around 3.14 billion consumed tokens. 175B was trained using the [C4 dataset](https://github.com/mlcommons/training/tree/master/large_language_model/paxml#2-dataset) and restores from an [initial MLPerf checkpoint](https://github.com/mlcommons/training/tree/master/large_language_model/paxml#initial-checkpoint). 126M and 5B were both trained using the Pile.

### Running an Experiment with Base Configs
To run an experiment with any base model configuration with the default parallel strategy reported in the table, copy [run_pile_multinode.sh](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/run_pile_multinode.sh) to your workspace and make the following modifications: replace `--fdl_config=paxml.contrib.gpu.scripts_gpu.configs.Pile126M` with the experiment you are interested in running (e.g. `paxml.contrib.gpu.scripts_gpu.configs.GPT5B` or `paxml.contrib.gpu.scripts_gpu.configs.GPT175B`) and remove `--fdl.ICI_MESH_SHAPE="[${NUM_GPUS}, 1, 1]"` and `--fdl.DCN_MESH_SHAPE="[${SLURM_JOB_NUM_NODES}, 1, 1]"`. The resulting bash script (call it `run_my_model_multinode.sh`) can be passed into `example_slurm_pile.sub` using the following command. This command presumes that `run_my_model_multinode.sh` lives in `BASE_WORKSPACE_DIR`.
```
BASE_SCRIPT=run_my_model_multinode.sh CONTAINER=<CONTAINER> BASE_WORKSPACE_DIR=<PATH_TO_WORKSPACE> BASE_TFDS_DATA_DIR=<PATH_TO_THE_PILE> BASE_VOCAB_PATH=<PATH_TO_SENTENCEPIECE_MODEL> LOG_DIR_LOCAL=<LOG_DIR_LOCAL> OUTPUT_DIR=<OUTPUT_DIR_LOCAL> PREC=<PRECISION> GPUS_PER_NODE=<GPUS_PER_NODE> PERCORE_BATCH_SIZE=<BS_PER_GPU> ENABLE_FP8=<ENABLE_FP8> sbatch -N <NUM_NODES> -A <ACCOUNT> -p <PARTITION> -J <JOBNAME> scripts/example_slurm_pile.sub
```
Here, it is assumed that you are running with the number of nodes reported in the table. If using a different node count, scale `DCN_MESH_SHAPE` accordingly. For example, the default value of `DCN_MESH_SHAPE` for `paxml.contrib.gpu.scripts_gpu.configs.GPT5B` is `[1,32,1]`. If running on 16 nodes, adjust `DCN_MESH_SHAPE` as follows:
```
--fdl.DCN_MESH_SHAPE=[1,16,1]
```


## Known Issues
* Pipeline parallelism is not supported with NVIDIA Transformer Engine enabled in the Paxml container.
* The Paxml nightlies disable `NCCL_NVLS_ENABLE=0` ([doc](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nvls-enable)). Future releases will re-enable this feature.
* The release container has a known XLA bug which affects single-process training in some cases. This bug has been fixed in newer XLA versions. If running into issues with single-process training, try using a Pax nightly container after 10/3. You can also try cherry-picking [this commit](https://github.com/openxla/xla/commit/aa8e7340cb319b9419a097155874bf105da05e1d) in the tested container.  
* Infrequent hangs have been observed in multinode settings. Setting `CUDA_MODULE_LOADING=EAGER` helps with these hangs. This environment variable is set by default in `nvcr.io/nvidia/jax:23.10-paxml-py3`.
* We currently see unexpected convergence behavior when dropout is used with Transformer Engine. Default configs do not enable dropout within transformer layers and thus should be unaffected by this bug, but users may encounter this bug if manually enabling dropout in their models.


## Changelog
### 10/26/2023
- Enabled BF16 Transformer Engine by default
- Added FP8 Transformer Engine support
- Updated 5B config to disable dropout in transformer layers
- bfloat16 performance
    - 126M performance is 6% higher than 8/29, bringing the overall regression with respect to 7/11 to around 10%. We will continue to improve 126M performance in future releases.

### 8/29/2023
- Added bfloat16 Transformer Engine support
- Disabled packing by default in all base configurations for TE compatibility
- Updated 5B config to use fully sharded data parallel (FSDP)
- bfloat16 perf changes (no TE)
    - 15% regression - 126M (this will be fixed in the next release)
    - 3% speedup - 5B
    - 4.5% speedup - 175B

### 7/11/2023
- Updated 175B config. 175B now trained on 32 nodes using fully sharded data parallel (FSDP)
- A100 perf gains
    - 22% speedup - 126M
    - 6% speedup - 5B
