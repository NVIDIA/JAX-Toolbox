# Pax
[Pax](https://github.com/google/paxml/tree/main) is a framework developed by Google optimized for running machine learning experiments using JAX. Pax consists of the Paxml and [Praxis](https://github.com/google/praxis/tree/main) repositories. Pax is maintained as a [distribution](../../../docs/DEVELOPMENT.md) within rosetta. This means that we cherry-pick the necessary changes to optimize Pax for GPUs on top of upstream Paxml and Praxis' `main` branches. 

Any `paxml/*` or `praxis/*` relative directory/file can be found in [google/paxml](https://github.com/google/paxml/tree/main) or [google/praxis](https://github.com/google/praxis/tree/main), respectively, but to
view the most up-to-date version of that directory/file with any GPU-specific patches, please see [Inspecting the source code](#inspecting-the-source-code).

## Containers
We provide a fully built and ready-to-use container which includes the latest optimizations, experimental features, and examples benchmarked for multi-node, multi-GPU training: `ghcr.io/nvidia/pax:pax-2023-06-23.2`. This container contains clones of the Paxml and Praxis repositories from 06/23/2023.

For more information on the Pax build and for details on how to manually build the Pax distribution, please refer to [DEVELOPMENT.md](../../../docs/DEVELOPMENT.md). 

*Note*: All paths mentioned in subsequent sections are relative to the top-level directory of the Paxml repository. When working interactively with containers, make sure you are in `/opt/paxml` before running any commmands.

### Launching a container
Use the following command to launch a container:
```
docker run -ti --gpus=all --net=host --ipc=host -v <DATASET_PATH>:/opt/paxml/datasets -v <WORKSPACE_PATH>:/opt/paxml/workspace -v <VOCAB_PATH>:/opt/paxml/vocab -w /opt/paxml <CONTAINER> /bin/bash
```
where `DATASET_PATH` is the path to the Pile or Lambada dataset. If these datasets have not yet been downloaded, they can be downloaded inside of the container (see [Downloading The Pile and Lambada Datasets](#Downloading-the-pile-and-lambada-datasets) for more). `WORKSPACE_PATH` is the path to the directory where you would like to store any persistent files, and `VOCAB_PATH` is the path to the pretrained sentencepiece model to use during tokenization (see [Downloading the SentencePiece Model](#Downloading-the-sentencepiece-model) for more). 

## Downloading The Pile and Lambada Datasets
All models are trained using The Pile dataset and evaluated using the Lambada dataset. The scripts [download_the_pile.py](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/download_the_pile.py) and [download_lambada.py](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/download_lambada.py) will download The Pile and the Lambada datasets to the `TFDS_DATA_DIR` enviroment variable. To control the location of the downloaded datasets, use the following command prior to running the download scripts: `export TFDS_DATA_DIR=<path_to_dowload_data_to>`. After the data has been successfully downloaded, use the same `TFDS_DATA_DIR` when running experiments.

## Downloading the SentencePiece Model
Pax models require a pretrained SentencePiece model to tokenize the datasets. The SentencePiece model used in the following experiments is `gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model`. This model was trained using [these instructions](https://github.com/sgpyc/training/blob/paxml-llm-draft/large_language_model/paxml/utils/generate_spm.md). See below for information on downloading `gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model` from Google Cloud. 

1. Ensure you have the [Google Clould SDK](https://cloud.google.com/sdk/docs/install) installed.
2. Log into the Cloud by using the following command: `gcloud auth login` and following the prompts. 
3. Once logged in, use the following command to download the vocab file to your current working directory: 
```
gsutil -m cp -r gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model .
```

_NOTE_: We are aware of an existing permissions limitation that prevents users from downloading the SentencePiece model using the above instructions. We are actively working to resolve this, but in the meantime, the SP model file can be downloaded as part of the full C4 dataset download [here](https://cloud.mlcommons.org/index.php/s/dataset_c4_spm). Note that accessing the SP model using this method will require downloading the full C4 dataset.

## Inspecting the source code
If you would like to inspect Pax's source code (`paxml/*` and `praxis/*`) to learn more about what is being run, you can do so by inspecting
the source within the container. Here are some examples:

```bash
# (Interactive = already in container): navigate to paxml/contrib/gpu/scripts_gpu/
cd $(python -c 'import paxml; print(paxml.__path__[0])')/../paxml/contrib/gpu/scripts_gpu

# (Non-interactive): View paxml/contrib/gpu/scripts_gpu/run_pile_singlenode.sh
FILE=paxml/contrib/gpu/scripts_gpu/run_pile_singlenode.sh
docker run --entrypoint="" --rm $CONTAINER sh -c 'cat $(python -c "import paxml; print(*paxml.__path__)" 2>/dev/null)/../'$FILE
```

## Running a Job
Note that when training with The Pile dataset, you must provide the `TFDS_DATA_DIR` as a command-line argument and a `VOCAB_PATH` (the path to a pretrained sentencepiece model) as an environment variable (see the bash scripts below for examples). 

### Quick Runs
#### Interactive: Single Node
See [run_pile_singlenode.sh](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/run_pile_singlenode.sh) for an example of training a 126m model on a single node using The Pile. Once inside of your container, this script can be run interactively using the following command:
``` 
bash paxml/contrib/gpu/scripts_gpu/run_pile_singlenode.sh <TFDS_DATA_DIR> <VOCAB_PATH> <PRECISION> <NUM_GPUS> <PERCORE_BATCH_SIZE> <LOGDIR>
```
where `TFDS_DATA_DIR` is the path to the downloaded datasets, `VOCAB_PATH` is the path to the pretrained SentencePiece `.model` file, and `LOGDIR` is the relative path of the directory to which to write checkpoints and logging information. `PERCORE_BATCH_SIZE` is the batch size per GPU _prior_ to sharding according to the parallel strategy. See [Customized Runs](#Customized-runs) for more information about this hyperparameter. 

See [run_lambada_singlenode.sh](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/run_lambada_singlenode.sh) for an example of running zero-shot evaluation on the 126m model using the Lambada dataset. Use the following command to run this script:
``` 
bash paxml/contrib/gpu/scripts_gpu/run_lambada_singlenode.sh <TFDS_DATA_DIR> <VOCAB_PATH> <PRECISION> <NUM_GPUS> <PERCORE_BATCH_SIZE> <LOGDIR>
```
`TFDS_DATA_DIR` should contain the path to the Lambada dataset and `LOGDIR` should match the `LOGDIR` from the pretraining run.

#### Multi Node
See [example_slurm_pile.sub](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/example_slurm_pile.sub) for an example slurm submit file that launches an 8-node run with a 126 million parameter GPT model.

To launch `example_slurm_pile.sub`, run the following command:
```
BASE_WORKSPACE_DIR=<PATH_TO_WORKSPACE> BASE_TFDS_DATA_DIR=<PATH_TO_THE_PILE> BASE_VOCAB_PATH=<PATH_TO_SENTENCEPIECE_MODEL> LOG_DIR_LOCAL=<LOG_DIR_LOCAL> OUTPUT_DIR=<OUTPUT_DIR_LOCAL> PREC=bfloat16 GPUS_PER_NODE=8 PERCORE_BATCH_SIZE=4 sbatch -N 8 -A <ACCOUNT> -p <PARTITION> -J <JOBNAME> paxml/contrib/gpu/scripts_gpu/example_slurm_pile.sub
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
- `ICI_MESH_SHAPE`: This refers to the parallelism strategy used on chips connected by a fast network (e.g. NVLink). `ICI_MESH_SHAPE` typically has 3 dimensions, `[data, fsdp, tensor]`, corresponding to data parallelism (DP), fully-sharded data parallelism (FSDP/ZeRO-3), and tensor parallelism (TP), respectively. To use pure data parallelism, you should set `ICI_MESH_SHAPE` to `[NUM_GPUS, 1, 1]`.
- `DCN_MESH_SHAPE`: This refers to the parallelism strategy for machines connected by a datacenter network. This is the generally parallel strategy used _across_ nodes.
- `PERCORE_BATCH_SIZE`: This is the batch size loaded by each worker _prior_ to sharding the data according to the parallel strategy. We should always have that `GLOBAL_BATCH_SIZE = PERCORE_BATCH_SIZE * NUM_GPUS`, regardless of the parallel strategy. Note that a consequence of this is that `PERCORE_BATCH_SIZE` will not always equal `MICROBATCH_SIZE`, particularly when using tensor parallelism (TP). If using 2-way TP, for example, `MICROBATCH_SIZE` will be twice the `PERCORE_BATCH_SIZE`. If using tensor or pipeline parallelism, `PERCORE_BATCH_SIZE` may be fractional. For example, when using 2-way TP, setting `PERCORE_BATCH_SIZE` to 0.5 will result in a microbatch size of `PERCORE_BATCH_SIZE * TP = 1`.
- `NUM_LAYERS`, `NUM_HEADS`, `MODEL_DIMS`, `HIDDEN_DIMS`: These are hyperparameters of the transformer model. `MODEL_DIMS` refers to the hidden dimension of the transformer, and `HIDDEN_DIMS` refers to the hidden dimension of the transformer feed-forward network. 

We provide three "base" configurations in [configs.py](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/configs.py). For more information about these configurations and how to run experiments using them, please refer to the [Configs](#Configs) section below.

## XLA flags
We recommend setting the following XLA flags when running experiments: 

1. `--xla_gpu_enable_latency_hiding_scheduler=true`: Allows XLA:GPU to move communication collectives to increase overlap with compute kernels
2. `--xla_gpu_enable_async_all_gather=true`: Allows XLA:GPU to run All Gather NCCL kernels on a separate CUDA stream to allow overlap with compute kernels
3. `--xla_gpu_enable_async_reduce_scatter=true`: Allows XLA:GPU to run Reduce Scatter NCCL kernels on a separate CUDA stream to allow overlap with compute kernels
4. `--xla_gpu_enable_triton_gemm=false`: Disallows Triton GeMM kernels; uses CUBLAS GeMM kernels instead. CUBLAS kernels are currently better tuned for GPUs and thus provide better performance
    
These flags are enabled by default in the pre-built container. The `XLA_FLAGS` environment variable controls these flags; to configure XLA flags explicitly, you can use the following command.
```
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true --xla_gpu_enable_triton_gemm=false"
```

## Configs
We provide three "base" model configurations in [configs.py](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/configs.py). The first is a 126 million parameter GPT model. Convergence using The Pile dataset has been verified with this model. The remaining configs are 5 billion and 175 billion parameter models. Both 5B and 175B are provided for benchmarking purposes and have not been thoroughly tested for convergence to date.

The table below describes current performance of the given configs. Experiments were run using NVIDIA DGX A100 (8x A100 80G) nodes. Note that Lambada accuracy reported corresponds to the best accuracy seen across the run.

| Size | #GPUs | DP | FSDP | TP | BS / GPU | Sequences/Sec | Estimated Walltime (days) | Lambada Accuracy | Convergence Log |
| ---- | ----- | -- | ---- | -- | ---------|---------------| ------------------------- | ---------------- |---------------- |
| 126M |  64    |64    |1    |1    | 4        |   2068.5      |   0.86                     |        0.397 (± 0.012)     | [log](https://tensorboard.dev/experiment/RCroDLAUQzGUoudzqD1NmQ/) |
| 5B   | 256    | 32    |4    |2    | 8       |    450.6       |     3.95                   |       N/A        | N/A             |
| 175B | 256    |1    |256    |1    | 6       |    17.5       |      76.15                 |    N/A           |  N/A           |

*Note*: Estimated walltime is computed assuming full throughput continuously. In practice, true walltime may be greater due to compilation overheads, interleaved evaluation, and checkpointing. Linked convergence logs were not necessarily done with the topology described in `configs.py` and may have different walltimes, but the configs provided are the most performant configs tested. The throughput for these performant configs is reported in the table above.

To run an experiment with any base model configuration with the default parallel strategy reported in the table, make the following modifications to [run_pile_multinode.sh](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/run_pile_multinode.sh): replace `--fdl_config=paxml.contrib.gpu.scripts_gpu.configs.Pile126M` with the experiment you are interested in running (e.g. `paxml.contrib.gpu.scripts_gpu.configs.GPT5B` or `paxml.contrib.gpu.scripts_gpu.configs.GPT175B`) and remove `--fdl.ICI_MESH_SHAPE="[${TRAIN_GPUS}, 1, 1]"`. The resulting bash script (call it `run_my_model_multinode.sh`) can be passed into `example_slurm_pile.sub` as follows:
```
BASE_SCRIPT=run_my_model_multinode.sh BASE_WORKSPACE_DIR=<PATH_TO_WORKSPACE> BASE_TFDS_DATA_DIR=<PATH_TO_THE_PILE> BASE_VOCAB_PATH=<PATH_TO_SENTENCEPIECE_MODEL> LOG_DIR_LOCAL=<LOG_DIR_LOCAL> OUTPUT_DIR=<OUTPUT_DIR_LOCAL> PREC=<PRECISION> GPUS_PER_NODE=<GPUS_PER_NODE> PERCORE_BATCH_SIZE=<BS_PER_GPU> sbatch -N <NUM_NODES> -A <ACCOUNT> -p <PARTITION> -J <JOBNAME> paxml/contrib/gpu/scripts_gpu/example_slurm_pile.sub
```

## Changelog
- Updated 175B config. 175B now trained on 32 nodes using fully sharded data parallel (FSDP)
- A100 perf gains
    - 22% speedup - 126M
    - 6% speedup - 5B
