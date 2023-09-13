# Pax
[Pax](https://github.com/google/paxml/tree/main) is a framework developed by Google optimized for running machine learning experiments using JAX. Pax consists of the Paxml and [Praxis](https://github.com/google/praxis/tree/main) repositories. Pax is maintained as a [distribution](../../../docs/DEVELOPMENT.md) within rosetta. This means that we cherry-pick the necessary changes to optimize Pax for GPUs on top of upstream Paxml and Praxis' `main` branches. 

Any `paxml/*` or `praxis/*` relative directory/file can be found in [google/paxml](https://github.com/google/paxml/tree/main) or [google/praxis](https://github.com/google/praxis/tree/main), respectively, but to
view the most up-to-date version of that directory/file with any GPU-specific patches, please see [Inspecting the source code](#inspecting-the-source-code).

## Hardware Specifications
Convergence and performance has been validated on NVIDIA DGX A100 (8x A100 80G) nodes; for details, please refer to the [Configs](#configs) section below.. We provide both singlenode and multinode pre-training support. If running on a machine with less than 80G memory, some of the default configurations may run out of memory; if you run out of memory and have more GPUs available, increase your GPU count and decrease your batch size per GPU.

## Containers
We provide a fully built and ready-to-use container which includes the latest optimizations, experimental features, and examples benchmarked for multi-node, multi-GPU training: `nvcr.io/nvidia/jax:23.08-paxml-py3`. This container also provides bfloat16 [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) support.

For more information on the Pax build and for details on how to manually build the Pax distribution, please refer to [DEVELOPMENT.md](../../../docs/DEVELOPMENT.md). 

*Note*: All paths mentioned in subsequent sections are relative to the top-level directory of the Paxml repository. When working interactively with containers, make sure you are in `/opt/paxml` before running any commmands.

### Launching a container
Use the following command to launch a container:
```
docker run -ti --gpus=all --net=host --ipc=host -v <DATASET_PATH>:/opt/paxml/datasets -v <WORKSPACE_PATH>:/opt/paxml/workspace -v <VOCAB_PATH>:/opt/paxml/vocab -w /opt/paxml <CONTAINER> /bin/bash
```
where `DATASET_PATH` is the path to the Pile or Lambada dataset. If these datasets have not yet been downloaded, they can be downloaded inside of the container (see [Downloading The Pile and Lambada Datasets](#Downloading-the-pile-and-lambada-datasets) for more). `WORKSPACE_PATH` is the path to the directory where you would like to store any persistent files, and `VOCAB_PATH` is the path to the pretrained sentencepiece model to use during tokenization (see [Downloading the SentencePiece Model](#Downloading-the-sentencepiece-model) for more). 

## Downloading The Pile and Lambada Datasets
The given models are trained using The Pile dataset and evaluated using the Lambada dataset. The scripts [download_the_pile.py](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/download_the_pile.py) and [download_lambada.py](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/download_lambada.py) will download The Pile and the Lambada datasets to the `TFDS_DATA_DIR` enviroment variable. To control the location of the downloaded datasets, use the following command prior to running the download scripts: `export TFDS_DATA_DIR=<path_to_dowload_data_to>`. After the data has been successfully downloaded, use the same `TFDS_DATA_DIR` when running experiments.

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

# (Non-interactive): View paxml/contrib/gpu/scripts_gpu/configs.py
FILE=paxml/contrib/gpu/scripts_gpu/configs.py
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
See [example_slurm_pile.sub](https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/rosetta/projects/pax/scripts/example_slurm_pile.sub) for an example slurm submit file that launches an 8-node run with a 126 million parameter GPT model.

To launch `example_slurm_pile.sub`, run the following command:
```
CONTAINER=<CONTAINER> BASE_WORKSPACE_DIR=<PATH_TO_WORKSPACE> BASE_TFDS_DATA_DIR=<PATH_TO_THE_PILE> BASE_VOCAB_PATH=<PATH_TO_SENTENCEPIECE_MODEL> LOG_DIR_LOCAL=<LOG_DIR_LOCAL> OUTPUT_DIR=<OUTPUT_DIR_LOCAL> PREC=bfloat16 GPUS_PER_NODE=8 PERCORE_BATCH_SIZE=4 sbatch -N 8 -A <ACCOUNT> -p <PARTITION> -J <JOBNAME> paxml/contrib/gpu/scripts_gpu/example_slurm_pile.sub
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

We provide three "base" configurations in `paxml/contrib/gpu/scripts_gpu/configs.py`. For more information about these configurations and how to run experiments using them, please refer to the [Configs](#Configs) section below.

### Transformer Engine
Training using Transformer Engine (TE) with bfloat16 precision can be enabled via the environment variable `ENABLE_TE`. To enable TE, simply add the following line to `run_pile_multinode.sh` (or whatever bash script you are using to run experiments):
```
export ENABLE_TE=1
```
Note that packing is currently not supported when using TE. All configs disable packing by default, but beware that if packing is manually enabled, training with TE will error. 

## XLA Flags
We recommend setting the following XLA flags when running experiments: 

1. `--xla_gpu_simplify_all_fp_conversions`: Allows all floating-point `f32 -> bf16 -> f32` conversion pairs to be simplified.
2. `--xla_gpu_enable_latency_hiding_scheduler=true`: Allows XLA:GPU to move communication collectives to increase overlap with compute kernels
3. `--xla_gpu_enable_async_all_gather=true`: Allows XLA:GPU to run All Gather NCCL kernels on a separate CUDA stream to allow overlap with compute kernels
4. `--xla_gpu_enable_async_reduce_scatter=true`: Allows XLA:GPU to run Reduce Scatter NCCL kernels on a separate CUDA stream to allow overlap with compute kernels
5. `--xla_gpu_enable_async_all_reduce=true`: Allows XLA:GPU to run All Reduce NCCL kernels on a separate CUDA stream to allow overlap with compute kernels.
6. `--xla_gpu_enable_highest_priority_async_stream=true`: Allows XLA to prioritize the launch of NCCL kernels before GeMMs to ensure enough SMs are available for async communication kernels.
7. `--xla_gpu_all_reduce_combine_threshold_bytes=51200`: Combines NCCL All Reduce kernels until threshold size is reached.
8. `--xla_gpu_enable_triton_gemm=false`: Disallows Triton GeMM kernels; uses CUBLAS GeMM kernels instead. CUBLAS kernels are currently better tuned for GPUs and thus provide better performance
9. `--xla_gpu_cuda_graph_level=0`: Disallows XLA from using CUDA graphs.

These flags are enabled by default in `paxml/contrib/gpu/scripts_gpu/run_pile_multinode.sh`. The `XLA_FLAGS` environment variable controls these flags; to configure XLA flags explicitly, you can use the following command.
```
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true --xla_gpu_enable_triton_gemm=false"
```

## Configs
We provide three "base" model configurations in `paxml/contrib/gpu/scripts_gpu/configs.py`. The first is a 126 million parameter GPT model. Convergence using The Pile dataset has been verified with this model. The remaining configs are 5 billion and 175 billion parameter models. Both 5B and 175B are provided for benchmarking purposes and have not been thoroughly tested for convergence to date.

The table below describes current performance of the given configs. Experiments were run using NVIDIA DGX A100 (8x A100 80G) nodes. Note that Lambada accuracy reported corresponds to the best accuracy seen across the run. Estimated walltime denotes the aproximate time to train each model to completion (i.e. number of days to reach `MAX_STEPS` number of steps as described in `configs.py`).

| Size | #GPUs | DP | FSDP | TP | BS / GPU | Sequences/Sec (bf16 / TE bf16) | Estimated Walltime (days, bf16 / TE bf16) | Lambada Accuracy | Convergence Log |
| ---- | ----- | -- | ---- | -- | ---------| ---------------| ------------------------- | ---------------- |---------------- |
| 126M |  64    |64    |1    |1    | 4        |  1761.3 / 2339.8   |         1.01 / 0.76             |        0.397 (± 0.012)     | [log](https://tensorboard.dev/experiment/RCroDLAUQzGUoudzqD1NmQ/) |
| 5B   | 256    | 1    |256    |1    | 8       |  465.45 / 598.83     |      3.82 /  2.97          |       N/A        | [log](https://tensorboard.dev/experiment/AyXAn8ZDRheUARN1NMJ1sw)           |
| 175B | 256    |1    |256    |1    | 6       |   18.29 / 19.62      |        72.92 / 67.97           |    N/A           | [log](https://tensorboard.dev/experiment/NJnv5LbdQby2PcZGPnTRrA/)  | N/A           |

*Note*: Estimated walltime is computed assuming full throughput continuously. In practice, true walltime may be greater due to compilation overheads, interleaved evaluation, and checkpointing. A number of the linked convergence runs were completed using older software; thus, reported throughput does not match current results (notably for 126M and 5B bf16). The most up-to-date throughput numbers are reported in the table. 

The runs in 5B convergence log were trained for around 26k (TE) and 45k (no TE) steps at a global batch size of 2048 and a sequence length of 2048, amounting to around 109 billion and 189 billion consumed tokens for TE, non-TE respectively. The 175B convergence log was trained for a total of around 700 steps at a global batch size of 1536 and a sequence length of 2048, amounting to around 2.2 billion consumed tokens. Finally, 175B was trained using the [C4 dataset](https://github.com/mlcommons/training/tree/master/large_language_model/paxml#2-dataset), while 126M and 5B were both trained using the Pile.

### Running an Experiment with Base Configs
To run an experiment with any base model configuration with the default parallel strategy reported in the table, copy [run_pile_multinode.sh](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/run_pile_multinode.sh) to your workspace and make the following modifications: replace `--fdl_config=paxml.contrib.gpu.scripts_gpu.configs.Pile126M` with the experiment you are interested in running (e.g. `paxml.contrib.gpu.scripts_gpu.configs.GPT5B` or `paxml.contrib.gpu.scripts_gpu.configs.GPT175B`) and remove `--fdl.ICI_MESH_SHAPE="[${TRAIN_GPUS}, 1, 1]"`. The resulting bash script (call it `run_my_model_multinode.sh`) can be passed into `example_slurm_pile.sub` using the following command. This command presumes that `run_my_model_multinode.sh` lives in `BASE_WORKSPACE_DIR`.
```
BASE_SCRIPT=run_my_model_multinode.sh CONTAINER=<CONTAINER> BASE_WORKSPACE_DIR=<PATH_TO_WORKSPACE> BASE_TFDS_DATA_DIR=<PATH_TO_THE_PILE> BASE_VOCAB_PATH=<PATH_TO_SENTENCEPIECE_MODEL> LOG_DIR_LOCAL=<LOG_DIR_LOCAL> OUTPUT_DIR=<OUTPUT_DIR_LOCAL> PREC=<PRECISION> GPUS_PER_NODE=<GPUS_PER_NODE> PERCORE_BATCH_SIZE=<BS_PER_GPU> sbatch -N <NUM_NODES> -A <ACCOUNT> -p <PARTITION> -J <JOBNAME> paxml/contrib/gpu/scripts_gpu/example_slurm_pile.sub
```

## Known Issues
* The Paxml container does not fully support Hopper yet. Future releases will add Hopper support.
* Pipeline parallelism is not supported with NVIDIA Transformer Engine enabled in the Paxml container.
* There are known Common Vulnerabilities and Exposures (CVE) that affect the Paxml container related to TensorFlow 2.9.x due to pinning TensorFlow to 2.9.x in Paxml and Lingvo. We will fix these in the next release. The known CVEs are:
    * CVE-2023-25668 
    * CVE-2023-25658
    * CVE-2023-25663
    * CVE-2023-25664
    * CVE-2023-25664
    * CVE-2023-25672
    * CVE-2023-25674
    * CVE-2023-25660
    * CVE-2023-27579
    * CVE-2023-25671
    * CVE-2023-25659
    * CVE-2023-25662
    * CVE-2023-25675
    * CVE-2023-25801
    * CVE-2023-25670
    * CVE-2023-25669
    * CVE-2023-25665
    * CVE-2023-25673
    * CVE-2023-25666


## Changelog
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
