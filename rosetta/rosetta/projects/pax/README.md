# Pax
[Pax](https://github.com/google/paxml/tree/main) is a framework developed by Google optimized for running machine learning experiments using JAX. Pax consists of the Paxml and [Praxis](https://github.com/google/praxis/tree/main) repositories and is maintained as a [distribution](../../../docs/DEVELOPMENT.md) within Rosetta. This means that we cherry-pick the necessary changes to optimize Pax for GPUs on top of upstream Paxml and Praxis' `main` branches. We also provide support for FP8 training via both [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) and native [XLA-FP8](../../../docs/NATIVE_FP8.md).

Any `paxml/*` or `praxis/*` relative directory/file can be found in [google/paxml](https://github.com/google/paxml/tree/main) or [google/praxis](https://github.com/google/praxis/tree/main), respectively, but to
view the most up-to-date version of that directory/file with any GPU-specific patches, please see [Inspecting the Source Code](#inspecting-the-source-code).

# Hardware and Software Specifications
Convergence and performance has been validated on NVIDIA DGX H100 (8x H100 80G) and A100 (8x A100 80G) nodes; for details, please refer to the [Configs](#configs) section below. We provide both singlenode and multinode pre-training support. If running on a machine with less than 80G memory, some of the default configurations may run out of memory; if you run out of memory and have more GPUs available, increase your GPU count and decrease your batch size per GPU.

The [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) is required to run the subsequent commands with GPU support. Ensure the NVIDIA Container Toolkit is installed before proceeding.

# Containers
We provide a fully built and ready-to-use multi-arch container which includes the latest optimizations, experimental features, and examples benchmarked for multi-node, multi-GPU training: `nvcr.io/nvidia/jax:24.04-paxml-py3` (amd64 and arm64 support). This container also provides FP8 support via [Transformer Engine](https://github.com/NVIDIA/TransformerEngine). Verified containers will be updated periodically, but if you wish to use the bleeding edge (which may come with unexpected behavior), please use `ghcr.io/nvidia/pax:latest`. We also provide nightly dated images with the naming pattern `ghcr.io/nvidia/pax:nightly-YYYY-MM-DD`, but we encourage you to use the latest ones for the best performance.

For more information on the Pax build and for details on how to manually build the Pax distribution, please refer to [DEVELOPMENT.md](../../../docs/DEVELOPMENT.md).

*Note*: All paths mentioned in subsequent sections are relative to the top-level directory of the Paxml repository. When working interactively with containers, make sure you navigate to `/opt/paxml` before running any commmands.

# Downloading the SentencePiece Model
Pax models require a pretrained SentencePiece model to tokenize the datasets. The SentencePiece model used in the following experiments is `gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model`. This model was trained using [these instructions](https://github.com/sgpyc/training/blob/paxml-llm-draft/large_language_model/paxml/utils/generate_spm.md). Use the following commands to download the tokenizer locally. This should be done _prior_ to launching the container.
```
wget -P c4_sentencepiece https://github.com/nvjax-svc-0/assets/raw/main/sentencepiece_c4/c4_en_301_5Mexp2_spm.model
```
You can then use the following mount to attach the tokenizer to your container:
```
docker run -v ${PWD}/c4_sentencepiece/c4_en_301_5Mexp2_spm.model:/opt/paxml/vocab ...
```

# Launching a container
Use the following command to launch a container:
```
docker run -ti --gpus=all --net=host --ipc=host -v <DATASET_PATH>:/opt/paxml/datasets -v <WORKSPACE_PATH>:/opt/paxml/workspace -v <VOCAB_PATH>:/opt/paxml/vocab -w /opt/paxml <CONTAINER> /bin/bash
```
where `DATASET_PATH` is the path to the Pile or Lambada dataset. If these datasets have not yet been downloaded, they can be downloaded from inside of the container (see [Downloading The Pile and Lambada Datasets](#Downloading-the-pile-and-lambada-datasets) for more). `WORKSPACE_PATH` is the path to the directory where you would like to store any persistent files, and `VOCAB_PATH` is the path to the pretrained SentencePiece model to use during tokenization (see [Downloading the SentencePiece Model](#Downloading-the-sentencepiece-model) for more).

# Downloading The Pile and Lambada Datasets
__IMPORTANT UPDATE__: Please be aware that as of October 2023, 'the_pile' dataset is no longer accessible. The team is actively updating our instructions and configurations to incorporate a more recent large language model (LLM) dataset. Additionally, we have provided updated instructions that include methods for using synthetic data, ensuring that our users can continue their work without interruption. Please see the [synthetic dataset](#Synthetic-dataset) section below for more information.

The GPT model configs we provide are trained using The Pile dataset and evaluated using the Lambada dataset. The scripts [download_the_pile.py](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/download_the_pile.py) and [download_lambada.py](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/download_lambada.py) will download The Pile and Lambada datasets to the `TFDS_DATA_DIR` enviroment variable. To control the location of the downloaded datasets, use the following command prior to running the download scripts: `export TFDS_DATA_DIR=<path_to_dowload_data_to>`. For example, the following commands download the Pile dataset to `/opt/paxml/datasets/`:
```
export TFDS_DATA_DIR=/opt/paxml/datasets/
python3 paxml/contrib/gpu/scripts_gpu/download_the_pile.py
```

After the data has been successfully downloaded, use the same `TFDS_DATA_DIR` when running experiments.

# Inspecting the Source Code
If you would like to inspect Pax's source code (`paxml/*` and `praxis/*`) to learn more about what is being run, you can do so by inspecting
the source within the container. Here are some examples:

```bash
# (Interactive = already in container): navigate to paxml/contrib/gpu/scripts_gpu/
cd $(python -c 'import paxml; print(paxml.__path__[0])')/../paxml/contrib/gpu/scripts_gpu

# (Non-interactive): View paxml/contrib/gpu/scripts_gpu/configs.py
FILE=paxml/contrib/gpu/scripts_gpu/configs.py
docker run --entrypoint="" --rm <CONTAINER> sh -c 'cat $(python -c "import paxml; print(*paxml.__path__)" 2>/dev/null)/../'$FILE
```

# Running a Job
Note that when training with The Pile dataset, you must provide the `TFDS_DATA_DIR` as a command-line argument and a `VOCAB_PATH` (the path to a pretrained SentencePiece model) as an environment variable. See the bash scripts below for examples.

## Quick Runs
### Interactive: Single Node
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

### Multi Node
The [scripts](scripts) directory provides a number of example submit files for launching the provided models on SLURM+pyxis cluster. For example, [example_slurm_pile.sub](scripts/example_slurm_pile.sub) launches an 8-node training run with a 126 million parameter GPT model.

To launch `example_slurm_pile.sub`, run the following command:
```
CONTAINER=<CONTAINER> BASE_WORKSPACE_DIR=<PATH_TO_WORKSPACE> BASE_TFDS_DATA_DIR=<PATH_TO_THE_PILE> BASE_VOCAB_PATH=<PATH_TO_SENTENCEPIECE_MODEL> LOG_DIR_LOCAL=<LOG_DIR_LOCAL> OUTPUT_DIR=<OUTPUT_DIR_LOCAL> PREC=bfloat16 GPUS_PER_NODE=8 PERCORE_BATCH_SIZE=4 ENABLE_FP8=<ENABLE_FP8> sbatch -N 8 -A <ACCOUNT> -p <PARTITION> -J <JOBNAME> scripts/example_slurm_pile.sub
```
where `BASE_WORKSPACE_DIR`, `BASE_TFDS_DATA_DIR`, and `BASE_VOCAB_PATH` are absolute paths and `LOG_DIR` and `OUTPUT_DIR` are relative to `BASE_WORKSPACE_DIR`.

Details on the other `.sub` files are provided in the [Configs](#configs) section.

## Customized Runs
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

## Transformer Engine
Training using Transformer Engine (TE) with bfloat16 precision is controlled via the environment variable `ENABLE_TE`. TE is enabled by default in the prebuilt container, but if you would like to disable TE, you can do so by flipping the value of `ENABLE_TE` in the container:
```
export ENABLE_TE=0
```

FP8 training is controlled via the `ENABLE_FP8` environment variable. To enable FP8 training, set `ENABLE_FP8=1`. For example, the following command trains a 126M model on a single node using FP8:
```
ENABLE_FP8=1 bash paxml/contrib/gpu/scripts_gpu/run_pile_singlenode.sh /opt/paxml/datasets /opt/paxml/vocab bfloat16 8 4 log_dir
```
Note that transformer engine must be enabled (`ENABLE_TE=1`) in order to train with FP8 using TE). Also, note that packing is currently not supported when using TE. All configs disable packing by default, but beware that if packing is manually enabled, training with TE will error.

## Native FP8
Rosetta Pax containers also provide support for native FP8 through XLA. Enabling FP8 can be done by adding the following command-line flag to your bash script: `--fdl.USE_FP8=True`. When using native FP8, TE must be disabled. For a detailed explanation of native FP8 support in Pax, as well as a comparison between native FP8 and TE FP8, please refer to the [NATIVE_FP8](../../../docs/NATIVE_FP8.md) documentation.

## Flash Attention
Flash attention is enabled by default in the given container. Divergence has been observed with the GPT 126M model with flash attention enabled. If you observe divergence when running GPT 126M, it is recommended to disable flash attention. If training with Transformer Engine, you can disable FA using the following environment variable: `NVTE_FUSED_ATTN=0`. If not using TE, FA can be disabled using the following XLA flag: `--set_xla_gpu_enable_cudnn_fmha=False`.

In addition to improving throughput, enabling flash attention provides a memory savings. Some of the given configurations may run out of memory if flash attention is disabled; if this is the case, try reducing your microbatch size and, if possible, increasing your GPU count.

## XLA Flags
The [GPU Performance document](../../../docs/GPU_performance.md) provides a detailed description of the XLA flags that can be set to optimize performance. Additionally, the scripts in `paxml/contrib/gpu/scripts_gpu` automatically set the suggested flags for each model. Please refer to these scripts to find the XLA flags used to reproduce the results documented below.

For the the 126M model, we recommend setting `--xla_gpu_all_reduce_combine_threshold_bytes=33554432`, which is different from the value recommended in `paxml/contrib/gpu/scripts_gpu/run_pile_multinode.sh`. To overwrite the default XLA flags set in the script, set the `BASE_XLA_FLAGS` environment variable prior to running `run_pile_multinode` as follows:

```
BASE_XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false
                --xla_gpu_simplify_all_fp_conversions --xla_gpu_enable_async_all_gather=true
                --xla_gpu_enable_async_reduce_scatter=true  --xla_gpu_enable_highest_priority_async_stream=true
                --xla_gpu_enable_triton_softmax_fusion=false  --xla_gpu_all_reduce_combine_threshold_bytes=33554432
                --xla_gpu_graph_level=0 --xla_gpu_enable_async_all_reduce=true" bash run_pile_multinode.sh ...
```

# Configs
## GPT
We provide three "base" GPT model configurations in `paxml/contrib/gpu/scripts_gpu/configs.py`. The first is a 126 million parameter GPT model. Convergence using The Pile dataset has been verified with this model. The remaining configs are 5 billion and 175 billion parameter models. Both 5B and 175B are provided primarily for benchmarking purposes and been less thoroughly tested for convergence.

The tables below describe current performance of the given configs. Experiments were run using NVIDIA DGX A100 80G and H100 80G nodes. Note that Lambada accuracy reported corresponds to the best accuracy seen across the run. Estimated walltime denotes the aproximate time to train each model to completion (i.e. number of days to reach `MAX_STEPS` number of steps as described in `configs.py`).

### A100 Results

| Size | GPU | Precision | #GPUs | DP | FSDP | TP | BS / GPU | Sequences/Sec | Est. Walltime (days) | Lambada Accuracy (± standard deviation) |
| ---- | ----- |----- |----- | -- | ---- | -- | ---------| ---------------| ------------------------- | ---------------- |
| 126M | A100 80G SXM | BF16 |  64    |64    |1    |1    | 4     |   2098.16  |        0.85        |   39.7% (± 1.2%)     |
| 5B   | A100 80G SXM | BF16 | 256    | 1    |256    |1    | 8       | 594.13    |    2.99    |       N/A        |
| 175B | A100 80G SXM | BF16 | 256    |1    |256    |1    | 6       |   *    |      *     |        N/A       |
| 126M | A100 80G SXM | TE BF16 |  64    |64    |1    |1    | 4     |  2526.72   |     0.70            |   N/A |
| 5B   | A100 80G SXM | TE BF16 | 256    | 1    |256    |1    | 8       | 718.19    |    2.48    |       N/A        |
| 175B | A100 80G SXM | TE BF16 | 256    |1    |256    |1    | 6       |  20.44   |      65.24     |        N/A       |

\* will be updated once final results have been gathered

### H100 Results

| Size | GPU | Precision | #GPUs | DP | FSDP | TP | BS / GPU | Sequences/Sec | Est. Walltime (days) | Lambada Accuracy (± standard deviation) |
| ---- | ----- |----- |----- | -- | ---- | -- | ---------| ---------------| ------------------------- | ---------------- |
| 126M | H100 80G SXM | TE BF16 |  64    |64    |1    |1    | 4        |  4709.12  |     0.38           |    42.5% (± 1.8%)        |
| 5B   | H100 80G SXM | TE BF16 | 256    | 1    |256    |1    | 8       |  1657.24  |   1.07   |       N/A        |
| 175B | H100 80G SXM | TE BF16 | 256    |1    |256    |1    | 6       |  51.00  |   26.15   |       N/A        |
| 5B   | H100 80G SXM | TE FP8 | 256    | 1    |256    |1    | 8       |  2374.66   |     0.749       |       N/A        |
| 175B | H100 80G SXM | TE FP8 | 256    |1    |256    |1    | 6       |   84.45   |     15.79      |       N/A        |


*Note*: Estimated walltime is computed assuming full throughput continuously. In practice, true walltime may be greater due to compilation overheads, interleaved evaluation, and checkpointing. 126M performance numbers were gathered _without_ flash attention (due to known convergence issues with flash attention, see [Known Issues](#Known-issues) for more). The other model sizes enable flash attention.

5B FP8 was trained for 75,000 steps at a global batch size of 2048 and a sequence length of 2048, amounting to around 300 billion consumed tokens. 175B FP8 was trained for a total of around 1,000 steps at a global batch size of 1536 and a sequence length of 2048, amounting to around 3.14 billion consumed tokens. 175B was trained using the [C4 dataset](https://github.com/mlcommons/training/tree/master/large_language_model/paxml#2-dataset) and restores from an [initial MLPerf checkpoint](https://github.com/mlcommons/training/tree/master/large_language_model/paxml#initial-checkpoint). 126M and 5B were both trained using the Pile.

## Mixture of Experts
We provide configs for two GLaM models. GLaM is a class of mixture of experts models with every other transformer layer replaced with a MoE layer with top-2 routing. The model sizes we provide are 126M/64E (126M base dense model, 64 experts, ~1.9B parameters) and 64B/64E (~1.14T parameters). Convergence has been validated on 126M/64E. Convergence results are outlined below.

| Model      | Num. params | Precision | #GPUs | DP | FSDP | TP | BS / GPU | Sequence length | Lambada Accuracy (fixed compute) | Lambada Accuracy (fixed steps) |
| ---------  |------------ | --------- | ----- | -- | ---- | -- | -------- | -------------- | ------ | ------ |
| 126M/64E   | 1.9B        |  BF16     | 64    | 1  | 64   | 1  |   8      | 2048           | 46.15% | 49.21% |
| 64B/64E    | 1.14T       |  BF16     | 512   | 1  | 64   | 8  |   4      | 2048           | N/A    | N/A    |

"Fixed compute" refers to the lambada accuracy given the same compute budget as GPT 126M dense (measured on H100), and "fixed steps" refers to the lambada accuracy given the same number of training steps as 126M dense.

The script `paxml/contrib/gpu/scripts_gpu/run_base_config_multinode.sh` can be used to run these GLaM configurations. See the [Running an Experiment with Base Configs](#Running-an-experiment-with-base-configs) section for more information about how to lauch a slurm job using this script.

_Note_: The GLaM configs provided currently do not have support for Transformer Engine. We are actively working on this and will update the configs as TE support becomes available.

## LLaMA
We also provide LLaMA-2 7B, 13B and 70B configs. These configs are variants of the [LLaMA configs](https://github.com/google/saxml/blob/main/saxml/server/pax/lm/params/lm_cloud.py) provided by Saxml and have been validated on the [BoolQ](https://github.com/google-research-datasets/boolean-questions) dataset. The table below reports BoolQ zero-shot accuracy for each model.

### Zero-shot Accuracy

| Size | Precision | #GPUs | DP | FSDP | TP | BS / GPU | BoolQ Accuracy |
| ---- |---------- | ----- | -- | ---- | -- | -------- | -------------- |
| 7B   | BF16      | 8     | 1  | 8    | 1  | 16       | 77.52%         |
| 13B  | BF16      | 8     | 1  | 8    | 1  | 8        | 82.99%         |
| 70B  | BF16      | 16    | 1  | 16   | 1  | 4        | 85.08%         |

### Fine-tuning
LLaMA fine-tuning is supported via full supervised fine-tuning (SFT) and LoRA parameter-efficient fine-tuning. Performance and convergence has been tested on LLaMA-2 7B, and results are reported below.

#### SFT Results
| Size | GPU | Precision | #GPUs | DP | FSDP | TP | BS / GPU | Sequence Length | Sequences/Sec | BoolQ Accuracy  (± standard deviation) |
| ---- | ----- |----- |----- | -- | ---- | -- | ---------| ---------------| ------------------------- | ---------------- |
| 7B | H100 80G SXM | BF16 |  16    | 1    |16    |1    | 2        |  4096 |  43.24 |      88.7%  (± 0.12%)      |
| 7B | H100 80G SXM | TE BF16 |  16    |1    |16    |1    | 2        |  4096 | 53.69  |      88.2% (± 0.17%)       |

#### LoRA Results

Default LoRA parameters for all runs:
- LORA_RANK = 32
- LORA_TARGET_LAYERS = all
- TRAIN_STEPS = 600

| Size | GPU | Precision | #GPUs | DP | FSDP | TP | BS / GPU | Sequence Length | Total Sequences | Sequences/Sec | BoolQ Accuracy  (± standard deviation) |
| ---- | ----- |----- |----- | -- | ---- | -- | ---------| ---------------| ---------------|  ------------------------- | ---------------- |
| 7B | H100 80G SXM | TE BF16 |  16    |1     |16    |1    | 2        |  4096 | 19,200 | 63.2 |     88.8933 (± 0.146) %          |
| 7B | H100 80G SXM | TE BF16 |  16    |1    |16    |1    | 1        |  4096 | 9,600 | 56  |     88.52 (± 0.198) %          |
| 7B | H100 80G SXM | BF16 |  16    |1    |16    |1    | 2        |  4096 | 19,200 | 43.8  |     88.57 (± 0.2275) %          |

### Running LLaMA Evaluation/Fine-tuning

Saxml provides a [script](https://github.com/google/saxml/blob/f3efdafed400d03be22efdb39a006f1420460d9f/saxml/tools/convert_llama_ckpt.py) to convert Meta's LLaMA checkpoints to Paxml format for zero-shot evaluation and fine-tuning. This script can be run inside of any JAX-Toolbox pax container. First, apply for access and download the Meta checkpoints and LLaMA tokenizer using [this link](https://llama.meta.com/llama-downloads/). Then, mount the Meta checkpoints to the container and run the following commands to convert the checkpoint:
```
pip install pytorch ## loading meta checkpoints requires pytorch
wget https://raw.githubusercontent.com/google/saxml/f3efdafed400d03be22efdb39a006f1420460d9f/saxml/tools/convert_llama_ckpt.py
python3 -m convert_llama_ckpt --base-model-path <meta checkpoint path> --pax-model-path <path to save checkpoint to> --model-size <7b, 13b, or 70b>
```

If you'd like to run LLaMA with transformer engine, the [Pax <--> TE checkpoint converter](../../../utils/te_pax_t5x_ckpt_converter) can be used to produce a TE-compatible checkpoint using the following command:
```
python  converter/main.py \
    --input-path=/your_path_to_src_ckpt \
    --output-path=/your_path_to_output_ckpt \
    --fw=pax \
    --direction=fw2te \
    --num-of-layer=<NUM_LAYERS> \
    --num-of-head=<NUM_HEADS> \
    --head-dim=<DIMS_PER_HEAD> \
    --mlp-intermediate-dim=<MLP_DIM> \
    --skip-bias \
    --weight-only \
    --use-gated-activations
```
if converting the 70B checkpoint, the following additional arguments should be passed to the converter:
```
    --num-gqa-groups=8 \
    --pax-split-qkv \
    --te-qkv-layout=kv_packed
```
Please refer to the checkpoint converter [readme](../../../utils/te_pax_t5x_ckpt_converter#readme) for more detailed instructions.

The script [download_boolq.py](https://github.com/google/paxml/blob/main/paxml/contrib/gpu/scripts_gpu/download_boolq.py) downloads the BoolQ dataset to the `TFDS_DATA_DIR` (see [Downloading the Pile and Lambada Datasets](#Downloading-the-pile-and-lambada-datasets) for more). Once BoolQ has been downloaded, the script [example_slurm_llama.sub](scripts/example_slurm_llama.sub) can be used to reproduce the results reported in the tables. The script calls `paxml/contrib/gpu/scripts_gpu/run_llama_boolq.sh`, which is configured to run the 7B model by default. Please inspect `run_llama_boolq.sh` in your container to see the arguments that can be overwritten if interested in running other model sizes. Launch `example_slurm_llama.sub` using the following command:

```
CONTAINER=<CONTAINER> BASE_WORKSPACE_DIR=<PATH_TO_WORKSPACE> BASE_TFDS_DATA_DIR=<PATH_TO_BOOLQ> BASE_VOCAB_PATH=<PATH_TO_LLAMA_TOKENIZER> OUTPUT_DIR=<OUTPUT_DIR_LOCAL> EVAL_ONLY=<EVAL_ONLY> USE_LORA=<USE_LORA> BASE_CHECKPOINT_RESTORE_PATH=<PATH_TO_PRETRAINED_CHECKPOINT> LOG_DIR_LOCAL=<DIR_TO_STORE_NEW_CHECKPOINTS_AND_LOGS> CONFIG=<LLaMA_CONFIG> ENABLE_TE=<ENABLE_TE> sbatch -N <NUM_NODES> -A <ACCOUNT> -p <PARTITION> -J <JOBNAME> scripts/example_slurm_llama.sub
```
`CONFIG` should be one of `LLaMA7B`, `LLaMA13B`, or `LLaMA70B`. `EVAL_ONLY` is a boolean indicating whether to run zero-shot evaluation (`EVAL_ONLY=1`) or fine-tuning. `CHECKPOINT_RESTORE_PATH` refers to the path to the pretrained checkpoint to restore from. The pretrained checkpoint is expected to have the following directory structure: `<FT_CHECKPOINT_DIR>/checkpoints/checkpoint_<STEP>`. In order for the checkpoint restore to work correctly, `CHECKPOINT_RESTORE_PATH` should be `<FT_CHECKPOINT_DIR>`.

The same script can also be used to fine tune LLaMA models using [LoRA](https://arxiv.org/abs/2106.09685). The environment variables that configure LoRA are specified below:
- USE_LORA: Specifies whether LoRA will be used for finetuning. Default value is 0. Set to 1 if you want to enable LoRA.
- LORA_RANK: Rank used for the LoRA weight matrices. Default value is 32.
- LORA_TARGET_LAYERS: Specifies which layers to target for LoRA. Default value is 'all' which targets all linear layers. Acceptable values are "all", "attention", "mlp" where "all" targets all linear layers; "attention" targets q, k, v and out projection; "mlp" targets all MLP layers.

For example, the following command will run LoRA fine-tuning on the LLaMA-2 7B model:

```
CONTAINER=<CONTAINER> BASE_WORKSPACE_DIR=$PWD BASE_TFDS_DATA_DIR=<PATH_TO_BOOLQ> BASE_VOCAB_PATH=<PATH_TO_LLAMA_TOKENIZER> OUTPUT_DIR=lora_stdout EVAL_ONLY=0 USE_LORA=1 BASE_CHECKPOINT_RESTORE_PATH=<PATH_TO_PRETRAINED_CHECKPOINT> LOG_DIR_LOCAL=7b_log_dir CONFIG=LLaMA7B ENABLE_TE=1 sbatch -N 2 -A <ACCOUNT> -p <PARTITION> -J <JOBNAME> scripts/example_slurm_llama.sub
```

_Note_: The given LLaMA configs currently do not support FP8 training via Transformer Engine. We are actively working on this and will update the configs as TE support becomes available.

## Running an Experiment with Base Configs
The `run_base_config_multinode.sh` script is provided to run any of the base configs provided in `paxml/contrib/gpu/scripts_gpu/configs.py` out of the box. [scripts/launch_base_script.sub](scripts/launch_base_script.sub) uses this script to train a model on a slurm cluster. Launch this script using the following command:
```
CONTAINER=<CONTAINER> CONFIG=<CONFIG_NAME> BASE_WORKSPACE_DIR=<PATH_TO_WORKSPACE> BASE_TFDS_DATA_DIR=<PATH_TO_THE_PILE> BASE_VOCAB_PATH=<PATH_TO_SENTENCEPIECE_MODEL> LOG_DIR_LOCAL=<LOG_DIR_LOCAL> OUTPUT_DIR=<OUTPUT_DIR_LOCAL> PREC=<PRECISION> GPUS_PER_NODE=<GPUS_PER_NODE> ENABLE_TE=<ENABLE_TE> ENABLE_FP8=<ENABLE_FP8> sbatch -N <NUM_NODES> -A <ACCOUNT> -p <PARTITION> -J <JOBNAME> scripts/launch_base_script.sub
```
where `CONFIG` is the name of the config from `paxml/contrib/gpu/scripts_gpu/configs.py`. Here, it is assumed that you are running with the number of nodes reported in the table. If using a different node count, scale `DCN_MESH_SHAPE` accordingly. For example, the default value of `DCN_MESH_SHAPE` for `GPT5B` is `[1,32,1]`. If running on 16 nodes, adjust `DCN_MESH_SHAPE` in your bash script as follows:
```
--fdl.DCN_MESH_SHAPE=[1,16,1]
```

### Synthetic Dataset
We also provide GPT 126M, 5B and 175B configurations with a dummy dataset for quick benchmarking. The script `run_base_config_multinode.sh` can also be used to benchmark any of the given base models using the synthetic dataset. [scripts/launch_base_script.sub](scripts/launch_base_script.sub) can be used to launch this script on a slurm cluster. When training using a dummy dataset, it is not required to pass in a `BASE_VOCAB_PATH` or `TFDS_DATA_DIR`:

```
BASE_WORKSPACE_DIR=<PATH_TO_WORKSPACE> CONFIG=Synthetic<126M, 5B, 175B> OUTPUT_DIR=<OUTPUT_DIR> PREC=bfloat16 ENABLE_TE=<ENABLE_TE> ENABLE_FP8=<ENABLE_FP8> LOG_DIR_LOCAL=<LOG_DIR> sbatch -N <NODES> -A <ACCOUNT> -p <PARTITION> -J <JOBNAME> -t <TIME_LIMIT> scripts/launch_base_script.sub
```

For example, the following command benchmarks the 5B model on 32 nodes with TE BF16 using the synthetic dataset:
```
BASE_WORKSPACE_DIR=<PATH_TO_WORKSPACE> CONFIG=Synthetic5B OUTPUT_DIR=output_synthetic_5b PREC=bfloat16 ENABLE_TE=1 ENABLE_FP8=0 LOG_DIR_LOCAL=log_dir_synthetic_5b sbatch -N 32 -A <ACCOUNT> -p <PARTITION> -J <JOBNAME> scripts/launch_base_config.sub
```
Note that with models that are particularly dataloading-bottlenecked (e.g. smaller models, such as 126M), the throughput observed using the synthetic dataset may be higher than the throughput observed when training on a real dataset.

# Known Issues
* Divergence has been observed with the GPT 126M model with flash attention enabled. If you observe divergence when running GPT 126M, it is recommended to disable flash attention.
* There is a known bug with cudnn flash attention that can cause divergence when using flash attention _without_ TE. We recommend running all models with TE enabled, but if you would like to disable TE, and you observe unexpected divergence, try disabling flash attention using the following XLA flag: `--xla_gpu_enable_cudnn_fmha=false`
* TE is currently not supported with GLaM models. Future releases will include TE support with GLaM.
* The provided LLaMA configs do not support TE FP8 for fine-tuning. Future releases will add FP8 support.
* The Paxml containers disable `NCCL_NVLS_ENABLE=0` ([doc](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nvls-enable)). Future releases will re-enable this feature.
* LoRA without TE is currently not supported for models using `CombinedQKVProjection` where `input_dim != num_heads * dims_per_head`. Fix for this issue will be available in the nightlies soon.

# Changelog
## 4/26/2024
- Added support for LLaMA SFT and LoRA fine-tuning (BF16 and TE BF16)
- Added support for MoE models: GLaM 126M and GLaM 64B (BF16)
- Enabled TE flash attention by default

## 10/26/2023
- Enabled BF16 Transformer Engine by default
- Added FP8 Transformer Engine support
- Updated 5B config to disable dropout in transformer layers
- bfloat16 performance
    - 126M performance is 6% higher than 8/29, bringing the overall regression with respect to 7/11 to around 10%. We will continue to improve 126M performance in future releases.

## 8/29/2023
- Added bfloat16 Transformer Engine support
- Disabled packing by default in all base configurations for TE compatibility
- Updated 5B config to use fully sharded data parallel (FSDP)
- bfloat16 perf changes (no TE)
    - 15% regression - 126M (this will be fixed in the next release)
    - 3% speedup - 5B
    - 4.5% speedup - 175B

## 7/11/2023
- Updated 175B config. 175B now trained on 32 nodes using fully sharded data parallel (FSDP)
- A100 perf gains
    - 22% speedup - 126M
    - 6% speedup - 5B
