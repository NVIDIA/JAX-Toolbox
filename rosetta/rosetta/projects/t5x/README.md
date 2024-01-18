# T5x

[T5x](https://github.com/google-research/t5x) is a project developed by Google, which is maintained as a [distribution](../../../docs/DEVELOPMENT.md) within rosetta.

Any `t5x/*` relative directory/file can be found in [google-research/t5x](https://github.com/google-research/t5x), but to
view the most up to date version of that directory/file, please see ["Inspecting the source code"](#inspecting-the-source-code)

## Hardware Specifications
Convergence and performance has been validated on NVIDIA DGX A100 and H100 nodes; for details, please refer to the [Convergence and performance](#Convergence-and-performance) section below. We provide both singlenode and multinode support for pre-training and fine-tuning. If running on a machine with less than 80G memory, some of the default configurations may run out of memory; in such instances, gradient accumulation can be used to reduce the memory requirement.

## GPU Scripts and Usage
The `t5x/contrib/gpu/scripts_gpu` directory contains scripts optimized for GPU usage and includes FP8 support via [Transformer Engine](https://github.com/NVIDIA/TransformerEngine).


## Prerequisites
The examples below will reuse these environment variables. Feel free to change them:
```bash
CONTAINER=ghcr.io/nvidia/t5x:latest
DATASET_PATH=<NEED TO SPECIFY>
WORKSPACE_PATH=""  # Path used for run outputs (unspecified = /t5x_home/workspace)
```

## Container
We provide the latest fully built, ready-to-use, and verified container here: `ghcr.io/nvidia/t5x:latest-verified`. The verified containers will be updated
periodically, but if you wish to use the bleeding edge (which may come with unexpected behavior), please use `ghcr.io/nvidia/t5x:latest`.
We also provide nightly dated images with the naming pattern [ghcr.io/nvidia/t5x:nightly-YYYY-MM-DD](https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/t5x), but we encourage
you to use the latest ones to get the best performance.

We **highly** recommend using the pre-built container, but if you'd like to build your own container, you can follow the instructions here: [Building rosetta manually](../../../README.md#building-rosetta-with-a-specific-base)

## Downloading The Pile
__IMPORTANT UPDATE__: Please be aware that as of October 2023, 'the_pile' dataset is no longer accessible. The team is actively updating our instructions and configurations to incorporate a more recent large language model (LLM) dataset.

We use The Pile for our pretraining experiments. If you would like to as well, run `download_the_pile.py` to download it. The download is approximately 1TB. It will download to the directory set in the environment variable: `TFDS_DATA_DIR`. After that, set the `TFDS_DATA_DIR` to the same directory in your scripts to use. Here is how you would run it:

```bash
docker run --rm -e TFDS_DATA_DIR=/t5x_home/datasets -v ${DATASET_PATH}:/t5x_home/datasets $CONTAINER python -m t5x.contrib.gpu.scripts_gpu.download_the_pile
```

## Running interactively
**Note**: this should only be done with singlenode jobs and/or for downloading The Pile. 

```bash
docker run --rm --gpus=all -it --net=host --ipc=host -v ${PWD}:/t5x_home -v ${DATASET_PATH}:/t5x_home/datasets -v ${WORKSPACE_PATH:-${PWD}/workspace}:/t5x_home/workspace --privileged $CONTAINER bash
```

## Inspecting the source code
If you would like to inspect t5x's source code (`t5x/*`) to learn more about what is being run, you can do so by inspecting
the source within the container. Here are some examples:

```bash
# (Interactive = already in container): navigate to t5x/contrib/gpu
cd $(python -c 'import t5x; print(*t5x.__path__)')/../t5x/contrib/gpu

# (Non-interactive): View t5x/contrib/gpu/Dockerfile
FILE=t5x/contrib/gpu/Dockerfile
docker run --entrypoint="" --rm $CONTAINER sh -c 'cat $(python -c "import t5x; print(*t5x.__path__)" 2>/dev/null)/../'$FILE
```

## Single Node runs
Pretraining and Finetuning can be done with `singlenode_*.sh`. These will build a T5X model with the Adam optimizer and relevant parameters. These will allow multi-gpu on one host.

**Note**: When running interactively, if you encounter shape mismatch issues, check the logs to see if the train script was restoring from a previous checkpoint. Some hyperparameters like batch size are saved in the dataloader state, which when restored may be incompatible with your current run.

```bash
# Pretraining (interactive: already inside container) with default args
bash t5x/contrib/gpu/scripts_gpu/singlenode_pretrain_pile.sh

# Pretraining (non-interactive)
docker run --rm --gpus=all --net=host --ipc=host -v ${DATASET_PATH}:/t5x_home/datasets $CONTAINER bash t5x/contrib/gpu/scripts_gpu/singlenode_pretrain_pile.sh

# Finetuning (interactive: already inside container) with default args
bash t5x/contrib/gpu/scripts_gpu/singlenode_ft_frompile.sh

# Finetuning (non-interactive)
docker run --rm --gpus=all --net=host --ipc=host -v ${DATASET_PATH}:/t5x_home/datasets $CONTAINER bash t5x/contrib/gpu/scripts_gpu/singlenode_ft_frompile.sh
```

## Multi Node runs
For a SLURM+pyxis cluster, [`example*.sub`](./scripts) files provide example slurm submit files, which are configurable via environment variables and command line args. The submit fules call `multiprocess*.sh` to execute training. You can add a binding script in the `.sub` file for your cluster, or remove it entirely (dropping some throughput).

## Convergence and performance
For our Pile convergence runs, we used a Global batch size of 2304 for XXL and 2016-2048 for all other models, where GBS is defined as #GPUs * BS/GPU / Tensor Parallel(TP). Below are example (tested) hardware topologies on NVIDIA DGX A100 (8x A100-SXM4-80G) and H100-SXM-80G nodes.

| size          | GPU              | Precision | #GPUs |  TP   | BS / GPU | Sequences/Sec | Seq/Sec/GPU | Est. Walltime | GPU-days | MNLI 2.0 - matched | SQuAD v1.1 (EM/F1) | Convergence Log                                                                              | Config | 
| ----          | ------------     | --------- | ----- | ----- | -------- | ------------- | ----------- | ------------- | -------- |------------------ | ------------------  | ---------------                                                                              | ----   |
| T5-v1.1-small | A100 80G SXM     | bf16      | 8     | 1     | 256      | ~5712         | 714         | 4.2 days      | 33       | 83.06%             | 78.33 / 86.63      | [log](https://tensorboard.dev/experiment/lWnHal7PRnOLeZuewyWVxQ/#scalars&_smoothingWeight=0) | `t5x/contrib/gpu/t5/t5_1_1/examples/small_pile_pretrain.gin` |
| T5-v1.1-large | A100 80G SXM     | bf16      | 64    | 1     | 32       | ~4853         | 75.8        | 4.8 days      | 309      | 89.23%             | 86.12 / 93.21      | [log](https://tensorboard.dev/experiment/aOxJBIvTQBeTJ8XGXxaL6Q/#scalars&_smoothingWeight=0) | `t5x/contrib/gpu/t5/t5_1_1/examples/large_pile_pretrain.gin` |
| T5-v1.1-xl    | A100 80G SXM     | bf16      | 144   | 1     | 8        | ~3021         | 21.0        | 7.9 days      | 1,133    | N/A(perf test)     | N/A (perf test)    |                                                                                              | `t5x/contrib/gpu/t5/t5_1_1/examples/xl_pile_pretrain.gin` |
| T5-v1.1-xl    | A100 80G SXM     | bf16      | 256   | 1     | 8        | ~4322         | 16.9        | 5.5 days      | 1,408    | 91.15%             | 89.36 / 95.29      | [log](https://tensorboard.dev/experiment/vuRoEYgkRgWiEtbvgxlOqw/#scalars&_smoothingWeight=0) | `t5x/contrib/gpu/t5/t5_1_1/examples/xl_pile_pretrain.gin` |
| T5-v1.1-xxl   | A100 80G SXM     | bf16      | 512   | 8     | 36       | ~1887         | 3.69        | 12.6 days     | 6,431    | N/A(partial run)   | N/A(partial run)   |                                                                                              | `t5x/contrib/gpu/t5/t5_1_1/examples/xxl_pile_pretrain.gin` |
| T5-v1.1-large | **H100 80G SXM** | TE-fp8    | 64    | 1     | 32       | ~11139        | **174.1**   | **2.1 days**  | **134**  | 89.1%              | 86.36 / 93.5       | [log](https://tensorboard.dev/experiment/QJYnDaaBSeuZtYPXXtAG3Q/#scalars&_smoothingWeight=0) | `t5x/contrib/gpu/t5/t5_1_1/examples/large_pile_pretrain.gin` |
| T5-v1.1-xl    | **H100 80G SXM** | TE-fp8    | 144   | 1     | 14       | ~7257         | **50.4**    | **3.3 days**  | **475**  | N/A (perf test)    | N/A (perf test)    |                                                                                              | `t5x/contrib/gpu/t5/t5_1_1/examples/xl_pile_pretrain.gin` |
| T5-v1.1-xl    | **H100 80G SXM** | TE-fp8    | 256   | 1     | 8        | ~9688         | **37.8**    | **2.4 days**  | **614**  | N/A (perf test)    | N/A (perf test)    |                                                                                              | `t5x/contrib/gpu/t5/t5_1_1/examples/xl_pile_pretrain.gin` |

Note: Convergence (as shown in log) was not necessarily done with the hardware topology listed, but the listed topology is tested. Estimated Walltime is calculated assuming full throughput (seq/sec) continuously. In practice, there are compilation overheads at the beginning of each run/restart (in cluster settings) + checkpointing overheads (if any).

Other hyperparameters are specified in the associated pile `gin` files in the `t5x/contrib/gpu/t5/t5_1_1/examples` directory.

## Pretraining run commands
All slurm commands below assume you have cloned this repo to make the submit scripts available locally:
```bash
git clone https://github.com/NVIDIA/JAX-Toolbox.git
cd JAX-Toolbox/rosetta/rosetta/projects/t5x/scripts
```

### Multinode
Arguments are set by environment variable as such:

```sh
PREC={PRECISION} T5_SIZE={SIZE} BSIZE_PER_GPU={BSIZE} ..... \
  sbatch -N {NODE_CT} scripts/example_slurm_pretrain_pile.sub {GPUS_PER_NODE}
```

All parameters can be found in the relevant script.

### Example Pretraining Commands
Assumes 8GPU 80GB A100/H100 Nodes. `ENABLE_FP8` uses transformer engine (included in container) and requires H100

* Note: To use, FP8 set `ENABLE_FP8` to `1`. This will automatically set `PREC` to `bfloat16` as is required by internals for `FP8` usage.
#### T5-v1.1-small (60M):
```sh
PREC=bfloat16 T5_SIZE=small BSIZE_PER_GPU=256 TRAIN_STEPS=1000000 NUM_MICROBATCHES=1 ENABLE_FP8=1 TP_SIZE=1 \
sbatch -N1 scripts/example_slurm_pretrain_pile.sub
```

#### T5-v1.1-large (770M):
```sh
PREC=bfloat16 T5_SIZE=large BSIZE_PER_GPU=32 TRAIN_STEPS=1000000 NUM_MICROBATCHES=1 ENABLE_FP8=1 TP_SIZE=1 \
sbatch -N8 scripts/example_slurm_pretrain_pile.sub
```

#### T5-v1.1-xl (3B):
```sh
PREC=bfloat16 T5_SIZE=large BSIZE_PER_GPU=8 TRAIN_STEPS=1000000 NUM_MICROBATCHES=1 ENABLE_FP8=1 TP_SIZE=1 \
sbatch -N 32 scripts/example_slurm_pretrain_pile.sub
```

### Example Finetuning Commands
Finetuning commands simply change the script and have an additional `{FT_TASK}` as the first argument (along with relevant hyperparameter changes). Your `MODEL_DIR` should contain the pretrained checkpoint to restore from.

#### MNLI v2:
```sh
FT_TASK=mnli2 PREC=bfloat16 T5_SIZE={SIZE} BSIZE_PER_GPU={BSIZE} NUM_MICROBATCHES=1 ENABLE_FP8=1 TP_SIZE=1 \
sbatch -N{NODE_CT} scripts/example_slurm_ft_frompile.sub
```

#### SQuAD v1.1:
```sh
FT_TASK=squad1 PREC=bfloat16 T5_SIZE={SIZE} BSIZE_PER_GPU={BSIZE} NUM_MICROBATCHES=1 ENABLE_FP8=1 TP_SIZE=1 \
sbatch -N{NODE_CT} scripts/example_slurm_ft_frompile.sub

```

## Performance Settings:
There are 3 major performance settings: `ENABLE_FP8`, `FUSE_QKV` and `TRANSPOSE_BS` (all of which are controllable via env var in the commands above).
We recommend always enabling `TRANSPOSE_BS` (default), but only using `FUSE_QKV` when using `ENABLE_FP8` for optimal performance.

On all finetuning runs, we use a Global Batch Size of 256 with bfloat16 precision + FP8.

WARNING: Finetuning is configured by default to save every checkpoint and delete none (to avoid accidentally deleting your pretrained checkpoint). Watch your disk space! This behavior can be changed in `t5x/configs/runs/finetune_{TASK}.gin`, however this puts the pretrained checkpoint at risk unless backed up.

### Singlenode (single process)
small:

```sh
t5x/contrib/gpu/scripts_gpu/singlenode_pretrain_pile.sh \
  small \
  bfloat16 \
  8 \
  256 \
  {LOGDIR - create before running} \
  {MODEL_DIR} \
  {GRADIENT_ACCUMULATION (1 by default)} \
  {ENABLE_FP8 (1 by default)} \
  {TRANSPOSE_BS (1 by default)} \
  {FUSE_QKV (1 by default)} \
  {PACK (0 by default)}
```

Finetuning:
MNLI v2:
```sh
t5x/contrib/gpu/scripts_gpu/singlenode_ft_frompile.sh \
  mnli2 \
  small \
  bfloat16 \
  8 \
  256 \
  {LOGDIR - create before running} \
  {MODEL_DIR(to restore pretrained checkpoint from)} \
  {GRADIENT_ACCUMULATION (1 by default)} \
  {MAKE_FT_DIR (false by default)}
  {ENABLE_FP8 (1 by default)} \
  {TRANSPOSE_BS (1 by default)} \
  {FUSE_QKV (1 by default)} \
  {PACK (0 by default)}
```

# Known Issues
* There is a known sporadic NCCL crash that happens when using the T5x container at node counts greater than or equal to 32 nodes. We will fix this in the next release. The issue is tracked [here](https://github.com/NVIDIA/JAX-Toolbox/issues/194).

# Changelog
- Added Transformer Engine + FP8 support
- Updated T5x and JAX=0.4.11
- A100 Perf gains! (BF16)
  - 80% speedup - T5-small
  - 23% speedup - T5-large
  - 18% speedup - T5-xl
  - 40% speedup - T5-xxl 
- H100 FP8 support, with gains over A100
  - 2.08x faster - T5-large (FP8)
  - 2.24x faster - T5-xl (FP8)
