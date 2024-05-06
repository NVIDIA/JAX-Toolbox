#!/bin/bash
#SBATCH -A example              # slurm account
#SBATCH -p partition            # slurm partition name
#SBATCH -N 1                    # number of nodes
#SBATCH -t 00:20:00             # wall time
#SBATCH -J "maxtext:test"         # job name
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --ntasks-per-node=8     # n tasks per machine (one task per gpu)
#SBATCH --overcommit
#SBATCH --dependency=singleton  # only run one instance at a time

# coding=utf-8
# Copyright 2024 The JAX-Toolbox Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# File system and volume glue code
#-------------------------------------------------------------------------------
CONTAINER="${CONTAINER:-ghcr.io/nvidia/jax:maxtext-2024-04-15}"

# << CHANGE ! >>
BASE_WORKSPACE_DIR=${BASE_WORKSPACE_DIR} ## location to write logs and checkpoints to
BASE_TFDS_DATA_DIR=${BASE_TFDS_DATA_DIR}
BASE_VOCAB_PATH=${BASE_VOCAB_PATH}
BASE_CHECKPOINT_DIR=${BASE_CHECKPOINT_DIR}
MAXTEXT_DIR=${MAXTEXT_DIR:-/opt/maxtext}

# Default env variables for paths required by MaxText training scripts
WORKSPACE_DIR=/opt/maxtext/workspace
TFDS_DATA_DIR=/mnt/datasets
GPT_VOCAB_PATH=/mnt/vocab
CHECKPOINT_DIR=/opt/maxtext/workspace/llama-checkpoint

# Add the MaxText/JAX specific mounts
MOUNTS="--container-mounts=$BASE_WORKSPACE_DIR:$WORKSPACE_DIR,$BASE_VOCAB_PATH:$GPT_VOCAB_PATH,$BASE_TFDS_DATA_DIR:/$TFDS_DATA_DIR,$BASE_CHECKPOINT_DIR:$CHECKPOINT_DIR"

# export some necessary flags
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=1
export NCCL_IB_SL=1

# Set XLA Flags
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
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
                --xla_disable_hlo_passes=rematerialization
                --xla_gpu_enable_custom_fusions=false
                --xla_gpu_enable_address_computation_fusion=false"

# Make directories that may not exist
mkdir -p $BASE_WORKSPACE_DIR

EXPORTS="--export=ALL,WORKSPACE_DIR=${WORKSPACE_DIR},TFDS_DATA_DIR=${TFDS_DATA_DIR},GPT_VOCAB_PATH=${GPT_VOCAB_PATH},CHECKPOINT_DIR=${CHECKPOINT_DIR}"
#-------------------------------------------------------------------------------

OUTPUT_DIR=${OUTPUT_DIR:-"outputs"}
MODEL=${MODEL:-llama2-7b}
RUN_NAME=${RUN_NAME:-"demo"}

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& cd $MAXTEXT_DIR \
&& nvidia-smi \
&& python3 MaxText/train.py \
    MaxText/configs/base.yml \
    model_name=${MODEL} \
    per_device_batch_size=2 \
    steps=15 \
    scan_layers=false \
    remat_policy=minimal_flash \
    attention=cudnn_flash_te\
    max_target_length=4096 \
    use_iota_embed=true \
    logits_dot_in_fp32=false\
    enable_checkpointing=false \
    base_output_directory=local_train \
    dataset_path=local \
    dataset_type=synthetic \
    hardware=gpu_multiprocess \
    run_name=$RUN_NAME
EOF

# create run specific output directory for ease of analysis
mkdir -p "${BASE_WORKSPACE_DIR}/${OUTPUT_DIR}/${RUN_NAME}"

# redirect both stdout and stderr in the same file for ease of analysis
OUTFILE="${BASE_WORKSPACE_DIR}/${OUTPUT_DIR}/${RUN_NAME}/output-%j-%n-%t.txt"

echo $cmd
srun -o $OUTFILE -e $OUTFILE --container-image="$CONTAINER" $MOUNTS $EXPORTS bash -c "${cmd}"