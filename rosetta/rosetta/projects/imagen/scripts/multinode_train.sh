#! /bin/bash
# A script for single-node pile pretraining

# Copyright (c) 2022-2023 NVIDIA Corporation
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
set -x

# Arguments
DATASET=$1
MODEL_TYPE=$2       # Model size (small, base, large)
PREC="$3"        # Precision (float32, float16, bfloat16)
NUM_GPUS=$4      # Number of GPUs (1, 2, 4, 8)
BSIZE_PER_GPU=$5 # Batch size per GPU (varies with model size)
MODEL_DIR_LOCAL=${6:-"runs/model_dir"}
MODEL_DIR=${ROSETTA_DIR}/${MODEL_DIR_LOCAL}
INF_SERV_CT=$7
HOSTNAMES_FILE=${8:-""}
NUM_MICROBATCHES=${9:-0}

STAT_PERIOD=500

echo $MODEL_DIR

echo "Please make sure ${NUM_GPUS} is the number of visible CUDA devices you have"

# Setting XLA flags
source ${ROSETTA_DIR}/rosetta/projects/diffusion/common/set_gpu_xla_flags.sh
echo "XLA_FLAGS: ${XLA_FLAGS}"

# Global batch size
TRAIN_GPUS=$(( NUM_GPUS * SLURM_JOB_NUM_NODES - INF_SERV_CT ))
BSIZE=$(( BSIZE_PER_GPU * (NUM_GPUS * SLURM_JOB_NUM_NODES - INF_SERV_CT) ))
PROC_CT=$((SLURM_NTASKS - INF_SERV_CT))
#--gin.train/mm_utils.WebDatasetConfig.batch_size=${BSIZE} \

echo CUDA_DEVICE_MAX_CONNECTIONS
echo $CUDA_DEVICE_MAX_CONNECTIONS
unset CUDA_DEVICE_MAX_CONNECTIONS
export CUDA_MODULE_LOADING=EAGER

CUDA_MODULE_LOADING=EAGER python3 -u -m t5x.train \
  --gin_file="${ROSETTA_DIR}/rosetta/projects/imagen/configs/${MODEL_TYPE}_${DATASET}.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.DTYPE=\"${PREC}\" \
  --gin.BATCH_SIZE=${BSIZE} \
  --gin.train.stats_period=${STAT_PERIOD} \
  --gin.trainer.Trainer.num_microbatches=${NUM_MICROBATCHES} \
  --gin.HOSTNAMES_FILE=\"${HOSTNAMES_FILE}\" \
  --gin.INFER_BS=$(( TRAIN_GPUS * 4 )) \
  --multiprocess_gpu \
  --coordinator_address=${SLURM_LAUNCH_NODE_IPADDR}:12345 \
  --process_count=${PROC_CT} \
  --process_index=${SLURM_PROCID} 

set +x
