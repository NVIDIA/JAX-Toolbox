#! /bin/bash
# A script for single-node pile pretraining to be used with specialize.py

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
NUM_GPUS=$4      # Number of GPUs in node (1, 2, 4, 8)
BSIZE_PER_GPU=$5 # Batch size per GPU (varies with model size)
LOG_DIR=$6       # Output log directory
MODEL_DIR_LOCAL=${7:-"runs/model_dir"}
MODEL_DIR=${PWD}/${MODEL_DIR_LOCAL}
INF_SERV_CT=${8:-0}
NUM_MICROBATCHES=${9:-0}
HOSTNAMES_FILE=${10:-""}

STAT_PERIOD=100

echo $MODEL_DIR

TRAIN_GPUS=$((NUM_GPUS - INF_SERV_CT))
echo "Please make sure ${NUM_GPUS} is the number of visible CUDA devices you have"

# Setting XLA flags
export XLA_FLAGS='--xla_gpu_simplify_all_fp_conversions --xla_gpu_all_reduce_combine_threshold_bytes=136314880'

# Global batch size
BSIZE=$(( TRAIN_GPUS * BSIZE_PER_GPU  ))
INFER_SAMPLES=${INFER_SAMPLES:=$BSIZE}


CUDA_VISIBLE_DEVICES=${PROC_ID} \
python3 -u /opt/t5x/t5x/train.py \
  --gin_file="/opt/rosetta/rosetta/projects/imagen/configs/${MODEL_TYPE}_${DATASET}.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.DTYPE=\"${PREC}\" \
  --gin.BATCH_SIZE=${BSIZE} \
  --gin.train.stats_period=${STAT_PERIOD} \
  --gin.trainer.Trainer.num_microbatches=${NUM_MICROBATCHES} \
  --gin.HOSTNAMES_FILE=\"${HOSTNAMES_FILE}\" \
  --gin.INFER_SAMPLES=${INFER_SAMPLES} \
  --multiprocess_gpu \
  --coordinator_address=127.0.0.1:12345 \
  --process_count=$TRAIN_GPUS \
  --process_index=${PROC_ID} &> ${LOG_DIR}/${MODEL_TYPE}_${DATASET}_gpu_${TRAIN_GPUS}_${PREC}_gbs_${BSIZE}-${PROC_ID}.log &
