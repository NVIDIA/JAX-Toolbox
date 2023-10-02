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

T5X_DIR=${PWD}

# Arguments
DATASET=$1
T5_SIZE=$2       # Model size (small, base, large)
PREC="$3"        # Precision (float32, float16, bfloat16)
NUM_GPUS=$4      # Number of GPUs in node (1, 2, 4, 8)
BSIZE_PER_GPU=$5 # Batch size per GPU (varies with model size)
LOG_DIR=$6       # Output log directory
MODEL_DIR_LOCAL=${7:-"model_dir"}
MODEL_DIR=${PWD}/${MODEL_DIR_LOCAL}
INF_SERV_CT=${8:-0}
NUM_MICROBATCHES=${9:-0}
HOSTNAMES_FILE=${10:-""}

STAT_PERIOD=100

mkdir -p $LOG_DIR
echo $MODEL_DIR

TRAIN_GPUS=$((NUM_GPUS - INF_SERV_CT))
echo "Please make sure ${NUM_GPUS} is the number of visible CUDA devices you have"

# Setting XLA flags
export XLA_FLAGS='--xla_gpu_simplify_all_fp_conversions --xla_gpu_all_reduce_combine_threshold_bytes=136314880'

# Global batch size
BSIZE=$(( TRAIN_GPUS * BSIZE_PER_GPU  ))


for i in $(seq 0 $(expr $TRAIN_GPUS - 1))
do
	CUDA_VISIBLE_DEVICES=$i \
	python3 -u ${T5X_DIR}/jax_multimodal/diffusion/train.py \
	  --gin_file="/t5x_home/jax_multimodal/diffusion/configs/${T5_SIZE}_${DATASET}.gin" \
	  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
	  --gin.network.DiffusionConfig.dtype=\"${PREC}\" \
	  --gin.BATCH_SIZE=${BSIZE} \
    --gin.train.stats_period=${STAT_PERIOD} \
	  --gin.trainer.Trainer.num_microbatches=${NUM_MICROBATCHES} \
    --gin.HOSTNAMES_FILE=\"${HOSTNAMES_FILE}\" \
    --multiprocess_gpu \
    --coordinator_address=127.0.0.1:12345 \
    --process_count=$TRAIN_GPUS \
    --process_index=$i &> ${LOG_DIR}/${T5_SIZE}_${DATASET}_gpu_${TRAIN_GPUS}_${PREC}_gbs_${BSIZE}-${i}.log &
done
