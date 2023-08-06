#! /bin/bash

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
set -o pipefail

# Arguments
SIZE=$1          # # Model size (small, base)
PREC="$2"        # Precision (float32, float16, bfloat16)
NUM_GPUS=$3      # Number of GPUs (1, 2, 4, 8)
BSIZE_PER_GPU=$4 # Batch size per GPU (varies with model size)
LOG_DIR=$5
MODEL_DIR_LOCAL=${6:-"model_dir"}
MODEL_DIR=${PWD}/${MODEL_DIR_LOCAL}
TRAIN_IDX_DIR=${7:-""}
EVAL_IDX_DIR=${8:-""}
NUM_MICROBATCHES=${9:-1}
MP=${10:-1} # model parallel

STAT_PERIOD=50

echo $MODEL_DIR

TRAIN_GPUS=${NUM_GPUS}
echo "Please make sure ${TRAIN_GPUS} is the number of visible CUDA devices you have"

# Setting XLA flags
export XLA_FLAGS="--xla_gpu_simplify_all_fp_conversions --xla_gpu_all_reduce_combine_threshold_bytes=136314880 ${XLA_FLAGS}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Global batch size
BSIZE=$(( BSIZE_PER_GPU * TRAIN_GPUS * NUM_MICROBATCHES / MP ))

OUTPUT_LOG=${LOG_DIR}/pt_${SIZE}_gpu_${NUM_GPUS}_${PREC}_gbs_${BSIZE}_mp_${MP}.log
mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR
python3 -u -m t5x.train \
  --gin_file="rosetta/projects/vit/configs/${SIZE}_finetune_imagenet.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.BATCH_SIZE=${BSIZE} \
  --gin.train.stats_period=${STAT_PERIOD} \
  --gin.config.GoogleViTConfig.dtype=\"${PREC}\" \
  --gin.trainer.Trainer.num_microbatches=${NUM_MICROBATCHES} \
  --gin.partitioning.PjitPartitioner.num_partitions=${MP} \
  --gin.TRAIN_INDEX_DIR=\"${TRAIN_IDX_DIR}\" \
  --gin.EVAL_INDEX_DIR=\"${EVAL_IDX_DIR}\" \
  --gin_search_paths=/opt/rosetta \
  2>&1 | tee $OUTPUT_LOG

EXP_STATUS=$?

if [ $EXP_STATUS != 0 ]; then
  echo "Run failed"
else
  echo "Run succeeded!"
fi

echo Output written to $OUTPUT_LOG
