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

T5X_DIR=${PWD}

# Arguments
SIZE=$1          # # Model size (small, base)
PREC="$2"        # Precision (float32, float16, bfloat16)
NUM_GPUS=$3      # Number of GPUs (1, 2, 4, 8)
BSIZE_PER_GPU=$4 # Batch size per GPU (varies with model size)
MODEL_DIR=${5:-"/opt/rosetta/model_dir"}
MODEL_DIR=${MODEL_DIR}
TRAIN_IDX_DIR=${6:-""}
EVAL_IDX_DIR=${7:-""}
NUM_MICROBATCHES=${8:-1}
MP=${9:-1}

STAT_PERIOD=50

echo $MODEL_DIR

TRAIN_GPUS=$(( NUM_GPUS * SLURM_JOB_NUM_NODES ))
echo "Please make sure ${TRAIN_GPUS} is the number of visible CUDA devices you have"

# Setting XLA flags
export XLA_FLAGS="--xla_gpu_simplify_all_fp_conversions --xla_gpu_all_reduce_combine_threshold_bytes=136314880 ${XLA_FLAGS}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Global batch size
BSIZE=$(( BSIZE_PER_GPU * TRAIN_GPUS * NUM_MICROBATCHES / MP ))

JAX_PROFILE_DIR="${WORKSPACE_DIR}/jax_trace"
PB_DIR="${WORKSPACE_DIR}/protobuf/"

mkdir -p ${PB_DIR}
rm -rf $MODEL_DIR
mkdir -p $MODEL_DIR

PB_NAME=${SIZE}_${PREC}_${BSIZE_PER_GPU}_${NUM_MICROBATCHES}_${MP}

case $PGLE_PROF in
  1)
    export XLA_FLAGS="--xla_gpu_enable_async_collective_permute=false --xla_gpu_enable_async_reduce_scatter=false --xla_gpu_enable_async_all_gather=false --xla_gpu_enable_async_all_to_all=false --xla_gpu_enable_async_all_reduce=false --xla_gpu_enable_pipelined_all_reduce=true ${XLA_FLAGS}"
    COLLECT_TRACE=1 \
    USE_JAX_PROFILER=1 \
    JAX_PROFILE_DIR="${JAX_PROFILE_DIR}"  \
    python3 -u -m t5x.train \
      --gin_file="rosetta/projects/vit/configs/${SIZE}_pretrain_imagenet.gin" \
      --gin.MODEL_DIR=\"${MODEL_DIR}\" \
      --gin.BATCH_SIZE=${BSIZE} \
      --gin.train.stats_period=${STAT_PERIOD} \
      --gin.config.GoogleViTConfig.dtype=\"${PREC}\" \
      --gin.trainer.Trainer.num_microbatches=${NUM_MICROBATCHES} \
      --gin.partitioning.PjitPartitioner.num_partitions=${MP} \
      --gin.utils.SaveCheckpointConfig.period=1000 \
      --multiprocess_gpu \
      --coordinator_address=${SLURM_LAUNCH_NODE_IPADDR}:12345 \
      --process_count=${SLURM_NTASKS} \
      --process_index=${SLURM_PROCID} \
      --gin.TRAIN_INDEX_DIR=\"${TRAIN_IDX_DIR}\" \
      --gin.EVAL_INDEX_DIR=\"${EVAL_IDX_DIR}\" \
      --gin_search_paths=/opt/rosetta \
      --gin.TRAIN_STEPS=20 \
      --gin.utils.CheckpointConfig.save=None
   
    ls -la ${ROSETTA_DIR}/jax_gen_pb.py 
    test ${SLURM_PROCID} -ne 0 && sleep 10
    test ${SLURM_PROCID} -eq 0 && python3 ${ROSETTA_DIR}/jax_gen_pb.py $JAX_PROFILE_DIR $PB_DIR $PB_NAME
	ls ${PB_DIR}/${PB_NAME}.pb

  ;;

  *)
    export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_collective_permute=true --xla_gpu_enable_async_reduce_scatter=true --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_all_to_all=true --xla_gpu_enable_async_all_reduce=true --xla_gpu_enable_pipelined_all_reduce=true ${XLA_FLAGS} --xla_gpu_pgle_profile_file_or_directory_path=${PB_DIR}/${PB_NAME}.pb"
    echo "XLA_FLAGS: ${XLA_FLAGS}"
    python3 -u -m t5x.train \
      --gin_file="rosetta/projects/vit/configs/${SIZE}_pretrain_imagenet.gin" \
      --gin.MODEL_DIR=\"${MODEL_DIR}\" \
      --gin.BATCH_SIZE=${BSIZE} \
      --gin.train.stats_period=${STAT_PERIOD} \
      --gin.config.GoogleViTConfig.dtype=\"${PREC}\" \
      --gin.trainer.Trainer.num_microbatches=${NUM_MICROBATCHES} \
      --gin.partitioning.PjitPartitioner.num_partitions=${MP} \
      --gin.utils.SaveCheckpointConfig.period=1000 \
      --multiprocess_gpu \
      --coordinator_address=${SLURM_LAUNCH_NODE_IPADDR}:12345 \
      --process_count=${SLURM_NTASKS} \
      --process_index=${SLURM_PROCID} \
      --gin.TRAIN_INDEX_DIR=\"${TRAIN_IDX_DIR}\" \
      --gin.EVAL_INDEX_DIR=\"${EVAL_IDX_DIR}\" \
      --gin_search_paths=/opt/rosetta \
  ;;
esac
