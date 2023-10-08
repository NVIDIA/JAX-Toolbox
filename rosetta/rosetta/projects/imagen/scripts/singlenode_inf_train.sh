#!/bin/bash
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

ROSETTA_DIR=/opt/rosetta/rosetta

# Arguments
DATASET=$1
MODEL_TYPE=$2       # Model size (small, base, large)
PREC="$3"        # Precision (float32, float16, bfloat16)
NUM_GPUS=$4      # Number of GPUs (1, 2, 4, 8)
BSIZE_PER_GPU=$5 # Batch size per GPU (varies with model size)
LOG_DIR=$6       # Output log directory
MODEL_DIR_LOCAL=${7:-"runs/model_dir"}
MODEL_DIR=${PWD}/${MODEL_DIR_LOCAL}
INF_SERV_CT=${8:-1}
INF_SIZE=${9:-"xxl"}
NUM_MICROBATCHES=${10:-0}

mkdir -p $LOG_DIR

HOSTNAMES_DIR="${MODEL_DIR}/hostnames"
mkdir -p $HOSTNAMES_DIR
HOSTNAMES_FILE="${HOSTNAMES_DIR}/$(date -u +%Y%m%d_%H%M%S)-hostnames.txt"
INF_LOG_FILE=${LOG_DIR}/inf_serv

TRAIN_CMD="INFER_SAMPLES=${INFER_SAMPLES} ${ROSETTA_DIR}/projects/imagen/scripts/singlenode_mp_train_singlegpu.sh ${DATASET} ${MODEL_TYPE} ${PREC} ${NUM_GPUS} ${BSIZE_PER_GPU} ${LOG_DIR} ${MODEL_DIR_LOCAL} ${INF_SERV_CT} ${NUM_MICROBATCHES} ${HOSTNAMES_FILE}" 

## GV100
#MAX_BS=-1
INF_CONFIG_BASE=${ROSETTA_DIR}/projects/inference_serving/configs/
if [ $INF_SIZE == "large" ]
then
    INF_CONFIG_FILE=${INF_CONFIG_BASE}/t5_large_server.yml
elif [ $INF_SIZE == "xxl" ]
then
    INF_CONFIG_FILE=${INF_CONFIG_BASE}/t5_xxl_server.yml
fi

INF_CMD="DISABLE_TE=True python ${ROSETTA_DIR}/projects/inference_serving/server.py --total_devices=${INF_SERV_CT} --gpu_name=gv100_32g --config_file=${INF_CONFIG_FILE}"

for i in $(seq 0 $(expr $NUM_GPUS - 1))
do
  echo Starting Process $i
	CUDA_VISIBLE_DEVICES=$i python ${ROSETTA_DIR}/projects/imagen/scripts/specialized_run.py --proc_id=$i --proc_total_ct=$NUM_GPUS --inf_server_ct=${INF_SERV_CT} --train_run_command="${TRAIN_CMD}" --inf_server_run_command="${INF_CMD}" --hostnames_file=${HOSTNAMES_FILE} --inf_log_file="${INF_LOG_FILE}_${i}.log" &

done
