#! /bin/bash

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

MODEL_SIZE=$1

# Arguments
PREC="bfloat16"

BSIZE=8 #Overridden

MP=1

MODEL_DIR_LOCAL="/tmp/${MODEL_SIZE}_t5_inference_debugs"

MODEL_DIR=${PWD}/${MODEL_DIR_LOCAL}

mkdir -p $MODEL_DIR

echo $MODEL_DIR

CFG_NAME="embed_${MODEL_SIZE}.gin"

T5X_SERVER_DIR="/opt/rosetta/rosetta/projects/inference_serving/t5/"

python ${T5X_SERVER_DIR}/embed_t5x.py \
  --gin_file="${T5X_SERVER_DIR}/${CFG_NAME}" \
  --gin.EVAL_OUTPUT_DIR=\"${MODEL_DIR}\" \
  --gin.network.T5Config.dtype=\"${PREC}\" \
  --gin.BATCH_SIZE=$BSIZE \
  --gin.partitioning.PjitPartitioner.num_partitions=$MP
