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
CFG=${CFG:=2}
BASE_PATH=${BASE_PATH:=\"/opt/rosetta/runs/imagen_base/checkpoint_5000\"}
SR1_PATH=${SR1_PATH:=\"/opt/rosetta/runs/efficient_sr1/checkpoint_5000\"}
SR2_PATH=${SR1_PATH:=\"/opt/rosetta/runs/efficient_sr2/checkpoint_5000\"}
PROMPT_TEXT_FILE=${PROMPT_TEXT_FILE:=\"/opt/rosetta/rosetta/projects/diffusion/tests/custom_eval_prompts/custom_eval_prompts.txt\"}

export DISABLE_TE=True
python /opt/rosetta/rosetta/projects/imagen/imagen_pipe.py \
    --gin_file="/opt/rosetta/rosetta/projects/imagen/configs/imagen_1024_sample.gin" \
    --gin.base_model/utils.RestoreCheckpointConfig.path="${BASE_PATH}" \
    --gin.sr256/utils.RestoreCheckpointConfig.path="${SR1_PATH}" \
    --gin.sr1024/utils.RestoreCheckpointConfig.path="${SR2_PATH}" \
    --gin.T5_CHECKPOINT_PATH="\"/opt/rosetta/rosetta/projects/inference_serving/checkpoints/checkpoint_1000000_t5_1_1_xxl\"" \
    --gin.base_model/samplers.CFGSamplingConfig.cf_guidance_weight=${CFG} \
    --gin.PROMPT_TEXT_FILE=${PROMPT_TEXT_FILE} \
    --gin.GLOBAL_BATCH_SIZE=4 \
    --gin.SAVE_DIR="\"generations/generations-${CFG}\"" \
    --gin.GEN_PER_PROMPT=1 \
    --gin.RESUME_FROM=0