#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

DIR="$(dirname "$0")"

# Arguments - General
DEBUG="false"
OUTPUT_DIR=${OUTPUT_DIR:-$(mktemp -d)}
NSYS_PROFILE_NAME=""

# Arguments - Model selection
MODEL_NAME=""
MODEL_PATH=""
USE_REAL_WEIGHTS="false"

# Arguments - vLLM runtime
VLLM_ENFORCE_EAGER="1"
VLLM_GPU_MEMORY_UTILIZATION="0.7"

# Arguments - Trainer runtime
TRANSFER_MODE="grouped"  # grouped | fused | unfused (trainer default: grouped)
USE_POLYMORPHIC_MESH="0"

# Arguments - Device assignment
N_GPUS_VLLM="4"
N_GPUS_JAX="4"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    # General
    --debug)
      DEBUG="true"
      shift
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --nsys-profile-name=*)
      NSYS_PROFILE_NAME="${1#*=}"
      shift
      ;;

    # Model selection
    --model-name=*)
      MODEL_NAME="${1#*=}"
      shift
      ;;
    --use-real-weights)
      USE_REAL_WEIGHTS="true"
      shift
      ;;
    --model-path=*)
      MODEL_PATH="${1#*=}"
      shift
      ;;

    # vLLM runtime
    --vllm-enforce-eager)
      VLLM_ENFORCE_EAGER="1"
      shift
      ;;
    --no-vllm-enforce-eager)
      VLLM_ENFORCE_EAGER="0"
      shift
      ;;
    --vllm-gpu-memory-utilization=*)
      VLLM_GPU_MEMORY_UTILIZATION="${1#*=}"
      shift
      ;;

    # Trainer runtime
    --transfer-mode=*)
      TRANSFER_MODE="${1#*=}"
      shift
      ;;
    --use-polymorphic-mesh)
      USE_POLYMORPHIC_MESH="1"
      shift
      ;;

    # Device assignment
    --n-gpus-vllm=*)
      N_GPUS_VLLM="${1#*=}"
      shift
      ;;
    --n-gpus-jax=*)
      N_GPUS_JAX="${1#*=}"
      shift
      ;;

    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --debug                    Enable debug mode with verbose logging."
      echo "  --output-dir=DIR           Directory to save logs and outputs. Default is a temporary directory."
      echo "  --nsys-profile-name=NAME   Enable NVIDIA Nsight Systems profiling with the given profile name."
      echo ""
      echo "  --model-name=NAME          HF repo id or model name (e.g., meta-llama/Llama-3.1-8B-Instruct)."
      echo "  --use-real-weights         Use real model weights instead of dummy weights."
      echo "  --model-path=PATH          HF snapshot directory; if set, vLLM loads from this path."
      echo ""
      echo "  --vllm-enforce-eager       Force vLLM eager mode (sets VLLM_ENFORCE_EAGER=1)."
      echo "  --no-vllm-enforce-eager    Disable vLLM eager mode (sets VLLM_ENFORCE_EAGER=0)."
      echo "  --vllm-gpu-memory-utilization=FLOAT  vLLM GPU memory utilization (e.g., 0.7)."
      echo ""
      echo "  --transfer-mode=MODE       Transfer mode for trainer->vLLM weights (grouped/fused/unfused)."
      echo "  --use-polymorphic-mesh     Enable polymorphic mesh for trainer (sets USE_POLYMORPHIC_MESH=1)."
      echo ""
      echo "  --n-gpus-vllm=N            Number of GPUs for vLLM (default: 4)."
      echo "  --n-gpus-jax=N             Number of GPUs for JAX (default: 4)."
      echo ""
      echo "  --help                     Show this help message and exit."
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      shift
      ;;
  esac
done

# Model selection default
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.1-8B-Instruct"}

# ------------------------------------------------------------------------------
# Kill all processes when done.
# ------------------------------------------------------------------------------
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# ------------------------------------------------------------------------------
# load environment variables from .env file
# ------------------------------------------------------------------------------
if [[ -f ../.env ]]; then
  echo "Loading environment variables from .env file"
  set -a && source "${DIR}/../.env" && set +a
else
  echo ".env file not found, skipping"
fi

# ------------------------------------------------------------------------------
# Ensure model is already present on disk (download only when using real weights)
# ------------------------------------------------------------------------------

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Please set it in the .env file or export it."
fi

if [[ "${USE_REAL_WEIGHTS}" == "true" ]]; then
  echo "Using real weights."
  if [[ -z "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH not provided, downloading HF snapshot..."
    MODEL_PATH=$(python "${DIR}/download_model.py" --hub=hf --model=${MODEL_NAME} --ignore="*.pth")
  else
    echo "Using provided MODEL_PATH: ${MODEL_PATH}"
  fi
else
  echo "Using dummy weights for JAX model. No download will be performed."
fi
  
# ------------------------------------------------------------------------------
# assign GPUs to vLLM and JAX
# ------------------------------------------------------------------------------
N_GPUS_VLLM=${N_GPUS_VLLM:-4}
N_GPUS_JAX=${N_GPUS_JAX:-4}
N_GPUS=$((N_GPUS_VLLM + N_GPUS_JAX))

# Derive CUDA_VISIBLE_DEVICES_ARRAY
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES_ARRAY=($(seq 0 $((N_GPUS - 1))))
else
  IFS=',' read -r -a CUDA_VISIBLE_DEVICES_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
fi

VLLM_GPU_ARRAY=("${CUDA_VISIBLE_DEVICES_ARRAY[@]:0:N_GPUS_VLLM}")
JAX_GPU_ARRAY=("${CUDA_VISIBLE_DEVICES_ARRAY[@]:N_GPUS_VLLM:N_GPUS}")

# ------------------------------------------------------------------------------
# common environment
# ------------------------------------------------------------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_DEVICE_MAX_CONNECTIONS=16
export NCCL_CUMEM_ENABLE=0  # https://docs.vllm.ai/en/v0.9.1/usage/troubleshooting.html#known-issues
export NCCL_BUFFSIZE=16777216
export GATEWAY_PORT=50051
export GATEWAY_URL="localhost:${GATEWAY_PORT}"
export MODEL_NAME
export MODEL_PATH
export TRANSFER_MODE
export USE_POLYMORPHIC_MESH
export VLLM_ENFORCE_EAGER
export VLLM_GPU_MEMORY_UTILIZATION
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
                  --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUDNN,CUSTOM_CALL
                  --xla_gpu_collective_permute_combine_threshold_bytes=8589934592
                  --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
                  --xla_gpu_all_gather_combine_threshold_bytes=8589934592
                  --xla_gpu_all_reduce_combine_threshold_bytes=8589934592"

if [ "$DEBUG" == "true" ]; then
    set -x
    export TF_CPP_MIN_LOG_LEVEL=0
    export NCCL_DEBUG=INFO  # Enable NCCL debug logs
else
    export TF_CPP_MIN_LOG_LEVEL=2  # Suppress TensorFlow debug logs
    export VLLM_CONFIGURE_LOGGING=0  # Suppress vLLM logging
fi

PIDS=()

mkdir -p "${OUTPUT_DIR}"

# Setup profiling command if enabled
if [[ -n "${NSYS_PROFILE_NAME:-}" ]]; then
  NSYS_OUTPUT="${OUTPUT_DIR}/${NSYS_PROFILE_NAME}"
  TRAINER_PROF_CMD="nsys profile -s none -o "${NSYS_OUTPUT}" --force-overwrite true --capture-range=cudaProfilerApi --cuda-graph-trace=node --trace=cuda,nvtx --capture-range-end=stop"
  echo "Nsys outputs will be saved to: ${NSYS_OUTPUT}"
else
  TRAINER_PROF_CMD=""
  echo "Nsys profiling not enabled. To enable, use --nsys-profile-name=PROFILE_NAME"
fi

echo "Logs will be saved to: ${OUTPUT_DIR}"

# todo: python -m jax_inference_offloading.controller_server ...
CUDA_VISIBLE_DEVICES= \
python "${DIR}/../jax_inference_offloading/controller/gateway.py" 2>&1 | tee ${OUTPUT_DIR}/gateway.log &
PIDS+=($!)

CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${VLLM_GPU_ARRAY[*]}") \
python "${DIR}/rollout.py" 2>&1 | tee ${OUTPUT_DIR}/rollout.log &
PIDS+=($!)

CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${JAX_GPU_ARRAY[*]}") \
${TRAINER_PROF_CMD} python "${DIR}/trainer.py" 2>&1 | tee ${OUTPUT_DIR}/trainer.log &
PIDS+=($!)

wait "${PIDS[@]}"
