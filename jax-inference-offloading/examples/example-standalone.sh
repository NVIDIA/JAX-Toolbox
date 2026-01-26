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

# Standalone example: weight transfer + rollout generation without Tunix.
# This script launches the gateway, vLLM rollout worker, and trainer_standalone.py.

set -euo pipefail

DIR="$(dirname "$0")"
JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/tmp/jax-compilation-cache}
mkdir -p ${JAX_COMPILATION_CACHE_DIR}

# Set default values
DEBUG="false"
OUTPUT_DIR=${OUTPUT_DIR:-$(mktemp -d)}

# Model selection
MODEL_NAME=""
MODEL_PATH=""

# Transfer mode
TRANSFER_MODE=""

# vLLM runtime
VLLM_ENFORCE_EAGER="0"
VLLM_GPU_MEMORY_UTILIZATION="0.9"

# Debug-only: use dummy weights for JAX model
USE_DUMMY_WEIGHT="0"

# Device assignment
N_GPUS_VLLM="4"
N_GPUS_JAX="4"

# Gateway
GATEWAY_PORT="50051"

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
    # Model selection
    --model-name=*)
      MODEL_NAME="${1#*=}"
      shift
      ;;
    --model-path=*)
      MODEL_PATH="${1#*=}"
      shift
      ;;
    # Transfer mode
    --transfer-mode=*)
      TRANSFER_MODE="${1#*=}"
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
    --use-dummy-weight)
      USE_DUMMY_WEIGHT="1"
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
    # Gateway
    --gateway-port=*)
      GATEWAY_PORT="${1#*=}"
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Standalone example: weight transfer + rollout generation without Tunix."
      echo ""
      echo "Options:"
      echo "  --debug                    Enable debug mode with verbose logging."
      echo "  --output-dir=DIR           Directory to save logs and outputs. Default is a temporary directory."
      echo ""
      echo "  --model-name=NAME          HF repo id or model name (e.g., meta-llama/Llama-3.1-8B-Instruct)."
      echo "  --model-path=PATH          HF snapshot directory; if set, vLLM loads from this path."
      echo ""
      echo "  --transfer-mode=MODE       Transfer mode for trainer->vLLM weights (grouped/stacked/fused/unfused)."
      echo ""
      echo "  --vllm-enforce-eager       Force vLLM eager mode (sets VLLM_ENFORCE_EAGER=1)."
      echo "  --no-vllm-enforce-eager    Disable vLLM eager mode (sets VLLM_ENFORCE_EAGER=0)."
      echo "  --vllm-gpu-memory-utilization=FLOAT  vLLM GPU memory utilization (e.g., 0.7)."
      echo "  --use-dummy-weight         Use randomly initialized JAX weights (DEBUG ONLY)."
      echo ""
      echo "  --n-gpus-vllm=N            Number of GPUs for vLLM (default: 4)."
      echo "  --n-gpus-jax=N             Number of GPUs for JAX (default: 4)."
      echo ""
      echo "  --gateway-port=PORT        gRPC gateway port (default: 50051)."
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
if [[ -f "${PWD}/.env" ]]; then
  echo "Loading ${PWD}/.env"
  set -a && source "${PWD}/.env" && set +a
else
  echo ".env not found in ${PWD}, skipping"
fi

# ------------------------------------------------------------------------------
# Ensure model is already present on disk (download only when using real weights)
# ------------------------------------------------------------------------------

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Please set it in the .env file or export it."
fi

if [[ "${USE_DUMMY_WEIGHT}" == "1" ]]; then
  echo "Using dummy weights for JAX model (DEBUG ONLY)."
  MODEL_PATH=
else
  if [[ -n "${MODEL_PATH:-}" ]]; then
    echo "Using provided MODEL_PATH: ${MODEL_PATH}"
  else
    echo "MODEL_PATH not provided, downloading HF snapshot..."
    MODEL_PATH=$(python "${DIR}/download_model.py" --hub=hf --model="${MODEL_NAME}" --ignore="*.pth")
  fi
fi

# ------------------------------------------------------------------------------
# assign GPUs to vLLM and JAX
# ------------------------------------------------------------------------------
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
export NCCL_BUFFSIZE=16777216
export GATEWAY_PORT
export GATEWAY_URL="localhost:${GATEWAY_PORT}"
export MODEL_NAME
export MODEL_PATH
export USE_DUMMY_WEIGHT
export VLLM_ENFORCE_EAGER
export VLLM_GPU_MEMORY_UTILIZATION
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
                  --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUDNN,CUSTOM_CALL
                  --xla_gpu_collective_permute_combine_threshold_bytes=8589934592
                  --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
                  --xla_gpu_all_gather_combine_threshold_bytes=8589934592
                  --xla_gpu_all_reduce_combine_threshold_bytes=8589934592"
if [[ -n "${TRANSFER_MODE:-}" ]]; then
  export TRANSFER_MODE
fi

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
echo "Logs will be saved to: ${OUTPUT_DIR}"

# ------------------------------------------------------------------------------
# Launch components
# ------------------------------------------------------------------------------

# Gateway server (no GPU)
CUDA_VISIBLE_DEVICES= \
python "${DIR}/../jax_inference_offloading/controller/gateway.py" 2>&1 | tee "${OUTPUT_DIR}/gateway.log" &
PIDS+=($!)

# vLLM rollout worker
CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${VLLM_GPU_ARRAY[*]}") \
MODEL_NAME=${MODEL_PATH:-$MODEL_NAME} \
python "${DIR}/rollout.py" 2>&1 | tee "${OUTPUT_DIR}/rollout.log" &
PIDS+=($!)

# Standalone trainer (weight transfer + generation demo)
CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${JAX_GPU_ARRAY[*]}") \
JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR} \
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.1 \
python "${DIR}/trainer_standalone.py" 2>&1 | tee "${OUTPUT_DIR}/trainer.log" &
PIDS+=($!)

wait "${PIDS[@]}"
