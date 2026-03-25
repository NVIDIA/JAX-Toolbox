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

# Async split architecture example: Autonomous prompt dispatch with opportunistic weight updates.
#
# This script launches 4 processes:
# 1. Gateway (no GPU) - gRPC message broker
# 2. vLLM Worker (vLLM GPUs) - runs inference, checks for weight updates between batches
# 3. JAX Controller (JAX GPUs) - accumulates rollouts, pushes weight updates
# 4. Prompt Dispatcher (no GPU) - continuously sends prompts (autonomous)
#
# Key differences from synchronous version:
# - Prompt dispatcher does NOT wait for sync signals from JAX controller
# - vLLM worker checks for weight updates between batches (opportunistic)
# - JAX controller accumulates streamed rollout results
# - Results are streamed per-rollout instead of batched

set -euo pipefail

DIR="$(dirname "$0")"
JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/tmp/jax-compilation-cache}
mkdir -p ${JAX_COMPILATION_CACHE_DIR}

# Set default values
DEBUG="false"
OUTPUT_DIR=${OUTPUT_DIR:-$(mktemp -d)}

# Model configuration
MODEL_NAME=""
MODEL_PATH=""
PARAM_MAPPING_PATH=""

# Transfer mode
TRANSFER_MODE=""

# vLLM runtime
VLLM_ENFORCE_EAGER="0"
VLLM_GPU_MEMORY_UTILIZATION="0.9"

# Async configuration
NUM_BATCHES="10"
BATCH_SIZE="3"
NUM_ROLLOUTS="4"
UPDATE_INTERVAL="1"
MAX_STALENESS="1"  # 0 = unlimited staleness; should be >= UPDATE_INTERVAL to avoid deadlock
MAX_COMPLETED_PROMPTS="30"
DISPATCH_DELAY="0.0"

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
    # Model configuration
    --model-name=*)
      MODEL_NAME="${1#*=}"
      shift
      ;;
    --model-path=*)
      MODEL_PATH="${1#*=}"
      shift
      ;;
    --param-mapping-path=*)
      PARAM_MAPPING_PATH="${1#*=}"
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
    # Async configuration
    --num-batches=*)
      NUM_BATCHES="${1#*=}"
      shift
      ;;
    --batch-size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    --num-rollouts=*)
      NUM_ROLLOUTS="${1#*=}"
      shift
      ;;
    --update-interval=*)
      UPDATE_INTERVAL="${1#*=}"
      shift
      ;;
    --max-staleness=*)
      MAX_STALENESS="${1#*=}"
      shift
      ;;
    --max-completed-prompts=*)
      MAX_COMPLETED_PROMPTS="${1#*=}"
      shift
      ;;
    --dispatch-delay=*)
      DISPATCH_DELAY="${1#*=}"
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
      echo "Async split architecture example: Autonomous prompt dispatch with opportunistic weight updates."
      echo ""
      echo "This example launches 4 processes:"
      echo "  1. Gateway (no GPU) - gRPC message broker"
      echo "  2. vLLM Worker (vLLM GPUs) - runs inference, checks for weight updates"
      echo "  3. JAX Controller (JAX GPUs) - accumulates rollouts, pushes weight updates"
      echo "  4. Prompt Dispatcher (no GPU) - continuously sends prompts"
      echo ""
      echo "Options:"
      echo "  --debug                    Enable debug mode with verbose logging."
      echo "  --output-dir=DIR           Directory to save logs and outputs."
      echo ""
      echo "  --model-name=NAME          HF model name (for architecture selection)."
      echo "  --model-path=PATH          HF snapshot directory containing model weights."
      echo "  --param-mapping-path=PATH  Path to JSON param mapping file (required)."
      echo ""
      echo "  --transfer-mode=MODE       Transfer mode for trainer->vLLM weights (grouped/fused/unfused)."
      echo ""
      echo "  --vllm-enforce-eager       Force vLLM eager mode (sets VLLM_ENFORCE_EAGER=1)."
      echo "  --no-vllm-enforce-eager    Disable vLLM eager mode (sets VLLM_ENFORCE_EAGER=0)."
      echo "  --vllm-gpu-memory-utilization=FLOAT  vLLM GPU memory utilization (e.g., 0.7)."
      echo ""
      echo "  --num-batches=N            Number of batches to dispatch (default: 10, 0 for infinite)."
      echo "  --batch-size=N             Number of prompts per batch (default: 3)."
      echo "  --num-rollouts=N           Number of rollouts per prompt (default: 4)."
      echo "  --update-interval=N        Push weight update after N prompts (default: 10)."
      echo "  --max-staleness=N          Max prompts vLLM processes before weight update (default: 0=unlimited)."
      echo "                             Must be >= update-interval to avoid deadlock."
      echo "  --max-completed-prompts=N  Stop after N completed prompts (default: 100)."
      echo "  --dispatch-delay=FLOAT     Delay between dispatches in seconds (default: 0.0)."
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
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.2-1B-Instruct"}

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
# Ensure model is already present on disk
# ------------------------------------------------------------------------------
if [[ -n "${MODEL_PATH:-}" ]]; then
  echo "Using provided MODEL_PATH: ${MODEL_PATH}"
else
  echo "MODEL_PATH not provided, aborting"
  exit 1
fi

# Validate required param_mapping_path
if [[ -z "${PARAM_MAPPING_PATH:-}" ]]; then
  echo "ERROR: --param-mapping-path is required for the async architecture."
  exit 1
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
export PARAM_MAPPING_PATH
export NUM_BATCHES
export BATCH_SIZE
export NUM_ROLLOUTS
export UPDATE_INTERVAL
export MAX_STALENESS
export MAX_COMPLETED_PROMPTS
export DISPATCH_DELAY
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
echo ""
echo "=== Async Split Architecture: 4 Processes ==="
echo "  1. Gateway (no GPU)"
echo "  2. vLLM Worker (GPUs: ${VLLM_GPU_ARRAY[*]}) - opportunistic weight updates"
echo "  3. JAX Controller (GPUs: ${JAX_GPU_ARRAY[*]}) - accumulates rollouts"
echo "  4. Prompt Dispatcher (no GPU) - autonomous dispatch"
echo ""
echo "Async Configuration:"
echo "  Batches to dispatch: ${NUM_BATCHES}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Rollouts per prompt: ${NUM_ROLLOUTS}"
echo "  Weight update interval: ${UPDATE_INTERVAL} prompts"
echo "  Max staleness: ${MAX_STALENESS} prompts (0=unlimited)"
echo "  Max completed prompts: ${MAX_COMPLETED_PROMPTS}"
echo "================================================"
echo ""

# ------------------------------------------------------------------------------
# Launch components
# ------------------------------------------------------------------------------

# 1. Gateway server (no GPU)
echo "Starting Gateway..."
CUDA_VISIBLE_DEVICES= \
python "${DIR}/../../jax_inference_offloading/controller/gateway.py" 2>&1 | tee "${OUTPUT_DIR}/gateway.log" &
PIDS+=($!)

# Give gateway time to start
sleep 2

# 2. vLLM Worker (async version with weight update checks)
echo "Starting Async vLLM Worker..."
CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${VLLM_GPU_ARRAY[*]}") \
MODEL_NAME=${MODEL_PATH:-$MODEL_NAME} \
python "${DIR}/vllm_worker.py" 2>&1 | tee "${OUTPUT_DIR}/vllm_worker.log" &
PIDS+=($!)

# 3. JAX Controller (async version with accumulation)
echo "Starting Async JAX Controller..."
CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${JAX_GPU_ARRAY[*]}") \
JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR} \
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.1 \
python "${DIR}/jax_controller.py" 2>&1 | tee "${OUTPUT_DIR}/jax_controller.log" &
PIDS+=($!)

# 4. Prompt Dispatcher (async version - autonomous)
echo "Starting Async Prompt Dispatcher..."
CUDA_VISIBLE_DEVICES= \
python "${DIR}/prompt_dispatcher.py" 2>&1 | tee "${OUTPUT_DIR}/prompt_dispatcher.log" &
PIDS+=($!)

echo ""
echo "All processes started. Waiting for completion..."
echo "Check logs in: ${OUTPUT_DIR}"
echo ""

wait "${PIDS[@]}"
