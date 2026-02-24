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

# Single-controller example:
# 1) Gateway (no GPU)
# 2) Single Controller (no GPU)
# 3) vLLM Worker (vLLM GPUs, subscribes to SC_* topics only)
# 4) JAX Worker (JAX GPUs, command-driven)
# 5) Prompt Source (no GPU)

set -euo pipefail

DIR="$(dirname "$0")"
JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/tmp/jax-compilation-cache}
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

# Defaults
DEBUG="false"
OUTPUT_DIR=${OUTPUT_DIR:-$(mktemp -d)}

SC_MODE="sync"  # sync | async

MODEL_NAME=""
MODEL_PATH=""
PARAM_MAPPING_PATH=""
TRANSFER_MODE=""

NUM_ITERATIONS="3"
NUM_BATCHES="10"
BATCH_SIZE="3"
NUM_ROLLOUTS="4"
UPDATE_INTERVAL="5"
MAX_STALENESS="0"
MAX_COMPLETED_PROMPTS="0"
DISPATCH_DELAY="0.0"

VLLM_ENFORCE_EAGER="0"
VLLM_GPU_MEMORY_UTILIZATION="0.9"

N_GPUS_VLLM="4"
N_GPUS_JAX="4"
GATEWAY_PORT="50051"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode=*)
      SC_MODE="${1#*=}"
      shift
      ;;
    --debug)
      DEBUG="true"
      shift
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
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
    --transfer-mode=*)
      TRANSFER_MODE="${1#*=}"
      shift
      ;;
    --num-iterations=*)
      NUM_ITERATIONS="${1#*=}"
      shift
      ;;
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
    --dispatch-delay=*)
      DISPATCH_DELAY="${1#*=}"
      shift
      ;;
    --max-completed-prompts=*)
      MAX_COMPLETED_PROMPTS="${1#*=}"
      shift
      ;;
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
    --n-gpus-vllm=*)
      N_GPUS_VLLM="${1#*=}"
      shift
      ;;
    --n-gpus-jax=*)
      N_GPUS_JAX="${1#*=}"
      shift
      ;;
    --gateway-port=*)
      GATEWAY_PORT="${1#*=}"
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Single-controller JAX-vLLM example."
      echo ""
      echo "Options:"
      echo "  --mode=sync|async          Scheduling mode (default: sync)"
      echo "  --model-path=PATH          HF snapshot directory containing model weights (required)"
      echo "  --param-mapping-path=PATH  JSON param mapping file (required)"
      echo "  --transfer-mode=MODE       grouped|fused|unfused"
      echo "  --num-iterations=N         Sync mode iterations (default: 3)"
      echo "  --num-batches=N            Async mode batches (default: 10, 0=infinite)"
      echo "  --batch-size=N             Prompts per request (default: 3)"
      echo "  --num-rollouts=N           Rollouts per prompt (default: 4)"
      echo "  --update-interval=N        Async weight update interval in prompts (default: 5)"
      echo "  --max-staleness=N          Async max in-flight prompts (default: 0=unlimited)"
      echo "  --max-completed-prompts=N  Async stop after N completed prompts (default: auto if num-batches>0)"
      echo "  --dispatch-delay=FLOAT     Async dispatch delay in seconds (default: 0.0)"
      echo "  --n-gpus-vllm=N            vLLM GPUs (default: 4)"
      echo "  --n-gpus-jax=N             JAX GPUs (default: 4)"
      echo "  --gateway-port=PORT        Gateway port (default: 50051)"
      echo "  --debug                    Verbose logging"
      echo "  --output-dir=DIR           Log output directory"
      echo "  --help                     Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      shift
      ;;
  esac
done

MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.2-1B-Instruct"}

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

if [[ -f "${PWD}/.env" ]]; then
  echo "Loading ${PWD}/.env"
  set -a && source "${PWD}/.env" && set +a
else
  echo ".env not found in ${PWD}, skipping"
fi

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "MODEL_PATH not provided, aborting"
  exit 1
fi
if [[ -z "${PARAM_MAPPING_PATH:-}" ]]; then
  echo "PARAM_MAPPING_PATH not provided, aborting"
  exit 1
fi
if [[ "${SC_MODE}" != "sync" && "${SC_MODE}" != "async" ]]; then
  echo "SC_MODE must be sync or async."
  exit 1
fi

if [[ "${SC_MODE}" == "async" && "${MAX_COMPLETED_PROMPTS}" == "0" && "${NUM_BATCHES}" != "0" ]]; then
  MAX_COMPLETED_PROMPTS=$((NUM_BATCHES * BATCH_SIZE))
fi

N_GPUS=$((N_GPUS_VLLM + N_GPUS_JAX))
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES_ARRAY=($(seq 0 $((N_GPUS - 1))))
else
  IFS=',' read -r -a CUDA_VISIBLE_DEVICES_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
fi

VLLM_GPU_ARRAY=("${CUDA_VISIBLE_DEVICES_ARRAY[@]:0:N_GPUS_VLLM}")
JAX_GPU_ARRAY=("${CUDA_VISIBLE_DEVICES_ARRAY[@]:N_GPUS_VLLM:N_GPUS}")

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_DEVICE_MAX_CONNECTIONS=16
export NCCL_BUFFSIZE=16777216
export GATEWAY_PORT
export GATEWAY_URL="localhost:${GATEWAY_PORT}"
export SC_MODE
export MODEL_NAME
export MODEL_PATH
export PARAM_MAPPING_PATH
export NUM_ITERATIONS
export NUM_BATCHES
export BATCH_SIZE
export NUM_ROLLOUTS
export UPDATE_INTERVAL
export MAX_STALENESS
export MAX_COMPLETED_PROMPTS
export DISPATCH_DELAY
export VLLM_ENFORCE_EAGER
export VLLM_GPU_MEMORY_UTILIZATION

export SC_JAX_COMMAND_TOPIC="sc/jax/commands"
export SC_JAX_EVENT_TOPIC="sc/jax/events"
export SC_JAX_RESULTS_TOPIC="inference/results/sc_forwarded"
export SC_SYNC_TOPIC="sync/weights_ready"
export SC_VLLM_EVENT_TOPIC="sc/vllm/events"

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

if [[ "${DEBUG}" == "true" ]]; then
  set -x
  export TF_CPP_MIN_LOG_LEVEL=0
  export NCCL_DEBUG=INFO
else
  export TF_CPP_MIN_LOG_LEVEL=2
  export VLLM_CONFIGURE_LOGGING=0
fi

mkdir -p "${OUTPUT_DIR}"
PIDS=()

echo "Logs directory: ${OUTPUT_DIR}"
echo "Starting single-controller stack in mode=${SC_MODE}"

echo "Starting Gateway..."
CUDA_VISIBLE_DEVICES= \
python "${DIR}/../../jax_inference_offloading/controller/gateway.py" \
  2>&1 | tee "${OUTPUT_DIR}/gateway.log" &
PIDS+=($!)
sleep 2

echo "Starting Single Controller..."
CUDA_VISIBLE_DEVICES= \
python "${DIR}/single_controller.py" \
  2>&1 | tee "${OUTPUT_DIR}/single_controller.log" &
PIDS+=($!)
sleep 1

echo "Starting vLLM Worker..."
CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${VLLM_GPU_ARRAY[*]}") \
MODEL_NAME=${MODEL_PATH:-$MODEL_NAME} \
python "${DIR}/vllm_worker.py" \
  2>&1 | tee "${OUTPUT_DIR}/vllm_worker.log" &
PIDS+=($!)
sleep 2

echo "Starting JAX Worker..."
CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${JAX_GPU_ARRAY[*]}") \
JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR} \
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.1 \
python "${DIR}/jax_worker.py" \
  2>&1 | tee "${OUTPUT_DIR}/jax_worker.log" &
PIDS+=($!)
sleep 2

echo "Starting Prompt Source..."
CUDA_VISIBLE_DEVICES= \
python "${DIR}/prompt_source.py" \
  2>&1 | tee "${OUTPUT_DIR}/prompt_source.log" &
PIDS+=($!)

echo "All processes started."
wait "${PIDS[@]}"
