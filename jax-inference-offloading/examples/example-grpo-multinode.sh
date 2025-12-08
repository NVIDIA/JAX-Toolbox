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
_ARGS="$*"

# Arguments - General
DEBUG="false"
OUTPUT_DIR=""
NSYS_PROFILE_NAME=""

# Arguments - Container runtime
CONTAINER_IMAGE=""
CONTAINER_MOUNTS=""
CONTAINER_NAME="main"

# Arguments - Process placement
N_NODES_JAX=""
N_NODES_VLLM=""

# Arguments - Model selection
MODEL_NAME=""
MODEL_PATH=""
JAX_USE_DUMMY_WEIGHTS="false"

# Arguments - Dataset
DATASET_DIR=""

# Arguments - Trainer runtime
ROLLOUT_ENGINE="vllm_gpu"      # vllm_gpu | vanilla
SCRATCHDIR="/content"
JAX_COMPILATION_CACHE_DIR="/tmp/jax-compilation-cache"

# Arguments - Ray settings
RAY_PORT="20527"
RAY_CLIENT_SERVER_PORT="24430"

# Arguments - vLLM runtime
VLLM_ENFORCE_EAGER="0"
VLLM_LOAD_FORMAT="dummy"
VLLM_GPU_MEMORY_UTILIZATION="0.75"
VLLM_DISTRIBUTED_BACKEND="ray"

# Arguments - Trainer runtime
TRANSFER_MODE="grouped"
USE_POLYMORPHIC_MESH="0"

# Arguments - Device assignment
N_GPUS_PER_NODE="8"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    # General
    --debug)
      DEBUG="true"
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      ;;
    --nsys-profile-name=*)
      NSYS_PROFILE_NAME="${1#*=}"
      ;;

    # Container runtime
    --container-image=*)
      CONTAINER_IMAGE="${1#*=}"
      ;;
    --container-mounts=*)
      CONTAINER_MOUNTS="${1#*=}"
      ;;
    --container-name=*)
      CONTAINER_NAME="${1#*=}"
      ;;

    # Node counts
    --n-nodes-jax=*)
      N_NODES_JAX="${1#*=}"
      ;;
    --n-nodes-vllm=*)
      N_NODES_VLLM="${1#*=}"
      ;;

    # Model selection
    --model-name=*)
      MODEL_NAME="${1#*=}"
      ;;
    --model-path=*)
      MODEL_PATH="${1#*=}"
      ;;
    --jax-use-dummy-weights)
      JAX_USE_DUMMY_WEIGHTS="true"
      ;;
    --jax-use-real-weights)
      JAX_USE_DUMMY_WEIGHTS="false"
      ;;

    # Dataset directory
    --dataset-dir=*)
      DATASET_DIR="${1#*=}"
      ;;

    # Trainer runtime
    --rollout-engine=*)
      ROLLOUT_ENGINE="${1#*=}"
      ;;
    --scratchdir=*)
      SCRATCHDIR="${1#*=}"
      ;;
    --transfer-mode=*)
      TRANSFER_MODE="${1#*=}"
      ;;
    --jax-compilation-cache-dir=*)
      JAX_COMPILATION_CACHE_DIR="${1#*=}"
      ;;

    # GRPO hyperparameter wildcard: --grpo-<key>=<value>
    --grpo-*=*)
      RAW_KV="${1#--grpo-}"    # e.g., num-epochs=2
      KEY="${RAW_KV%%=*}"      # num-epochs
      VAL="${RAW_KV#*=}"       # 2
      UKEY=$(echo "$KEY" | tr '[:lower:]-' '[:upper:]_')  # NUM_EPOCHS
      VAR="GRPO_${UKEY}"       # GRPO_NUM_EPOCHS
      export "$VAR"="$VAL"     # export GRPO_NUM_EPOCHS=2
      ;;

    # vLLM runtime
    --vllm-enforce-eager)
      VLLM_ENFORCE_EAGER="1"
      ;;
    --no-vllm-enforce-eager)
      VLLM_ENFORCE_EAGER="0"
      ;;
    --vllm-load-format=*)
      VLLM_LOAD_FORMAT="${1#*=}"
      ;;
    --vllm-gpu-memory-utilization=*)
      VLLM_GPU_MEMORY_UTILIZATION="${1#*=}"
      ;;
    --vllm-distributed-backend=*)
      VLLM_DISTRIBUTED_BACKEND="${1#*=}"
      ;;

    # Trainer runtime
    --use-polymorphic-mesh)
      USE_POLYMORPHIC_MESH="1"
      ;;

    # Device assignment
    --n-gpus-per-node=*)
      N_GPUS_PER_NODE="${1#*=}"
      ;;

    # Ray settings
    --ray-port=*)
      RAY_PORT="${1#*=}"
      ;;
    --ray-client-server-port=*)
      RAY_CLIENT_SERVER_PORT="${1#*=}"
      ;;

    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --debug                    Enable debug mode with verbose logging."
      echo "  --output-dir=DIR           Directory to save logs and outputs. Default is a temporary directory."
      echo "  --nsys-profile-name=NAME   Enable NVIDIA Nsight Systems profiling with the given profile name."
      echo "  --container-image=IMAGE    Container image to use for all srun segments."
      echo "  --container-name=NAME      Container name for persistence across srun steps (default: main)."
      echo ""
      echo "  --n-nodes-jax=N            Number of nodes for JAX (default: allocation_nodes-N_NODES_VLLM)."
      echo "  --n-nodes-vllm=M           Number of nodes for vLLM (multi-node supported, default: allocation_nodes-N_NODES_JAX)."
      echo ""
      echo "  --model-name=NAME          HF repo id or model name (e.g., meta-llama/Llama-3.1-8B-Instruct)."
      echo "  --model-path=PATH          HF snapshot directory; if set, vLLM loads from this path."
      echo "  --dataset-dir=PATH         Persistent dataset/cache directory for TFDS (optional)."
      echo "  --jax-use-dummy-weights    Use dummy weights for JAX (default)."
      echo "  --jax-use-real-weights     Use real model weights for JAX (must be passed in via --model-path)."
      echo ""
      echo "  --rollout-engine=ENGINE    Rollout engine: vllm_gpu | vanilla."
      echo "  --scratchdir=DIR           Scratch directory for checkpoints/logs."
      echo "  --transfer-mode=MODE       Transfer mode for trainer->vLLM weights (unfused/grouped/stacked/stacked_graph/etc.)."
      echo "  --jax-compilation-cache-dir=DIR  JAX compilation cache directory."
      echo ""
      echo "  --grpo-<key>=<value>       GRPO hyperparameter override; e.g., --grpo-num-epochs=2 (exports GRPO_NUM_EPOCHS=2)."
      echo ""
      echo "  --vllm-enforce-eager       Force vLLM eager mode (sets VLLM_ENFORCE_EAGER=1)."
      echo "  --no-vllm-enforce-eager    Disable vLLM eager mode (sets VLLM_ENFORCE_EAGER=0)."
      echo "  --vllm-load-format=FORMAT  vLLM model load format (e.g., dummy/auto/pt/safetensors)."
      echo "  --vllm-gpu-memory-utilization=FLOAT vLLM GPU memory utilization (e.g., 0.7)."
      echo "  --vllm-distributed-backend=BACKEND  vLLM distributed backend (ray/mp)."
      echo ""
      echo "  --use-polymorphic-mesh     Enable polymorphic mesh for trainer (sets USE_POLYMORPHIC_MESH=1)."
      echo ""
      echo "  --n-gpus-per-node=N        Number of GPUs on the target node (default: 8)."
      echo ""
      echo "  --ray-port=PORT            Ray control-plane port (default: 20527)."
      echo "  --ray-client-server-port=PORT Ray client-server port (default: 24430)."
      echo ""
      echo "  --help                     Show this help message and exit."
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
  shift
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR=$(mktemp -p "$PWD" -d output.XXXXXXXX)
fi
mkdir -p "${OUTPUT_DIR}"
echo "Artifacts will be saved to: ${OUTPUT_DIR}"

# Save job metadata
{
  echo "timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "user=${USER}"
  echo "host_submit=$(hostname -f 2>/dev/null || hostname)"
  echo "script=$0"
  echo "args=$_ARGS"
  echo "container_image=${CONTAINER_IMAGE:-}"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_job_name=${SLURM_JOB_NAME:-}"
  echo "slurm_nnodes=${SLURM_NNODES:-}"
  echo "slurm_nodelist=${SLURM_NODELIST:-}"
} > "${OUTPUT_DIR}/job.txt"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job "${SLURM_JOB_ID}" > "${OUTPUT_DIR}/scontrol.txt" 2>&1 || true
fi

env | grep -v TOKEN > "${OUTPUT_DIR}/env.txt" || true

# If no mounts provided, at least mount the output directory
MOUNTS="${OUTPUT_DIR}:${OUTPUT_DIR}"
if [[ -n "${CONTAINER_MOUNTS}" ]]; then
  MOUNTS="${MOUNTS},${CONTAINER_MOUNTS}"
fi

# If a dataset directory is provided, ensure it exists and mount it
if [[ -z "${DATASET_DIR}" ]]; then
  DATASET_DIR=$(mktemp -p "$PWD" -d)
  MOUNTS="${MOUNTS},${DATASET_DIR}:${DATASET_DIR}"
fi

# Determine allocation hosts and partition for JAX/vLLM
NODELIST="${SLURM_JOB_NODELIST:-${SLURM_NODELIST}}"
readarray -t HOSTS_RAW < <(scontrol show hostnames "$NODELIST")
HOSTS=()
for h in "${HOSTS_RAW[@]}"; do
  HOSTS+=("${h%%.*}")
done
TOTAL_NODES=${#HOSTS[@]}

if [[ -z "${N_NODES_JAX:-}" && -z "${N_NODES_VLLM:-}" ]]; then
  echo "You must specify at least one of --n-nodes-jax or --n-nodes-vllm." >&2
  exit 1
fi
if [[ -z "${N_NODES_JAX:-}" ]]; then
  N_NODES_JAX=$(( TOTAL_NODES - N_NODES_VLLM ))
fi
if [[ -z "${N_NODES_VLLM:-}" ]]; then
  N_NODES_VLLM=$(( TOTAL_NODES - N_NODES_JAX ))
fi
if (( N_NODES_JAX < 1 )); then
  echo "N_NODES_JAX must be >= 1 (computed: ${N_NODES_JAX})." >&2
  exit 1
fi
if (( N_NODES_VLLM < 1 )); then
  echo "N_NODES_VLLM must be >= 1 (computed: ${N_NODES_VLLM})." >&2
  exit 1
fi
if (( N_NODES_JAX + N_NODES_VLLM > TOTAL_NODES )); then
  echo "Requested nodes exceed allocation: JAX(${N_NODES_JAX}) + vLLM(${N_NODES_VLLM}) > TOTAL(${TOTAL_NODES})." >&2
  exit 1
fi

JAX_HOSTS=("${HOSTS[@]:0:N_NODES_JAX}")
VLLM_HOSTS=("${HOSTS[@]:N_NODES_JAX:N_NODES_VLLM}")

JAX_HOSTS_CSV=$(IFS=','; echo "${JAX_HOSTS[*]}")
JAX_NODELIST=$(IFS=','; echo "${JAX_HOSTS[*]}")
VLLM_HOSTS_CSV=$(IFS=','; echo "${VLLM_HOSTS[*]}")
VLLM_NODELIST=$(IFS=','; echo "${VLLM_HOSTS[*]}")

JAX_COORDINATOR_ADDR="${JAX_HOSTS[0]}"
JAX_COORDINATOR_PORT="${JAX_COORDINATOR_PORT:-12345}"
VLLM_CONTROLLER_ADDR="${VLLM_HOSTS[0]}"
RAY_HEAD_HOST="${VLLM_HOSTS[0]}"
RAY_HEAD_IP=$(nslookup -type=a ${RAY_HEAD_HOST} 2>/dev/null | awk '/^Name:/{f=1;next} f && /^Address:/{print $2}')

echo "Allocation nodes: ${HOSTS[*]}"
echo "JAX nodes (${#JAX_HOSTS[@]}): ${JAX_HOSTS[*]}"
echo "vLLM nodes (${#VLLM_HOSTS[@]}): ${VLLM_HOSTS[*]}"

# Model selection default
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.1-8B-Instruct"}

# ------------------------------------------------------------------------------
# Function to (optionally) wrap a command with nsys
# ------------------------------------------------------------------------------
maybe_run_nsys() {
  if [[ -n "${NSYS_PROFILE_NAME:-}" ]]; then
    nsys profile -s none \
      -o "${NSYS_OUTPUT}-$(hostname -s)" \
      --force-overwrite true \
      --capture-range=cudaProfilerApi \
      --cuda-graph-trace=node \
      --trace=cuda,nvtx \
      --capture-range-end=stop \
      "$@"
  else
    "$@"
  fi
}

# ------------------------------------------------------------------------------
# Kill all processes when done.
# ------------------------------------------------------------------------------
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# ------------------------------------------------------------------------------
# load environment variables from .env file
# ------------------------------------------------------------------------------
if [[ -f "${DIR}/../.env" ]]; then
  echo "Loading environment variables from .env file"
  set -a && source "${DIR}/../.env" && set +a
else
  echo ".env file not found, skipping"
fi

# ------------------------------------------------------------------------------
# Ensure HF_TOKEN is provided
# ------------------------------------------------------------------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. Please set it in the .env file or export it."
fi
  
# ------------------------------------------------------------------------------
# assign GPUs to vLLM and JAX
# ------------------------------------------------------------------------------
JAX_LOCAL_DEVICE_IDS=$(seq -s, 0 $((N_GPUS_PER_NODE - 1)))
VLLM_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS_PER_NODE - 1)))
VLLM_TENSOR_PARALLEL_SIZE=$((N_GPUS_PER_NODE * N_NODES_VLLM))

# ------------------------------------------------------------------------------
# Setup profiling variables if enabled
# ------------------------------------------------------------------------------
if [[ -n "${NSYS_PROFILE_NAME:-}" ]]; then
  NSYS_OUTPUT="${OUTPUT_DIR}/${NSYS_PROFILE_NAME}"
  echo "Nsys outputs will be saved to: ${NSYS_OUTPUT}-<hostname>"
else
  echo "Nsys profiling not enabled. To enable, use --nsys-profile-name=PROFILE_NAME"
fi

# ------------------------------------------------------------------------------
# common environment
# ------------------------------------------------------------------------------
export -f maybe_run_nsys
export HF_TOKEN
export OUTPUT_DIR
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_DEVICE_MAX_CONNECTIONS=16
export NCCL_BUFFSIZE=16777216
export GATEWAY_PORT=50051
export GATEWAY_URL="${JAX_COORDINATOR_ADDR}:${GATEWAY_PORT}"
export MODEL_NAME
export MODEL_PATH
export ROLLOUT_ENGINE
export SCRATCHDIR
export TRANSFER_MODE
export DATASET_DIR
export VLLM_ENFORCE_EAGER
export VLLM_LOAD_FORMAT
export VLLM_GPU_MEMORY_UTILIZATION
export VLLM_TENSOR_PARALLEL_SIZE
export VLLM_DISTRIBUTED_BACKEND
export JAX_COORDINATOR_ADDR
export JAX_COORDINATOR_PORT
export JAX_NODELIST
export JAX_LOCAL_DEVICE_IDS
export JAX_NUM_PROCESSES="${#JAX_HOSTS[@]}"
export JAX_COMPILATION_CACHE_DIR
export VLLM_NODELIST
export VLLM_NUM_PROCESSES="${#VLLM_HOSTS[@]}"
export VLLM_CONTROLLER_ADDR
export RAY_HEAD_HOST
export RAY_HEAD_IP
export RAY_PORT
export RAY_CLIENT_SERVER_PORT
export RAY_PROT_ADDRESS
export N_GPUS_PER_NODE
export NSYS_PROFILE_NAME
export NSYS_OUTPUT
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

# If debugging, show collected GRPO_* overrides
if [[ "$DEBUG" == "true" ]]; then
  echo "GRPO_* env overrides:"
  env | grep '^GRPO_' || true
fi

# Per-segment CPU allocation
export CPUS_PER_TASK_GATEWAY=4
export CPUS_PER_TASK_VLLM_CONTROLLER=4
export CPUS_PER_TASK_JAX=$((SLURM_CPUS_ON_NODE - CPUS_PER_TASK_GATEWAY))
export CPUS_PER_TASK_RAY=$((SLURM_CPUS_ON_NODE - CPUS_PER_TASK_VLLM_CONTROLLER))

# Warm-up: create persistent containers on all nodes

srun --label --unbuffered -K0 --kill-on-bad-exit=1 --mpi=none \
  --ntasks-per-node=1 \
  --cpus-per-task=${SLURM_CPUS_ON_NODE} \
  --container-name="${CONTAINER_NAME}" \
  --container-image="${CONTAINER_IMAGE}" --container-mounts="${MOUNTS:-}" --container-writable \
  --export=ALL hostname

# Coordinator gateway on the first JAX node (no GPU required)

srun --nodes=1 --ntasks=1 --cpus-per-task=${CPUS_PER_TASK_GATEWAY} -w "$JAX_COORDINATOR_ADDR" \
  --label --unbuffered \
  --container-name="${CONTAINER_NAME}" \
  --container-image="${CONTAINER_IMAGE}" --container-mounts="${MOUNTS:-}" --container-writable \
  --export=ALL,CUDA_VISIBLE_DEVICES= \
  bash -lc '
  python -u ../jax_inference_offloading/controller/gateway.py |& tee "${OUTPUT_DIR}/gateway.log"
  ' &
PIDS+=($!)

# Launch JAX (multi-node) - GRPO trainer
if [[ "${JAX_USE_DUMMY_WEIGHTS}" == "true" ]]; then
  JAX_MODEL_PATH=""
else
  if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "Error: --jax-use-real-weights requires --model-path=PATH" >&2
    exit 1
  fi
  JAX_MODEL_PATH="${MODEL_PATH}"
fi
CUDA_VISIBLE_DEVICES="${JAX_LOCAL_DEVICE_IDS}" \
MODEL_PATH="${JAX_MODEL_PATH}" \
srun --label --unbuffered -K0 --kill-on-bad-exit=1 --mpi=none \
  --nodes=${#JAX_HOSTS[@]} --ntasks=${#JAX_HOSTS[@]} --ntasks-per-node=1 -w "${JAX_HOSTS_CSV}" \
  --cpus-per-task=${CPUS_PER_TASK_JAX} \
  --container-name="${CONTAINER_NAME}" \
  --container-image="${CONTAINER_IMAGE}" --container-mounts="${MOUNTS:-}" --container-writable \
  --export=ALL \
  bash -lc 'maybe_run_nsys python -u trainer_grpo.py |& tee "${OUTPUT_DIR}/trainer-$(hostname -s).log" ' &
PIDS+=($!)

# Launch vLLM (multi-node)
# First, start Ray
CUDA_VISIBLE_DEVICES="${VLLM_VISIBLE_DEVICES}" \
srun --label --unbuffered -K0 --kill-on-bad-exit=1 --mpi=none \
  --nodes=${#VLLM_HOSTS[@]} --ntasks=${#VLLM_HOSTS[@]} --ntasks-per-node=1 -w "${VLLM_HOSTS_CSV}" \
  --cpus-per-task=${CPUS_PER_TASK_RAY}  \
  --container-name="${CONTAINER_NAME}" \
  --container-image="${CONTAINER_IMAGE}" --container-mounts="${MOUNTS:-}" --container-writable \
  --export=ALL \
  bash -lc '
  if [[ "`hostname -s`" == "${RAY_HEAD_HOST}" ]]; then
    ray start \
      --head \
      --port=${RAY_PORT} \
      --ray-client-server-port=${RAY_CLIENT_SERVER_PORT} \
      --num-cpus=${CPUS_PER_TASK_RAY} \
      --num-gpus=${N_GPUS_PER_NODE} \
      --block \
      --disable-usage-stats \
    |& tee "${OUTPUT_DIR}/ray-head.log"
  else
    ray start \
      --address="${RAY_HEAD_IP}:${RAY_PORT}" \
      --num-cpus=${CPUS_PER_TASK_RAY} \
      --num-gpus=${N_GPUS_PER_NODE} \
      --block \
    |& tee "${OUTPUT_DIR}/ray-worker-$(hostname -s).log"
  fi
  ' &
PIDS+=($!)

sleep 10  # wait for Ray to start

# Then start vLLM controller process
CUDA_VISIBLE_DEVICES= \
srun --label --unbuffered -K0 --kill-on-bad-exit=1 --mpi=none \
  --nodes=1 --ntasks=1 --ntasks-per-node=1 -w "${VLLM_CONTROLLER_ADDR}" \
  --cpus-per-task=${CPUS_PER_TASK_VLLM_CONTROLLER} \
  --container-name="${CONTAINER_NAME}" \
  --container-image="${CONTAINER_IMAGE}" --container-mounts="${MOUNTS:-}" --container-writable \
  --export=ALL \
  bash -lc '
  ray status
  python -u rollout.py |& tee "${OUTPUT_DIR}/rollout-$(hostname -s).log"
  ' &
PIDS+=($!)

wait "${PIDS[@]}"
