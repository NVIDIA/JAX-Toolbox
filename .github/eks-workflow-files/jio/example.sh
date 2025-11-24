#!/bin/bash
set -euo pipefail
ROLE="${K8S_ROLE:-unknown}"

# Setup
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_DEVICE_MAX_CONNECTIONS=16
export NCCL_CUMEM_ENABLE=0
export NCCL_BUFFSIZE=16777216

# Load .env if present
if [[ -f "/workspace/.env" ]]; then
  echo "Loading environment variables from .env file"
  set -a && source "/workspace/.env" && set +a
fi

# Validate HF_TOKEN
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set."
fi

# XLA Flags
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
                  --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUDNN,CUSTOM_CALL \
                  --xla_gpu_collective_permute_combine_threshold_bytes=8589934592 \
                  --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 \
                  --xla_gpu_all_gather_combine_threshold_bytes=8589934592 \
                  --xla_gpu_all_reduce_combine_threshold_bytes=8589934592"

echo "Starting role: ${ROLE}"

case "${ROLE}" in
  gateway)
    echo "=== Starting Gateway ==="
    export CUDA_VISIBLE_DEVICES=
    python -u /opt/jtbx/jax-inference-offloading/jax_inference_offloading/controller/gateway.py |& tee "${OUTPUT_DIR}/gateway.log"
    ;;

  jax-trainer)
    echo "=== Starting JAX Trainer ==="

    # Determine process rank from StatefulSet pod index
    if [[ -n "${JOB_COMPLETION_INDEX:-}" ]]; then
      PROCESS_ID="${JOB_COMPLETION_INDEX}"
    elif [[ -n "${POD_NAME:-}" ]]; then
      PROCESS_ID=$(echo "${POD_NAME}" | grep -oE '[0-9]+$' || echo "0")
    else
      PROCESS_ID=0
    fi

    export JAX_PROCESS_INDEX="${PROCESS_ID}"
    export JAX_NUM_PROCESSES="${JAX_REPLICAS}"
    export JAX_COORDINATOR_ADDRESS="${JAX_COORDINATOR_ADDR}:${JAX_COORDINATOR_PORT}"

    # GPU assignment
    export CUDA_VISIBLE_DEVICES="${JAX_LOCAL_DEVICE_IDS:-0,1,2,3,4,5,6,7}"
    export JAX_LOCAL_DEVICE_IDS="${JAX_LOCAL_DEVICE_IDS:-0,1,2,3,4,5,6,7}"

    # Model path handling
    if [[ "${JAX_USE_DUMMY_WEIGHTS:-true}" == "true" ]]; then
      export MODEL_PATH=""
    fi
    echo "JAX Trainer starting: process ${JAX_PROCESS_INDEX}/${JAX_NUM_PROCESSES}"
    ${PROF_CMD} python -u trainer.py |& tee "${OUTPUT_DIR}/trainer-${POD_NAME}.log"
    ;;

  ray-head)
    echo "=== Starting Ray Head ==="
    export CUDA_VISIBLE_DEVICES="${VLLM_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

    # Get pod IP address for Ray to bind to
    POD_IP=$(hostname -i)
    echo "Pod IP: ${POD_IP}"

    # Count GPUs
    GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    echo "Starting Ray head with ${GPU_COUNT} GPUs: ${CUDA_VISIBLE_DEVICES}"

    ray start \
      --head \
      --port=${RAY_PORT} \
      --ray-client-server-port=${RAY_CLIENT_SERVER_PORT} \
      --num-cpus=${RAY_CPUS_PER_NODE:-64} \
      --num-gpus=${GPU_COUNT} \
      --block \
      --disable-usage-stats \
      --temp-dir=/tmp/ray \
    |& tee "${OUTPUT_DIR}/ray-head.log"
    ;;

  ray-worker)
    echo "=== Starting Ray Worker ==="
    export CUDA_VISIBLE_DEVICES="${VLLM_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

    # Get pod IP
    POD_IP=$(hostname -i)
    echo "Pod IP: ${POD_IP}"

    # Count GPUs
    GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    echo "Starting Ray worker with ${GPU_COUNT} GPUs: ${CUDA_VISIBLE_DEVICES}"

    # Wait for ray head to be ready
    echo "Waiting for Ray head at ${RAY_HEAD_IP}:${RAY_PORT}..."
    while ! nc -z "${RAY_HEAD_IP}" "${RAY_PORT}"; do
      sleep 2
    done

    ray start \
      --address="${RAY_HEAD_IP}:${RAY_PORT}" \
      --node-ip-address=${POD_IP} \
      --num-cpus=${RAY_CPUS_PER_NODE:-64} \
      --num-gpus=${N_GPUS_PER_NODE:-8} \
      --block \
    |& tee "${OUTPUT_DIR}/ray-worker-${POD_NAME}.log"
    ;;

  vllm-controller)
    echo "=== Starting vLLM Controller ==="
    export CUDA_VISIBLE_DEVICES=

    echo "==================================================================="
    echo "vLLM Controller Configuration:"
    echo "  Ray Head DNS: ${RAY_HEAD_IP}"
    echo "  Ray Head IP: ${RAY_HEAD_ACTUAL_IP}"
    echo "  Ray Port: ${RAY_PORT}"
    echo "  Model: ${MODEL_NAME}"
    echo "  Tensor Parallel Size: ${VLLM_TENSOR_PARALLEL_SIZE}"
    echo "  Gateway: ${GATEWAY_URL}"
    echo "==================================================================="

    # Set RAY_ADDRESS to the actual IP
    export RAY_ADDRESS="ray://${RAY_HEAD_ACTUAL_IP}:${RAY_PORT}"

    # Wait for Ray to be ready
    echo "Waiting for Ray cluster..."
    sleep 30

    echo "Modifying rollout"
    cat > /tmp/rollout_wrapper.py <<'EOF'
import logging
import os
import ray

logging.basicConfig(level=logging.INFO)

# Initialize Ray BEFORE importing rollout
ray_address = os.environ.get("RAY_ADDRESS")
logging.info(f"Pre-initializing Ray connection to {ray_address}...")
ray.init(address=ray_address, ignore_reinit_error=True, namespace="default")

resources = ray.cluster_resources()
logging.info(f"Connected to Ray cluster with resources: {resources}")

gpu_count = resources.get("GPU", 0)
tensor_parallel_size = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "8"))

if gpu_count < tensor_parallel_size:
    logging.error(f"Not enough GPUs! Need {tensor_parallel_size}, has {gpu_count}")
    raise RuntimeError("Insufficient GPUs")

logging.info(f"✅ Ray cluster has {int(gpu_count)} GPUs, proceeding with rollout...")

# Now run the original rollout.py
import sys
sys.path.insert(0, '/opt/jtbx/jax-inference-offloading/examples')
import rollout
rollout.main()
EOF
    # run vllm
    python rollout.py |& tee "${OUTPUT_DIR}/rollout.log"
    ;;

  *)
    echo "Unknown role: ${ROLE}"
    echo "Valid roles: gateway, jax-trainer, ray-head, ray-worker, vllm-controller"
    exit 1
    ;;
esac
