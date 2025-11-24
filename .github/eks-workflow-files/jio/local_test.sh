#!/bin/bash
set -euo pipefail

# Configuration
N_NODES_JAX=${N_NODES_JAX:-1}
N_NODES_VLLM=${N_NODES_VLLM:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NAMESPACE="default"
IMAGE=941377147396.dkr.ecr.us-east-1.amazonaws.com/sbosisio/jio:latest #"ghcr.io/nvidia/jax-toolbox-internal:19496638418-jio-amd64"

echo "==================================================================="
echo "Configuring JIO JobSet on Kubernetes"
echo "==================================================================="
echo "JAX nodes: ${N_NODES_JAX}"
echo "vLLM nodes: ${N_NODES_VLLM}"
echo "GPUs per node: ${N_GPUS_PER_NODE}"
echo "Namespace: ${NAMESPACE}"
echo "==================================================================="

# Calculate derived values
VLLM_TENSOR_PARALLEL_SIZE=$((N_GPUS_PER_NODE * N_NODES_VLLM))
RAY_WORKERS=$((N_NODES_VLLM - 1))  # Subtract head node

# If ray workers is 0, set to 0 (single node vLLM)
if [ $RAY_WORKERS -lt 0 ]; then
  RAY_WORKERS=0
fi

echo "Calculated: VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE}, RAY_WORKERS=${RAY_WORKERS}"

# 1. Create RBAC (Service Account, Role, RoleBinding)
echo ""
echo "Step 1: Creating RBAC for JAX pod discovery..."
kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: jio-job-sa
  namespace: ${NAMESPACE}
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: jio-ray-sa
  namespace: ${NAMESPACE}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: jio-pod-reader
  namespace: ${NAMESPACE}
rules:
  - apiGroups: [""]
    resources: ["pods", "services"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["jobset.x-k8s.io"]
    resources: ["jobsets"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list", "watch"]  # ADD THIS LINE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: jio-configmap-manager
  namespace: ${NAMESPACE}
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list", "watch", "create", "update", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: jio-pod-reader-binding
  namespace: ${NAMESPACE}
subjects:
  - kind: ServiceAccount
    name: jio-job-sa
    namespace: ${NAMESPACE}
roleRef:
  kind: Role
  name: jio-pod-reader
  apiGroup: rbac.authorization.k8s.io
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: jio-ray-configmap-binding
  namespace: ${NAMESPACE}
subjects:
  - kind: ServiceAccount
    name: jio-ray-sa
    namespace: ${NAMESPACE}
roleRef:
  kind: Role
  name: jio-configmap-manager
  apiGroup: rbac.authorization.k8s.io
EOF
echo "RBAC created"

# 2. Create HF token secret (if HF_TOKEN is set)
echo ""
echo "Step 2: Creating secrets..."
if [[ -n "${HF_TOKEN:-}" ]]; then
  kubectl create secret generic huggingface-token \
    --from-literal=token="${HF_TOKEN}" \
    --namespace=${NAMESPACE} \
    --dry-run=client -o yaml | kubectl apply -f -
  echo "HuggingFace token secret created"
else
  echo "Warning: HF_TOKEN not set, skipping token secret creation"
fi

# 3. Create ConfigMap with entrypoint script
echo ""
echo "Step 3: Creating entrypoint script ConfigMap..."
if [[ ! -f "./example.sh" ]]; then
  echo "ERROR: example.sh not found in current directory"
  exit 1
fi
kubectl create configmap jio-entrypoint \
  --from-file=example.sh=./example.sh \
  --namespace=${NAMESPACE} \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap jio-entrypoint \
  --from-file=example.sh=./example.sh \
  --namespace=${NAMESPACE} \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap jio-rollout \
  --from-file=rollout.py=./rollout.py \
  --namespace=${NAMESPACE} \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Entrypoint and Rollout ConfigMaps created ✓"
# 4. Create configuration ConfigMap
echo ""
echo "Step 4: Creating configuration ConfigMap..."
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: jio-config
  namespace: ${NAMESPACE}
data:
  N_GPUS_PER_NODE: "${N_GPUS_PER_NODE}"
  JAX_COORDINATOR_PORT: "12345"
  GATEWAY_PORT: "50051"
  RAY_PORT: "20527"
  RAY_CLIENT_SERVER_PORT: "24430"
  MODEL_NAME: "meta-llama/Llama-3.1-8B-Instruct"
  TRANSFER_MODE: "grouped"
  USE_POLYMORPHIC_MESH: "0"
  VLLM_ENFORCE_EAGER: "1"
  VLLM_LOAD_FORMAT: "dummy"
  VLLM_GPU_MEMORY_UTILIZATION: "0.7"
  VLLM_DISTRIBUTED_BACKEND: "ray"
  JAX_USE_DUMMY_WEIGHTS: "true"
  DEBUG: "false"
  RAY_CPUS_PER_NODE: "160"
  CPU_PER_TASK_RAY: "160"
EOF
echo "Configuration ConfigMap created"

# 5. Deploy JobSet with dynamic parameters
echo ""
echo "Step 5: Deploying JobSet..."
if [[ ! -f "./jio-jobset.yaml" ]]; then
  echo "ERROR: jio-jobset.yaml not found in current directory"
  exit 1
fi


cat jio-jobset.yaml | \
  sed "s/namespace: jio-training/namespace: ${NAMESPACE}/g" | \
  sed "s/subdomain: jio-training/subdomain: ${NAMESPACE}/g" | \
  sed "s/parallelism: 2  # N_NODES_JAX/parallelism: ${N_NODES_JAX}  # N_NODES_JAX/" | \
  sed "s/completions: 2  # N_NODES_JAX/completions: ${N_NODES_JAX}  # N_NODES_JAX/" | \
  sed "s/value: \"2\"  # Must match completions above/value: \"${N_NODES_JAX}\"  # Must match completions above/" | \
  sed "s/parallelism: 1  # N_NODES_VLLM - 1/parallelism: ${RAY_WORKERS}  # N_NODES_VLLM - 1/" | \
  sed "s/completions: 1  # N_NODES_VLLM - 1/completions: ${RAY_WORKERS}  # N_NODES_VLLM - 1/" | \
  sed "s/value: \"16\"  # N_GPUS_PER_NODE \* N_NODES_VLLM/value: \"${VLLM_TENSOR_PARALLEL_SIZE}\"  # N_GPUS_PER_NODE * N_NODES_VLLM/" | \
  kubectl apply -f -

echo "JobSet deployed"
