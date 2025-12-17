#!/bin/bash
set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-default}"
JOBSET_NAME="jax-vllm-singlenode"
JOBSET_YAML="jio-singlenode.yaml"

echo "Deploying JobSet: ${JOBSET_NAME}..."
kubectl apply -f "${JOBSET_YAML}" -n "${NAMESPACE}"
echo  "JobSet created successfully"
