#!/bin/bash
set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-default}"
JOBSET_NAME="jax-vllm-multinode"
JOBSET_YAML="jio-jobset.yaml"

echo "Deploying JobSet: ${JOBSET_NAME}..."
kubectl apply -f "${JOBSET_YAML}" -n "${NAMESPACE}"
echo  "JobSet created successfully"
