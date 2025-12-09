#!/bin/bash
set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-default}"
JOBSET_NAME="jax-vllm-multinode"
DELETE_SECRETS="${DELETE_SECRETS:-false}"
FORCE="${FORCE:-false}"

if ! kubectl get jobset "${JOBSET_NAME}" -n "${NAMESPACE}" >/dev/null 2>&1; then
    echo "JobSet '${JOBSET_NAME}' not found in namespace '${NAMESPACE}'"
return 0
fi
echo "Deleting JobSet: ${JOBSET_NAME}..."
kubectl delete jobset "${JOBSET_NAME}" -n "${NAMESPACE}" --wait=false

echo "Waiting for pods to terminate (max 120s)..."
kubectl wait --for=delete pod \
    -l jobset.sigs.k8s.io/jobset-name="${JOBSET_NAME}" \
    -n "${NAMESPACE}" \
    --timeout=120s 2>/dev/null || log_warn "Some pods may still be terminating"

echo "JobSet deleted"
