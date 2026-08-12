#!/bin/bash
set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-default}"
JOBSET_NAME="jax-vllm-multinode"
JOBSET_YAML="jio-jobset.yaml"
# Your dev image, e.g. <account>.dkr.ecr.<region>.amazonaws.com/<user>/jio:latest
JIO_IMAGE="${JIO_IMAGE:?Set JIO_IMAGE to the container image to deploy}"

echo "Deploying JobSet: ${JOBSET_NAME} with image ${JIO_IMAGE}..."
sed "s|JIO_IMAGE_PLACEHOLDER|${JIO_IMAGE}|g" "${JOBSET_YAML}" | kubectl apply -f - -n "${NAMESPACE}"
echo  "JobSet created successfully"
