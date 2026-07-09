#!/bin/bash
set -exuo pipefail

if [[ -z "${NVSHMEM_VERSION}" ]]; then
  echo "NVSHMEM_VERSION not set; do not know what to install; aborting..."
  exit 1
fi

# Repository for NVSHMEM
UBUNTU_ARCH=$(dpkg --print-architecture)
if [[ "${UBUNTU_ARCH}" == "arm64" ]]; then
  # nvshmem is only published for sbsa, not arm64
  REPO_ARCH="sbsa"
elif [[ "${UBUNTU_ARCH}" == "amd64" ]]; then
  REPO_ARCH="x86_64"
else
  echo "Do not know how to map ${UBUNTU_ARCH} onto a compute/cuda repository URL"
  exit 1
fi
UBUNTU_VERSION=$(. /etc/os-release && echo ${ID}${VERSION_ID/./}) # e.g. ubuntu2204
curl -o /tmp/keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/${REPO_ARCH}/cuda-keyring_1.1-1_all.deb"
dpkg -i /tmp/keyring.deb

export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles

apt-get update
apt-get install -y libnvshmem3{,-dev,-static}-cuda-${CUDA_VERSION:0:2}=${NVSHMEM_VERSION}*
apt-get clean

rm -rf /var/lib/apt/lists/* /tmp/keyring.deb
