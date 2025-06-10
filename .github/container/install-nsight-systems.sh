#!/bin/bash
set -exuo pipefail

version=$1
if [[ -z "${version}" ]]; then
  echo "Usage: $0 major.minor.patch"
  exit 1
fi

# Remove the symlink that makes `nsys` refer to the CUDA-bundled version:
rm /usr/local/cuda/bin/nsys

# Repo for newer nsight versions
UBUNTU_ARCH=$(dpkg --print-architecture)
UBUNTU_VERSION=$(. /etc/os-release && echo ${ID}${VERSION_ID/./}) # e.g. ubuntu2204
DEVTOOLS_URL=https://developer.download.nvidia.com/devtools/repos/${UBUNTU_VERSION}/${UBUNTU_ARCH}
curl -o /usr/share/keyrings/nvidia.pub "${DEVTOOLS_URL}/nvidia.pub"
echo "deb [signed-by=/usr/share/keyrings/nvidia.pub] ${DEVTOOLS_URL}/ /" > /etc/apt/sources.list.d/devtools-${UBUNTU_VERSION}-${UBUNTU_ARCH}.list

export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles

apt-get update
apt-get install -y nsight-systems-cli-${version}
apt-get clean

rm -rf /var/lib/apt/lists/*
