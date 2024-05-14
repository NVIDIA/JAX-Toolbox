#!/bin/bash

set -ex

export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles

CUDNN_MAJOR_VERSION=9

apt-get update

# Extract major CUDA version from `nvcc --version` output line
# Input: "Cuda compilation tools, release X.Y, VX.Y.Z"
# Output: X
cuda_major_version=$(nvcc --version | sed -n 's/^.*release \([0-9]*\.[0-9]*\).*$/\1/p' | cut -d. -f1)

# Find latest cuDNN version compatible with existing CUDA by matching
# ${cuda_major_version} in the package version string
# In most cases cuDNN release is behind CUDA ones. It is considered, that major 
# version of CUDA and cuDNN are compatible.
# For example, CUDA 12.3 + cuDNN 8.9.6 (libcudnn8 version: 8.9.6.50-1+cuda12.2) is 
# considered to be compatible.
libcudnn_version=$(apt-cache show libcudnn${CUDNN_MAJOR_VERSION} |  sed -n "s/^Version: \(.*+cuda${cuda_major_version}\.[0-9]*\)$/\1/p" | head -n 1)
libcudnn_dev_version=$(apt-cache show libcudnn${CUDNN_MAJOR_VERSION}-dev | sed -n "s/^Version: \(.*+cuda${cuda_major_version}\.[0-9]\)$/\1/p" | head -n 1)
if [[ -z "${libcudnn_version}" || -z "${libcudnn_dev_version}" ]]; then
    echo "Could not find compatible cuDNN version for CUDA ${cuda_version}"
    exit 1
fi

apt-get update
apt-get install -y \
    libcudnn${CUDNN_MAJOR_VERSION}=${libcudnn_version} \
    libcudnn${CUDNN_MAJOR_VERSION}-dev=${libcudnn_dev_version}

apt-get clean
rm -rf /var/lib/apt/lists/*
