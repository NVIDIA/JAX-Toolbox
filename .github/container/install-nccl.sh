#!/bin/bash

set -ex -o pipefail

export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles


# Skip NCCL installation if both JAX_NCCL_VERSION (user defined) and
# NCCL_VERSION (defined in nvidia/cuda containers) are unset.
# This case means that the base container is built from a custom image with
# a custom network communicator or unset NCCL_VERSION env variable.
if [[ -z $JAX_NCCL_VERSION && -z $NCCL_VERSION ]]; then
    echo "Skip NCCL installation"
    exit 0
fi

JAX_NCCL_VERSION=${JAX_NCCL_VERSION:-$NCCL_VERSION}

apt-get update

# Extract CUDA version from `nvcc --version` output line
# Input: "Cuda compilation tools, release X.Y, VX.Y.Z"
# Output: X.Y
cuda_version=$(nvcc --version | sed -n 's/^.*release \([0-9]*\.[0-9]*\).*$/\1/p')

# Find latest NCCL version compatible with existing CUDA by matching
# ${cuda_version} in the package version string
libnccl2_version=$(apt-cache show libnccl-dev | sed -n "s/^Version: \(${JAX_NCCL_VERSION}.*+cuda${cuda_version}\)$/\1/p" | head -n 1)
libnccl_dev_version=$(apt-cache show libnccl-dev | sed -n "s/^Version: \(${JAX_NCCL_VERSION}.*+cuda${cuda_version}\)$/\1/p" | head -n 1)
if [[ -z "${libnccl2_version}" || -z "${libnccl_dev_version}" ]]; then
    echo "Could not find compatible NCCL version for CUDA ${cuda_version}"
    exit 1
fi

apt-get install -y --allow-change-held-packages \
    libnccl2=${libnccl2_version} \
    libnccl-dev=${libnccl_dev_version}

dpkg -s libnccl2 libnccl-dev

apt-get clean
rm -rf /var/lib/apt/lists/*