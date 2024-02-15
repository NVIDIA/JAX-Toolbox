#!/bin/bash
set -ex -o pipefail

# Repo for newer nsight versions
UBUNTU_ARCH=$(dpkg --print-architecture)
UBUNTU_VERSION=$(. /etc/os-release && echo ${ID}${VERSION_ID/./}) # e.g. ubuntu2204
DEVTOOLS_URL=https://developer.download.nvidia.com/devtools/repos/${UBUNTU_VERSION}/${UBUNTU_ARCH}
curl -o /usr/share/keyrings/nvidia.pub "${DEVTOOLS_URL}/nvidia.pub"
echo "deb [signed-by=/usr/share/keyrings/nvidia.pub] ${DEVTOOLS_URL}/ /" > /etc/apt/sources.list.d/devtools-${UBUNTU_VERSION}-${UBUNTU_ARCH}.list

export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles

apt-get update
apt-get install -y nsight-compute nsight-systems-cli
apt-get clean

rm -rf /var/lib/apt/lists/*

# "Wrong event order has been detected when adding events to the collection"
# workaround during nsys report post-processing with 2024.1.1 and CUDA 12.3
NSYS202411=/opt/nvidia/nsight-systems-cli/2024.1.1
if [[ -d "${NSYS202411}" ]]; then
  LIBCUPTI123=/opt/nvidia/nsight-compute/2023.3.0/host/target-linux-x64/libcupti.so.12.3
  if [[ ! -f "${LIBCUPTI123}" ]]; then
    echo "2024.1.1 workaround expects to be running inside 12.3.0 container"
    exit 1
  fi
  # Use libcupti.so.12.3 because this is a CUDA 12.3 container
  ln -s "${LIBCUPTI123}" "${NSYS202411}/target-linux-x64/libcupti.so.12.3"
  mv "${NSYS202411}/target-linux-x64/libcupti.so.12.4" "${NSYS202411}/target-linux-x64/_libcupti.so.12.4"
fi
