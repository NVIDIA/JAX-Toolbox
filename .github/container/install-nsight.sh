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
