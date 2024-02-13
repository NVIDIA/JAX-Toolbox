#!/bin/bash
set -ex -o pipefail

# Repo for newer nsight versions
export UBUNTU_ARCH=$(dpkg --print-architecture)
export UBUNTU_VERSION=$(. /etc/os-release && echo ${ID}${VERSION_ID/./}) # e.g. ubuntu2204
curl -o /usr/share/keyrings/nvidia.pub https://developer.download.nvidia.com/devtools/repos/${UBUNTU_VERSION}/${UBUNTU_ARCH}/nvidia.pub
echo "deb [signed-by=/usr/share/keyrings/nvidia.pub] https://developer.download.nvidia.com/devtools/repos/${UBUNTU_VERSION}/${UBUNTU_ARCH}/ /" > /etc/apt/sources.list.d/devtools-${UBUNTU_VERSION}-${UBUNTU_ARCH}.list

export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles

apt-get update
apt-get install -y nsight-compute nsight-systems-cli
apt-get clean

rm -rf /var/lib/apt/lists/*
