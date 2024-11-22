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
apt-get install -y nsight-compute nsight-systems-cli-2024.6.1
apt-get clean

rm -rf /var/lib/apt/lists/*

for NSYS in /opt/nvidia/nsight-systems-cli/2024.5.1 /opt/nvidia/nsight-systems-cli/2024.6.1; do
  if [[ -d "${NSYS}" ]]; then
    # * can match at least sbsa-armv8 and x86
    (cd ${NSYS}/target-linux-*/python/packages && git apply < /opt/nvidia/nsys-2024.5-tid-export.patch)
  fi
done

# Install extra dependencies needed for `nsys recipe ...` commands. These are
# used by the nsys-jax wrapper script.
ln -s $(dirname $(realpath $(command -v nsys)))/python/packages/nsys_recipe/requirements/common.txt /opt/pip-tools.d/requirements-nsys-recipe.in
