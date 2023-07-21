#!/bin/bash

set -ex

export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles

# Install libnl (Netlink Protocol Library Suite), which provides a low-level API
# for communication between kernel and user space processes in Linux. Essential for managing
# networking components such as routing tables, network interfaces, and address resolution.

apt-get update
apt-get install -y \
    curl \
    libnl-route-3-200 \
    libnl-3-dev \
    libnl-route-3-dev

# Download NVIDIA/Mellanox's OFED distribution and install

WORKDIR=$(mktemp -d)
pushd ${WORKDIR}

MLNX_OFED_LINK="https://content.mellanox.com/ofed/MLNX_OFED-23.04-1.1.3.0/MLNX_OFED_LINUX-23.04-1.1.3.0-ubuntu22.04-$(uname -i).tgz"
curl -s -L "${MLNX_OFED_LINK}" -o - | tar xz --no-anchored --wildcards 'DEBS/*' --strip-components=3

dpkg -i libibverbs1_*.deb \
        libibverbs-dev_*.deb \
        librdmacm1_*.deb \
        librdmacm-dev_*.deb \
        libibumad3_*.deb \
        libibumad-dev_*.deb \
        ibverbs-utils_*.deb \
        ibverbs-providers_*.deb

popd

# cleanup

apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf ${WORKDIR}
