#!/bin/bash

set -ex

# Update distro
apt-get update

# Install required packages
apt-get install -y curl

# clean up all previously installed library to avoid conflicts
# while installing Amazon EFA version
dpkg --purge efa-config efa-profile libfabric openmpi \
             ibacm ibverbs-providers ibverbs-utils infiniband-diags \
             libibmad-dev libibmad5 libibnetdisc-dev libibnetdisc5 \
             libibumad-dev libibumad3 libibverbs-dev libibverbs1 librdmacm-dev \
             librdmacm1 rdma-core rdmacm-utils

# Download Amazon EFA package and install
EFA_INSTALLER_VERSION=latest
WORKDIR=$(mktemp -d)

pushd ${WORKDIR}

AMAZON_EFA_LINK="https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
curl -O "$AMAZON_EFA_LINK" 
tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && cd aws-efa-installer
./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify

popd

# check the installation is successful
/opt/amazon/efa/bin/fi_info --version

# Clean up
apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf ${WORKDIR}
