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

# install AWS OFI NCCL plugin

export OFI_PREFIX=/opt/aws-ofi-nccl
AWS_OFI_NCCL_VERSION=1.9.2-aws
apt-get install -y libhwloc-dev
curl -OL https://github.com/aws/aws-ofi-nccl/releases/download/v${AWS_OFI_NCCL_VERSION}/aws-ofi-nccl-${AWS_OFI_NCCL_VERSION}.tar.gz
tar -xf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION}.tar.gz
pushd aws-ofi-nccl-${AWS_OFI_NCCL_VERSION}
./configure --prefix=${OFI_PREFIX} \
        --with-mpi=/opt/amazon/openmpi \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --enable-platform-aws
make -j $(nproc)
make install
popd
rm -rf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION}
rm aws-ofi-nccl-${AWS_OFI_NCCL_VERSION}.tar.gz
echo "PATH=${OFI_PREFIX}/bin:\$PATH" > /etc/profile.d/ofi-aws.sh
echo "${OFI_PREFIX}/lib" > /etc/ld.so.conf.d/000_ofi_aws.conf

# Clean up
apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf ${WORKDIR}
