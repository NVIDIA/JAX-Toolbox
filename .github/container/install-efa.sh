#!/bin/bash
set -ex

EFA_INSTALLER_VERSION=1.34.0 # or: latest
AWS_OFI_NCCL_PREFIX=/opt/aws-ofi-nccl
AWS_OFI_NCCL_VERSION=1.11.0

apt update

EFA_TMP=$(mktemp -d)
pushd $EFA_TMP
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz
tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz
cd aws-efa-installer
rm -v DEBS/UBUNTU2204/x86_64/{libpmix,openmpi,prrte}* # block installation of MPI components
apt-get purge -y ibverbs-providers libibverbs-dev libibverbs1 libibumad-dev libibumad3 librdmacm1 librdmacm-dev ibverbs-utils
./efa_installer.sh -g -y --skip-kmod --skip-limit-conf --no-verify |& tee install.log
mv -v install.log /opt/amazon/efa/install.log
popd
rm -rf $EFA_TMP

AWS_OFI_NCCL_TMP=$(mktemp -d)
pushd $AWS_OFI_NCCL_TMP
apt-get install -y libhwloc-dev
curl -OL https://github.com/aws/aws-ofi-nccl/releases/download/v${AWS_OFI_NCCL_VERSION}-aws/aws-ofi-nccl-${AWS_OFI_NCCL_VERSION}-aws.tar.gz
tar -xf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION}-aws.tar.gz
cd aws-ofi-nccl-${AWS_OFI_NCCL_VERSION}-aws
./configure --prefix=${AWS_OFI_NCCL_PREFIX} --with-libfabric=/opt/amazon/efa --with-cuda=/usr/local/cuda --with-mpi=/usr/local/mpi
make -j$(nproc) install
popd
rm -rf $AWS_OFI_NCCL_TMP

rm -rf /var/lib/apt/lists/*

# Ranks higher than HPC-X => newly-installed libnccl-net.so becomes the default
echo "${AWS_OFI_NCCL_PREFIX}/lib" > /etc/ld.so.conf.d/000_aws_ofi_nccl.conf
ldconfig
