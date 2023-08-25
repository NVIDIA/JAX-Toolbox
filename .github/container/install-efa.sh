#!/bin/bash

set -ex

# Update distro
apt-get update

# Install required packages
apt-get install -y curl

# Hack to install Amazon EFA correctly: for some reason the base docker 
# image has incorrecly installed <libibverbs-dev>, as a result libefa* libraries
# cannot be found under /usr/lib/x86_64-linux-gnu/
# Temporary workaround: reinstall libibverbs-dev

apt-get remove -y libibverbs-dev ibverbs-providers libibverbs1
apt-get install -y libibverbs-dev

# Download Amazon EFA package and install
EFA_INSTALLER_VERSION=latest
WORKDIR=$(mktemp -d)

pushd ${WORKDIR}

AMAZON_EFA_LINK="https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
curl -O "$AMAZON_EFA_LINK" 
tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && cd aws-efa-installer
./efa_installer.sh -y --skip-kmod

popd

# check the installation is successful
/opt/amazon/efa/bin/fi_info --version

LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
PATH=/opt/amazon/efa/bin:$PATH

# Clean up
apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf ${WORKDIR}
