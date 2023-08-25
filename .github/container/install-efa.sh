#!/bin/bash

function copy_libefa() {
    PACKAGE_NAME=$1
    EXTRACT_FOLDER="${PACKAGE_NAME}_"
    # unpack downloaded package and copy libefa* stuff to /usr/lib    
    dpkg -x ${PACKAGE_NAME}* ${EXTRACT_FOLDER}
    # copy all libefa stuff to /usr/lib
    for n in $(find ${EXTRACT_FOLDER}/ -name libefa*)
    do
        mkdir -p "$(dirname $(echo ${n} | cut -d'_' -f2-))"
        cp -a "${n}" $(echo ${n} | cut -d'_' -f2-)
    done
}

set -ex

# Update distro
apt-get update

# Workaround to have libefa* for Amazon EFA: 
# for some reason NVIDIA/Mellanox's OFED distribution does not contain libefa*
# Temporary solution:
# 1. download libibverbs-dev and ibverbs-providers that contains required libefa*
# 2. unpack them
# 3. copy libefa to /usr/lib

WORKDIR=$(mktemp -d)
pushd ${WORKDIR}

# just in case remove related packages that contains libefa
apt-get remove -y libibverbs-dev ibverbs-providers

# download deb-files without installation
apt-get download -y libibverbs-dev ibverbs-providers

copy_libefa libibverbs-dev
copy_libefa ibverbs-providers

popd

# Clean up
apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf ${WORKDIR}
