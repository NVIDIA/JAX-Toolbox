#!/bin/bash
set -exuo pipefail

# Create a prefix with bin/, include/ and lib/ directories containing symlinks
# to the NVSHMEM version installed at the system level; this is useful to pass
# to XLA to avoid it fetching its own copy.
prefix=/opt/nvidia/nvshmem
if [[ -d "${prefix}" ]]; then
  echo "$0: ${prefix} already exists -- this is not expected!"
  exit 0
fi
mkdir "${prefix}"
packages=$(dpkg -l 'libnvshmem*' | awk '/^ii/ {print $2}')

if [[ -z "${packages}" ]]; then
  echo "No NVSHMEM packages installed."
  exit 1
fi

for path in $(dpkg -L ${packages} | sort -u); do
  dirn=$(dirname "${path}")
  name=$(basename "${path}")
  # ${prefix}/lib points to the directory where libnvshmem_host.so is
  if [[ "${name}" == "libnvshmem_host.so" ]]; then
    ln -sv "${dirn}" "${prefix}/lib"
  # ${prefix}/include points to the directory where nvshmem.h is
  elif [[ "${name}" == "nvshmem.h" ]]; then
    ln -sv "${dirn}" "${prefix}/include"
  # ${prefix}/bin points to the directory where nvshmem-info is
  elif [[ "${name}" == "nvshmem-info" ]]; then
    ln -sv "${dirn}" "${prefix}/bin"
  fi
done
ls -l "${prefix}"
