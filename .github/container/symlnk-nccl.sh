#!/bin/bash
set -euo pipefail

# Create a prefix with include/ and lib/ directories containing symlinks to the NCCL
# version installed at the system level; this is useful to pass to XLA to avoid it
# fetching its own copy.
prefix=/opt/nvidia/nccl
if [[ -d "${prefix}" ]]; then
  echo "Skipping link farm creation"
  exit 0
fi
arch=$(uname -m)-linux-gnu
nccl_packages=$(dpkg -l 'libnccl*' | awk '/^ii/ {print $2}')

if [[ -z "${nccl_packages}" ]]; then
  echo "No NCCL packages installed."
  exit 1
fi

for nccl_file in $(dpkg -L ${nccl_packages} | sort -u); do
  # Real files and symlinks are linked into $prefix
  if [[ -f "${nccl_file}" || -h "${nccl_file}" ]]; then
    # Replace /usr with $prefix and remove arch-specific lib directories
    nosysprefix="${nccl_file#"/usr/"}"
    noarchlib="${nosysprefix/#"lib/${arch}"/lib}"
    link_name="${prefix}/${noarchlib}"
    link_dir=$(dirname "${link_name}")
    mkdir -p "${link_dir}"
    ln -s "${nccl_file}" "${link_name}"
  else
    echo "Skipping ${nccl_file}"
  fi
done
