#!/bin/bash

set -ex

CUDNN_MAJOR_VERSION=9

# Create a prefix with include/ and lib/ directories containing symlinks to the cuDNN
# version that was just installed; this is useful to pass to XLA to avoid it fetching
# its own copy of cuDNN.
prefix=/opt/nvidia/cudnn
if [[ -d "${prefix}" ]]; then
  echo "Skipping link farm creation"
  exit 1
fi

arch=$(uname -m)-linux-gnu
libcudnn_pkgs=$(dpkg -l 'libcudnn*' | awk '/^ii/ {print $2}')
if [[ -z "${libcudnn_pkgs}" ]]; then
  echo "No libcudnn packages installed."
  exit 1
fi

for cudnn_file in $(dpkg -L ${libcudnn_pkgs} | sort -u); do
  # Real files and symlinks are linked into $prefix
  if [[ -f "${cudnn_file}" || -h "${cudnn_file}" ]]; then
    # Replace /usr with $prefix
    nosysprefix="${cudnn_file#"/usr/"}"
    # include/x86_64-linux-gpu -> include/
    noarchinclude="${nosysprefix/#"include/${arch}"/include}"
    # cudnn_v9.h -> cudnn.h
    noverheader="${noarchinclude/%"_v${CUDNN_MAJOR_VERSION}.h"/.h}"
    # lib/x86_64-linux-gnu -> lib/
    noarchlib="${noverheader/#"lib/${arch}"/lib}"
    link_name="${prefix}/${noarchlib}"
    link_dir=$(dirname "${link_name}")
    mkdir -p "${link_dir}"
    ln -s "${cudnn_file}" "${link_name}"
  else
    echo "Skipping ${cudnn_file}"
  fi
done

# # replicate the original symlinks too, so we'll have /opt/nvidia/cudnn/include/cudnn.sh
# find /usr/include -maxdepth 1 -name "cudnn*.h" -type l | while read -r symlink; do
#   symlink_name=$(basename "${symlink}")
#   symlink_target=$(readlink "${symlink}")
#   # Check if the symlink points to x86_64-linux-gnu/
#   if [[ "${symlink_target}" == "${arch}/"* ]]; then
#     # Adjust the symlink target to point within our symlink directory
#     adjusted_target="${prefix}/include/${symlink_target#${arch}/}"
#     # Destination symlink within the symlink directory
#     link_name="${prefix}/include/${symlink_name}"
#     link_dir=$(dirname "${link_name}")
#     mkdir -p "${link_dir}"
#     # Check if the symlink already exists
#     if [[ -e "${link_name}" ]]; then
#       echo "Symlink ${link_name} already exists. Skipping."
#     else
#       ln -s "${adjusted_target}" "${link_name}"
#     fi
#   else
#     echo "Skipping symlink ${symlink} with target ${symlink_target}"
#   fi
# done