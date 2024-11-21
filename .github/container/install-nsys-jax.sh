#!/bin/bash
set -exo pipefail

SHA="$1"
if [[ ! $SHA =~ ^[0-9a-f]{40}$ ]]; then
  echo "$0: <SHA of JAX-Toolbox>"
  exit 1
fi

# This script gets installed via pip-finalize.sh eventually as nsys-jax-patch-nsys, but
# it's cleaner to include the result in the base image, so run it eagerly here.
python /opt/patch_nsys.py

# Install extra dependencies needed for `nsys recipe ...` commands. These are
# used by the nsys-jax wrapper script.
NSYS_DIR=$(dirname $(realpath $(command -v nsys)))
ln -s ${NSYS_DIR}/python/packages/nsys_recipe/requirements/common.txt /opt/pip-tools.d/requirements-nsys-recipe.in

# Install the jax-nsys package, which includes nsys-jax, nsys-jax-combine,
# install-protoc (called from pip-finalize.sh), and nsys-jax-patch-nsys as well as the
# jax_nsys Python library.
URL="git+https://github.com/NVIDIA/JAX-Toolbox.git@${SHA}#subdirectory=.github/container/jax_nsys&egg=jax-nsys"
echo "-e '${URL}'" > /opt/pip-tools.d/requirements-nsys-jax.in
