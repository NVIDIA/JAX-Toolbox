#!/bin/bash
#
# Usage: ./install.sh [optional arguments to virtualenv]
#
# If it doesn't already exist, this creates a virtual environment named
# `nsys_jax_env` in the current directory and installs Jupyter Lab and the
# dependencies of the Analysis.ipynb notebook that is shipped alongside this
# script inside the output archives of the `nsys-jax` wrapper.
#
# The expectation is that those archives will be copied and extracted on a
# laptop or workstation, and this installation script will be run there, while
# the `nsys-jax` wrapper is executed on a remote GPU cluster.
set -ex
SCRIPT_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
VIRTUALENV="${SCRIPT_DIR}/nsys_jax_venv"
if [[ ! -d "${VIRTUALENV}" ]]; then
  # Let `virtualenv` find/choose a Python. Currently >=3.10 is supported.
  virtualenv -p 3.13 -p 3.12 -p 3.11 -p 3.10 "$@" "${VIRTUALENV}"
  . "${VIRTUALENV}/bin/activate"
  python -m pip install -U pip
  # FIXME: install from JAX-Toolbox GitHub? include [jupyter] variant?
  python -m pip install -e "${SCRIPT_DIR}/python/jax_nsys[jupyter]"
  install-flamegraph "${VIRTUALENV}"
  install-protoc "${VIRTUALENV}"
else
  echo "Virtual environment already exists, not installing anything..."
fi
if [ -z ${NSYS_JAX_INSTALL_SKIP_LAUNCH+x} ]; then
  # TODO: point to jax_nsys/analysis/Analysis.ipynb
  echo "Launching: cd ${SCRIPT_DIR} && ${VIRTUALENV}/bin/python -m jupyterlab Analysis.ipynb"
  cd "${SCRIPT_DIR}" && "${VIRTUALENV}/bin/python" -m jupyterlab Analysis.ipynb
else
  echo "Skipping launch of jupyterlab due to NSYS_JAX_INSTALL_SKIP_LAUNCH"
fi
