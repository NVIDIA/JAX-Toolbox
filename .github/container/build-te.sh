#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
set -eo pipefail

## Parse command-line arguments
usage() {
    echo "Configure, build, and install TransformerEngine"
    echo ""
    echo "  Usage: $0 [OPTIONS]"
    echo ""
    echo "    OPTIONS                        DESCRIPTION"
    echo "    --clean                        Clear build caches under --src-path-te."
    echo "    -h, --help                     Print usage."
    echo "    --no-install                   Only build a wheel; do not install."
    echo "    --src-path-te                  Path to TransformerEngine source code."
    echo "    --src-path-xla                 Path to XLA source code."
    echo "    --sm SM1,SM2,...               Comma-separated list of CUDA SM versions"
    echo "                                   to compile for, e.g. 7.5,8.0 -- PTX will"
    echo "                                   only be emitted for the last one."
    echo "    --sm local                     Compile for the local GPUs (default)."
    echo "    --sm all                       Compile for a default set of SM versions."
    exit $1
}

# Set defaults
CLEAN=0
INSTALL=1
SM="local"
SRC_PATH_TE="/opt/transformer-engine"
SRC_PATH_XLA="/opt/xla"

args=$(getopt -o h --long clean,help,no-install,src-path-te:,src-path-xla:,sm: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi

eval set -- "$args"
while [ : ]; do
    case "$1" in
        --clean)
            CLEAN=1
            shift 1
            ;;
        -h | --help)
            usage 1
            ;;
        --no-install)
            INSTALL=0
            shift 1
            ;;
        --src-path-te)
            SRC_PATH_TE=$(realpath $2)
            shift 2
            ;;
        --src-path-xla)
            SRC_PATH_XLA=$(realpath $2)
            shift 2
            ;;
        --sm)
            SM=$2
            shift 2
            ;;
        --)
            shift;
            break
            ;;
        *)
            echo "UNKNOWN OPTION $1"
            usage 1
    esac
done

# Return a default list of CUDA SM architectures to compile for, in 1.2,3.4,5.6 format.
# If CUDA_ARCH_LIST is set, respect that, otherwise use hard-coded cpu-arch-specific lists.
default_compute_capabilities() {
    CPU_ARCH="$(dpkg --print-architecture)"
    # Infer the compute capabilities from the CUDA_ARCH_LIST variable if it is set;
    # this is in 1.2 3.4 5.6 format
    if [[ -n "${CUDA_ARCH_LIST}" ]]; then
        echo ${CUDA_ARCH_LIST// /,}
    elif [[ "${CPU_ARCH}" == "amd64" ]]; then
        echo "7.5,8.0,8.6,9.0,10.0,12.0"
    elif [[ "${CPU_ARCH}" == "arm64" ]]; then
        echo "8.0,8.6,9.0,10.0,12.0"
    else
        echo "Invalid arch '$CPU_ARCH' (expected 'amd64' or 'arm64')" 1>&2
        return 1
    fi
}

print_var() {
    echo "$1: ${!1}"
}

clean() {
    pushd "${SRC_PATH_TRANSFORMER_ENGINE}"
    rm -rf build/ .eggs/
    popd
}

# This should standardise on 1.2,3.4,5.6 format
if [[ "$SM" == "all" ]]; then
    SM_LIST=$(default_compute_capabilities)
elif [[ "$SM" == "local" ]]; then
    SM_LIST=$("${SCRIPT_DIR}/local_cuda_arch")
else
    SM_LIST=${SM}
fi

## Print info
echo "=================================================="
echo "                  Configuration                   "
echo "--------------------------------------------------"
print_var CLEAN
print_var INSTALL
print_var SM
print_var SM_LIST
print_var SRC_PATH_TE
print_var SRC_PATH_XLA
echo "=================================================="

# Parse SM_LIST into the format accepted by TransformerEngine's build system
# "1.2,3.4,5.6" -> "12-real;34-real;56", i.e. SASS plus PTX for the last one
NVTE_CUDA_ARCHS="${SM_LIST//,/-real;}"
set -x
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS//./}"
# Parallelism within nvcc invocations.
export NVTE_BUILD_THREADS_PER_JOB=8
export NVTE_FRAMEWORK=jax
# TransformerEngine needs FFI headers from XLA
export XLA_HOME=${SRC_PATH_XLA}

pushd ${SRC_PATH_TRANSFORMER_ENGINE}
# Install required packages that were removed in https://github.com/NVIDIA/TransformerEngine/pull/1852
pip install "pybind11[global]"

# The wheel filename includes the TE commit; if this has changed since the last
# incremental build then we would end up with multiple wheels.
rm -fv dist/*.whl
python setup.py bdist_wheel
ls dist/
popd

## Install the built packages
if [[ "${INSTALL}" == "1" ]]; then
    pip install ${SRC_PATH_TRANSFORMER_ENGINE}/dist/*.whl
    pip list | grep ^transformer_engine
fi

## Cleanup
if [[ "$CLEAN" == "1" ]]; then
    clean
fi
