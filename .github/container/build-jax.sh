#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
set -e

## Utility methods

print_var() {
    echo "$1: ${!1}"
}

supported_compute_capabilities() {
    ARCH=$1
    if [[ "${ARCH}" == "amd64" ]]; then
        echo "5.2,6.0,6.1,7.0,7.5,8.0,8.6,8.9,9.0,9.0a"
    elif [[ "${ARCH}" == "arm64" ]]; then
        echo "5.3,6.2,7.0,7.2,7.5,8.0,8.6,8.7,8.9,9.0,9.0a"
    else
        echo "Invalid arch '$ARCH' (expected 'amd64' or 'arm64')" 1>&2
        return 1
    fi
}

## Parse command-line arguments

usage() {
    echo "Configure, build, and install JAX and Jaxlib"
    echo ""
    echo "  Usage: $0 [OPTIONS]"
    echo ""
    echo "    OPTIONS                        DESCRIPTION"
    echo "    --bazel-cache URI              Path for local bazel cache or URL of remote bazel cache"
    echo "    --build-param PARAM            Param passed to the jaxlib build command. Can be passed many times."
    echo "    --build-path-jaxlib PATH       Editable install location for jaxlib"
    echo "    --clean                        Delete local configuration and bazel cache"
    echo "    --clean-only                   Do not build, just cleanup"
    echo "    --cpu-arch                     Target CPU architecture, e.g. amd64, arm64, etc."
    echo "    --debug                        Build in debug mode"
    echo "    --dry                          Dry run, parse arguments only"
    echo "    -h, --help                     Print usage."
    echo "    --jaxlib_only                  Only build and install jaxlib"
    echo "    --no-clean                     Do not delete local configuration and bazel cache (default)"
    echo "    --src-path-jax                 Path to JAX source"
    echo "    --src-path-xla                 Path to XLA source"
    echo "    --sm SM1,SM2,...               Comma-separated list of CUDA SM versions to compile for, e.g. '7.5,8.0'"
    echo "    --sm local                     Query the SM of available GPUs (default)"
    echo "    --sm all                       All current SM"
    echo "                                   If you want to pass a bazel parameter, you must do it like this:"
    echo "                                       --build-param=--bazel_options=..."
    exit $1
}

# Set defaults
BAZEL_CACHE=""
BUILD_PATH_JAXLIB="/opt/jaxlib"
BUILD_PARAM=""
CLEAN=0
CLEANONLY=0
CPU_ARCH="$(dpkg --print-architecture)"
CUDA_COMPUTE_CAPABILITIES="local"
DEBUG=0
DRY=0
JAXLIB_ONLY=0
SRC_PATH_JAX="/opt/jax"
SRC_PATH_XLA="/opt/xla"
XLA_ARM64_PATCH_LIST=""

args=$(getopt -o h --long bazel-cache:,build-param:,build-path-jaxlib:,clean,cpu-arch:,debug,jaxlib_only,no-clean,clean-only,dry,help,src-path-jax:,src-path-xla:,sm:,xla-arm64-patch: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi

eval set -- "$args"
while [ : ]; do
    case "$1" in
        --bazel-cache)
            BAZEL_CACHE=$2
            shift 2
            ;;
        --build-param)
            BUILD_PARAM="$BUILD_PARAM $2"
            shift 2
            ;;
        --build-path-jaxlib)
            BUILD_PATH_JAXLIB="$2"
            shift 2
            ;;
        -h | --help)
            usage 1
            ;;
        --clean)
            CLEAN=1
            shift 1
            ;;
        --clean-only)
            CLEANONLY=1
            shift 1
            ;;
        --cpu-arch)
            CPU_ARCH="$2"
            shift 2
            ;;
        --no-clean)
            CLEAN=0
            shift 1
            ;;
        --debug)
            DEBUG=1
            shift 1
            ;;
        --dry)
            DRY=1
            shift 1
            ;;
        --jaxlib_only)
            JAXLIB_ONLY=1
            shift 1
            ;;
        --src-path-jax)
            SRC_PATH_JAX=$2
            shift 2
            ;;
        --src-path-xla)
            SRC_PATH_XLA=$2
            shift 2
            ;;
        --sm)
            CUDA_COMPUTE_CAPABILITIES=$2
            shift 2
            ;;
        --xla-arm64-patch)
            XLA_ARM64_PATCH_LIST=$2
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

## Set internal variables

SRC_PATH_JAX=$(realpath $SRC_PATH_JAX)
SRC_PATH_XLA=$(realpath $SRC_PATH_XLA)

clean() {
    pushd "${SRC_PATH_JAX}"
    bazel clean --expunge || true
    rm -rf bazel
    rm -rf .jax_configure.bazelrc
    rm -rf ${HOME}/.cache/bazel
    popd
}

export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles

export TF_NEED_CUDA=1
export TF_NEED_CUTENSOR=1
export TF_NEED_TENSORRT=0
export TF_CUDA_PATHS=/usr,/usr/local/cuda
export TF_CUDNN_PATHS=/usr/lib/$(uname -p)-linux-gnu
export TF_CUDA_VERSION=$(ls /usr/local/cuda/lib64/libcudart.so.*.*.* | cut -d . -f 3-4)
export TF_CUBLAS_VERSION=$(ls /usr/local/cuda/lib64/libcublas.so.*.*.* | cut -d . -f 3)
export TF_CUDNN_VERSION=$(echo "${NV_CUDNN_VERSION}" | cut -d . -f 1)
export TF_NCCL_VERSION=$(echo "${NCCL_VERSION}" | cut -d . -f 1)

case "${CPU_ARCH}" in
    "amd64")
        export CC_OPT_FLAGS="-march=sandybridge -mtune=broadwell"
        ;;
    "arm64")
        export CC_OPT_FLAGS="-march=armv8-a"
        ;;
esac

if [[ ! -z "${CUDA_COMPUTE_CAPABILITIES}" ]]; then
    if [[ "$CUDA_COMPUTE_CAPABILITIES" == "all" ]]; then
        export TF_CUDA_COMPUTE_CAPABILITIES=$(supported_compute_capabilities ${CPU_ARCH})
        if [[ $? -ne 0 ]]; then exit 1; fi
    elif [[ "$CUDA_COMPUTE_CAPABILITIES" == "local" ]]; then
        export TF_CUDA_COMPUTE_CAPABILITIES=$("${SCRIPT_DIR}/local_cuda_arch")
    else
        export TF_CUDA_COMPUTE_CAPABILITIES="${CUDA_COMPUTE_CAPABILITIES}"
    fi
fi

if [[ "${BAZEL_CACHE}" == http://* ]] || \
   [[ "${BAZEL_CACHE}" == grpc://* ]]; then
    BUILD_PARAM="${BUILD_PARAM} --bazel_options=--remote_cache=${BAZEL_CACHE}"
elif [[ ! -z "${BAZEL_CACHE}" ]] ; then
    BUILD_PARAM="${BUILD_PARAM} --bazel_options=--disk_cache=${BAZEL_CACHE}"
fi

if [[ "$DEBUG" == "1" ]]; then
    BUILD_PARAM="${BUILD_PARAM} --bazel_options=-c --bazel_options=dbg --bazel_options=--strip=never --bazel_options=--cxxopt=-g --bazel_options=--cxxopt=-O0"
fi

## Print info

echo "=================================================="
echo "                  Configuration                   "
echo "--------------------------------------------------"

print_var BAZEL_CACHE
print_var BUILD_PATH_JAXLIB
print_var BUILD_PARAM
print_var CLEAN
print_var CLEANONLY
print_var CPU_ARCH
print_var CUDA_COMPUTE_CAPABILITIES
print_var DEBUG
print_var SRC_PATH_JAX
print_var SRC_PATH_XLA

print_var TF_CUDA_VERSION
print_var TF_CUDA_COMPUTE_CAPABILITIES
print_var TF_CUBLAS_VERSION
print_var TF_CUDNN_VERSION
print_var TF_NCCL_VERSION
print_var CC_OPT_FLAGS

print_var XLA_ARM64_PATCH_LIST

echo "=================================================="

if [[ ${DRY} == 1 ]]; then
    echo "Dry run, exiting..."
    exit
fi

if [[ ${CLEANONLY} == 1 ]]; then
    clean
    exit
fi

set -x

# apply patch for XLA
pushd $SRC_PATH_XLA

if [[ "${CPU_ARCH}" == "arm64" ]]; then
    # apply patches if any
    for p in $(echo $XLA_ARM64_PATCH_LIST | tr "," "\n"); do
        echo Apply patch $p
        patch -p1 < $p
    done
fi

popd

## Build jaxlib
mkdir -p "${BUILD_PATH_JAXLIB}"
time python "${SRC_PATH_JAX}/build/build.py" \
    --editable \
    --use_clang \
    --enable_cuda \
    --cuda_path=$TF_CUDA_PATHS \
    --cudnn_path=$TF_CUDNN_PATHS \
    --cuda_version=$TF_CUDA_VERSION \
    --cudnn_version=$TF_CUDNN_VERSION \
    --cuda_compute_capabilities=$TF_CUDA_COMPUTE_CAPABILITIES \
    --enable_nccl=true \
    --bazel_options=--linkopt=-fuse-ld=lld \
    --bazel_options=--override_repository=xla=$SRC_PATH_XLA \
    --output_path=${BUILD_PATH_JAXLIB} \
    $BUILD_PARAM

# Make sure that JAX depends on the local jaxlib installation
# https://jax.readthedocs.io/en/latest/developer.html#specifying-dependencies-on-local-wheels
line="jaxlib @ file://${BUILD_PATH_JAXLIB}"
if ! grep -xF "${line}" "${SRC_PATH_JAX}/build/requirements.in"; then
    pushd "${SRC_PATH_JAX}"
    echo "${line}" >> build/requirements.in
    PYTHON_VERSION=$(python -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))')
    bazel run //build:requirements_dev.update --repo_env=HERMETIC_PYTHON_VERSION="${PYTHON_VERSION}"
    popd
fi

## Install the built packages

# Uninstall jaxlib in case this script was used before.
if [[ "$JAXLIB_ONLY" == "0" ]]; then
    pip uninstall -y jax jaxlib
else
    pip uninstall -y jaxlib
fi

# install jaxlib
pip --disable-pip-version-check install -e ${BUILD_PATH_JAXLIB}

# install jax
if [[ "$JAXLIB_ONLY" == "0" ]]; then
    pip --disable-pip-version-check install -e "${SRC_PATH_JAX}"
fi

## Cleanup

pushd $SRC_PATH_JAX

if [[ "$CLEAN" == "1" ]]; then
    clean
fi

popd
