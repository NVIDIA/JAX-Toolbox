#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
set -e

## Utility methods

print_var() {
    echo "$1: ${!1}"
}

# CUDA_ARCH_LIST comes from the dl-cuda-base image starting with CUDA 12.9
# It is a space-separated list of compute capabilities, e.g. "7.5 8.0 12.0"
supported_compute_capabilities() {
    ARCH=$1
    # Infer the compute capabilities from the CUDA_ARCH_LIST variable if it is set
    # Example: "7.5 8.0 12.0" -> "sm_75,sm_80,compute_120"
    if [[ -n "${CUDA_ARCH_LIST}" ]]; then
        read -r -a _CUDA_ARCH_LIST <<< "${CUDA_ARCH_LIST}"
        SM_LIST=""
        for _ARCH in "${_CUDA_ARCH_LIST[@]}"; do
            if [[ "${_ARCH}" == "${_CUDA_ARCH_LIST[-1]}" ]]; then
                SM_LIST="${SM_LIST}compute_${_ARCH//./}"
            else
                SM_LIST="${SM_LIST}sm_${_ARCH//./},"
            fi
        done
        echo "${SM_LIST}"
    elif [[ "${ARCH}" == "amd64" ]]; then
        echo "sm_75,sm_80,sm_86,sm_90,sm_100,compute_120"
    elif [[ "${ARCH}" == "arm64" ]]; then
        echo "sm_80,sm_86,sm_90,sm_100,compute_120"
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
    echo "    --bazel-cache-namespace NAME   Namespace for bazel cache content"
    echo "    --build-param PARAM            Param passed to the jaxlib build command. Can be passed many times."
    echo "    --build-path-jaxlib PATH       Editable install prefix for jaxlib and plugins"
    echo "    --clean                        Delete local configuration and bazel cache"
    echo "    --clean-only                   Do not build, just cleanup"
    echo "    --cpu-arch                     Target CPU architecture, e.g. amd64, arm64, etc."
    echo "    --debug                        Build in debug mode"
    echo "    --dry                          Dry run, parse arguments only"
    echo "    -h, --help                     Print usage."
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
BAZEL_CACHE_NAMESPACE="jax${CUDA_BASE_IMAGE:+:}${CUDA_BASE_IMAGE}"
BUILD_PATH_JAXLIB="/opt/jaxlibs"
BUILD_PARAM=""
CLEAN=0
CLEANONLY=0
CPU_ARCH="$(dpkg --print-architecture)"
CUDA_COMPUTE_CAPABILITIES="local"
DEBUG=0
DRY=0
SRC_PATH_JAX="/opt/jax"
SRC_PATH_XLA="/opt/xla"
XLA_ARM64_PATCH_LIST=""

args=$(getopt -o h --long bazel-cache:,bazel-cache-namespace:,build-param:,build-path-jaxlib:,clean,cpu-arch:,debug,no-clean,clean-only,dry,help,src-path-jax:,src-path-xla:,sm:,xla-arm64-patch: -- "$@")
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
        --bazel-cache-namespace)
            BAZEL_CACHE_NAMESPACE=$2
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
export TF_CUDA_MAJOR_VERSION=$(ls /usr/local/cuda/lib64/libcudart.so.*.*.* | cut -d . -f 3)
export TF_CUBLAS_VERSION=$(ls /usr/local/cuda/lib64/libcublas.so.*.*.* | cut -d . -f 3)
export TF_NCCL_VERSION=$(echo "${NCCL_VERSION}" | cut -d . -f 1)

TF_CUDNN_MAJOR_VERSION=$(grep "#define CUDNN_MAJOR" /usr/include/cudnn_version.h | awk '{print $3}')
TF_CUDNN_MINOR_VERSION=$(grep "#define CUDNN_MINOR" /usr/include/cudnn_version.h | awk '{print $3}')
TF_CUDNN_PATCHLEVEL_VERSION=$(grep "#define CUDNN_PATCHLEVEL" /usr/include/cudnn_version.h | awk '{print $3}')
export TF_CUDNN_VERSION="${TF_CUDNN_MAJOR_VERSION}.${TF_CUDNN_MINOR_VERSION}.${TF_CUDNN_PATCHLEVEL_VERSION}"

case "${CPU_ARCH}" in
    "amd64")
        export CC_OPT_FLAGS="-march=sandybridge -mtune=broadwell"
        ;;
    "arm64")
        export CC_OPT_FLAGS="-march=armv8-a"
        # ARM ACL build issue introduced in PR#23225
        BUILD_PARAM="${BUILD_PARAM} --disable_mkl_dnn"
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
    if [[ -n "${BAZEL_CACHE_NAMESPACE}" ]]; then
        BUILD_PARAM="${BUILD_PARAM} --bazel_options=--remote_instance_name=${BAZEL_CACHE_NAMESPACE}"
    fi
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
print_var TF_CUDA_MAJOR_VERSION
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
if [[ ! -e "/usr/local/cuda/lib" ]]; then
    ln -s /usr/local/cuda/lib64 /usr/local/cuda/lib
fi

if ! grep 'try-import %workspace%/.local_cuda.bazelrc' "${SRC_PATH_JAX}/.bazelrc"; then
    echo -e '\ntry-import %workspace%/.local_cuda.bazelrc' >> "${SRC_PATH_JAX}/.bazelrc"
fi
cat > "${SRC_PATH_JAX}/.local_cuda.bazelrc" << EOF
build --repo_env=LOCAL_CUDA_PATH="/usr/local/cuda"
build --repo_env=LOCAL_CUDNN_PATH="/opt/nvidia/cudnn"
build --repo_env=LOCAL_NCCL_PATH="/opt/nvidia/nccl"
EOF

pushd ${SRC_PATH_JAX}
time python "${SRC_PATH_JAX}/build/build.py" build \
    --editable \
    --use_clang \
    --use_new_wheel_build_rule \
    --wheels=jax,jaxlib,jax-cuda-plugin,jax-cuda-pjrt \
    --cuda_compute_capabilities=$TF_CUDA_COMPUTE_CAPABILITIES \
    --bazel_options=--linkopt=-fuse-ld=lld \
    --local_xla_path=$SRC_PATH_XLA \
    --output_path=${BUILD_PATH_JAXLIB} \
    --bazel_options=--repo_env=ML_WHEEL_TYPE=release \
    $BUILD_PARAM
popd

sed -i "s|      f'jaxlib >={_minimum_jaxlib_version}, <={_jax_version}',|      f'jaxlib>=0.5.0',|" /opt/jax/setup.py
# Make sure that JAX depends on the local jaxlib installation
# https://jax.readthedocs.io/en/latest/developer.html#specifying-dependencies-on-local-wheels
line="jax @ file://${BUILD_PATH_JAXLIB}/jax"
if ! grep -xF "${line}" "${SRC_PATH_JAX}/build/requirements.in"; then
    pushd "${SRC_PATH_JAX}"
    echo "${line}" >> build/requirements.in
    echo "jaxlib @ file://${BUILD_PATH_JAXLIB}/jaxlib" >> build/requirements.in
    echo "jax-cuda${TF_CUDA_MAJOR_VERSION}-pjrt @ file://${BUILD_PATH_JAXLIB}/jax_cuda${TF_CUDA_MAJOR_VERSION}_pjrt" >> build/requirements.in
    echo "jax-cuda${TF_CUDA_MAJOR_VERSION}-plugin @ file://${BUILD_PATH_JAXLIB}/jax_cuda${TF_CUDA_MAJOR_VERSION}_plugin" >> build/requirements.in
    PYTHON_VERSION=$(python -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))')
    bazel run --verbose_failures=true //build:requirements.update --repo_env=HERMETIC_PYTHON_VERSION="${PYTHON_VERSION}"
    popd
fi
## Install the built packages

# Uninstall jaxlib in case this script was used before.
pip uninstall -y jax jaxlib jax-cuda${TF_CUDA_MAJOR_VERSION}-pjrt jax-cuda${TF_CUDA_MAJOR_VERSION}-plugin

# install jax and jaxlib
# pip --disable-pip-version-check install -e ${BUILD_PATH_JAXLIB}/jaxlib -e ${BUILD_PATH_JAXLIB}/jax_cuda${TF_CUDA_MAJOR_VERSION}_pjrt -e ${BUILD_PATH_JAXLIB}/jax_cuda${TF_CUDA_MAJOR_VERSION}_plugin 
# jaxlib_version=$(pip show jaxlib | grep Version | tr ':' '\n' | tail -1)
# sed -i "s|^_current_jaxlib_version.*|_current_jaxlib_version = '${jaxlib_version}'|" /opt/jax/setup.py

# pip --disable-pip-version-check install -e ${BUILD_PATH_JAXLIB}/jax

## after installation (example)
# jax                     0.5.4.dev20250325    /opt/jaxlibs/jax
# jax-cuda12-pjrt         0.5.4.dev20250325    /opt/jaxlibs/jax_cuda12_pjrt
# jax-cuda12-plugin       0.5.4.dev20250325    /opt/jaxlibs/jax_cuda12_plugin
# jaxlib                  0.5.4.dev20250325    /opt/jaxlibs/jaxlib
pip list | grep jax

# Ensure directories are readable by all for non-root users
chmod 755 $BUILD_PATH_JAXLIB/*

## Cleanup

pushd $SRC_PATH_JAX

if [[ "$CLEAN" == "1" ]]; then
    clean
fi

popd
