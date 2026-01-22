#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
set -euo pipefail

## Utility methods

print_var() {
    echo "$1: ${!1}"
}

# CUDA_ARCH_LIST comes from the dl-cuda-base image starting with CUDA 12.9
# It is a space-separated list of compute capabilities, e.g. "7.5 8.0 12.0"
supported_compute_capabilities() {
    # Infer the compute capabilities from the CUDA_ARCH_LIST variable if it is set
    # Example: "7.5 8.0 12.0" -> "sm_75,sm_80,compute_120"
    if [[ -z "${CUDA_ARCH_LIST}" ]]; then
        echo "CUDA_ARCH_LIST was not set; this is defined by the dl-cuda-base image"
        return 1
    fi
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
    echo "    --debug                        Build in debug mode"
    echo "    --extra-targets T1[,T2[...]    Extra bazel targets that will be built and copied to --extra-target-dest."
    echo "    --extra-target-dest PATH       Where extra target output files will be copied to."
    echo "    -h, --help                     Print usage."
    echo "    --install                      Install the JAX wheels when build succeeds"
    echo "    --no-install                   Do not install the JAX wheels when build succeeds"
    echo "    --no-clean                     Do not delete local configuration and bazel cache (default)"
    echo "    --src-path-jax                 Path to JAX source"
    echo "    --src-path-xla                 Path to XLA source"
    echo "    --sm SM1,SM2,...               Comma-separated list of CUDA SM versions to compile for, e.g. '7.5,8.0'"
    echo "    --sm local                     Query the SM of available GPUs"
    echo "    --sm all                       Default list of SMs taken from the base container (default)"
    echo "                                   If you want to pass a bazel parameter, you must do it like this:"
    echo "                                   --build-param=--bazel_options=..."
    exit $1
}

# Set defaults
BAZEL_CACHE=""
BAZEL_CACHE_NAMESPACE="jax${CUDA_BASE_IMAGE:+:}${CUDA_BASE_IMAGE}"
BUILD_PATH_JAXLIB="/opt/jaxlibs"
BUILD_PARAM=""
CLEAN=0
CLEANONLY=0
CUDA_COMPUTE_CAPABILITIES="all"
DEBUG=0
EXTRA_TARGETS=()
EXTRA_TARGET_DEST=""
INSTALL=1
IS_RELEASE=0
SRC_PATH_JAX="/opt/jax"
SRC_PATH_XLA="/opt/xla"

args=$(getopt -o h,r --long bazel-cache:,bazel-cache-namespace:,build-param:,build-path-jaxlib:,clean,release,cpu-arch:,debug,extra-targets:,extra-target-dest:,no-clean,clean-only,help,install,no-install,src-path-jax:,src-path-xla:,sm: -- "$@")
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
        --clean)
            CLEAN=1
            shift 1
            ;;
        --clean-only)
            CLEANONLY=1
            shift 1
            ;;
        --no-clean)
            CLEAN=0
            shift 1
            ;;
        --debug)
            DEBUG=1
            shift 1
            ;;
        --extra-targets)
            IFS=',' read -r -a EXTRA_TARGETS <<< "$2"
            shift 2
            ;;
        --extra-target-dest)
            EXTRA_TARGET_DEST="$2"
            shift 2
            ;;
        -r | --release)
            IS_RELEASE=1
            shift 1
            ;;
        -h | --help)
            usage 1
            ;;
        --install)
            INSTALL=1
            shift 1
            ;;
        --no-install)
            INSTALL=0
            shift 1
            ;;
        --src-path-jax)
            SRC_PATH_JAX=$(realpath $2)
            shift 2
            ;;
        --src-path-xla)
            SRC_PATH_XLA=$(realpath $2)
            shift 2
            ;;
        --sm)
            CUDA_COMPUTE_CAPABILITIES=$2
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

clean() {
    pushd "${SRC_PATH_JAX}"
    # Remove --remote_cache and --remote_instance_name from .jax_configure.bazelrc
    sed -i '/^build --\(remote_cache\|remote_instance_name\)=/d' .jax_configure.bazelrc
    bazel clean --expunge || true
    rm -rf ${HOME}/.cache/bazel
    popd
}

# Derive CUDA_MAJOR_VERSION from CUDA_VERSION if it isn't set
if [ -z ${CUDA_MAJOR_VERSION+x} ]; then
    CUDA_MAJOR_VERSION="${CUDA_VERSION:0:2}"
fi
PYTHON_MAJOR_VERSION=$(python -c 'import sys; print(sys.version_info[0])')
PYTHON_MINOR_VERSION=$(python -c 'import sys; print(sys.version_info[1])')
PYTHON_VERSION="${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}"
if [[ "$CUDA_COMPUTE_CAPABILITIES" == "all" ]]; then
    CUDA_COMPUTE_CAPABILITIES=$(supported_compute_capabilities)
elif [[ "$CUDA_COMPUTE_CAPABILITIES" == "local" ]]; then
    CUDA_COMPUTE_CAPABILITIES=$("${SCRIPT_DIR}/local_cuda_arch")
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
print_var CUDA_COMPUTE_CAPABILITIES
print_var CUDA_MAJOR_VERSION
print_var DEBUG
# bash -u does not accept set-but-empty arrays in all versions
echo EXTRA_TARGETS: ${EXTRA_TARGETS[@]+"${EXTRA_TARGETS[@]}"}
print_var EXTRA_TARGET_DEST
print_var INSTALL
print_var PYTHON_VERSION
print_var SRC_PATH_JAX
print_var SRC_PATH_XLA
print_var IS_RELEASE

echo "=================================================="

if [[ -n "${EXTRA_TARGET_DEST}" && ! -d "${EXTRA_TARGET_DEST}" ]]; then
    echo "You must pass a directory to --extra-target-dest"
    exit 1
fi

set -x
if [[ ${CLEANONLY} == 1 ]]; then
    clean
    exit
fi

# MOSAIC_GPU_NVSHMEM_{BC,SO}_PATH settings are a workaround for NVSHMEM
# installations at the system level, not via PyPI wheels

## Build the compiled parts of JAX
pushd ${SRC_PATH_JAX}
time python "${SRC_PATH_JAX}/build/build.py" build \
    --cuda_major_version=${CUDA_MAJOR_VERSION} \
    --editable \
    --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt \
    "--cuda_compute_capabilities=${CUDA_COMPUTE_CAPABILITIES}" \
    --bazel_options=--repo_env=LOCAL_CUDA_PATH="/usr/local/cuda" \
    --bazel_options=--repo_env=LOCAL_CUDNN_PATH="/opt/nvidia/cudnn" \
    --bazel_options=--repo_env=LOCAL_NCCL_PATH="/opt/nvidia/nccl" \
    --bazel_options=--repo_env=LOCAL_NVSHMEM_PATH="/opt/nvidia/nvshmem" \
    --bazel_options=--test_env=MOSAIC_GPU_NVSHMEM_BC_PATH="/opt/nvidia/nvshmem/lib/libnvshmem_device.bc" \
    --bazel_options=--test_env=MOSAIC_GPU_NVSHMEM_SO_PATH="/opt/nvidia/nvshmem/lib/libnvshmem_host.so" \
    --bazel_options=--linkopt=-fuse-ld=lld \
    "--local_xla_path=${SRC_PATH_XLA}" \
    "--output_path=${BUILD_PATH_JAXLIB}" \
    $BUILD_PARAM

# Make sure that JAX depends on the local jaxlib installation
# https://jax.readthedocs.io/en/latest/developer.html#specifying-dependencies-on-local-wheels
# We do not run the requirements.update target because it is very slow and
# downloads the pip world
#
# Point the jaxlib/jax-cuda*-plugin|pjrt lines in requirements files to the
# local builds, removing --hash on continued lines.
for component in jaxlib "jax-cuda${CUDA_MAJOR_VERSION}-pjrt" "jax-cuda${CUDA_MAJOR_VERSION}-plugin"; do
    sed \
        -i \
        "/^${component}/{:a;/\\\\$/N;s/\\\\\n\s*/ /;ta;s#.*#${component} @ file://${BUILD_PATH_JAXLIB}/${component//-/_}#;}" \
        build/requirements.in \
        build/requirements_lock_${PYTHON_MAJOR_VERSION}_${PYTHON_MINOR_VERSION}.txt
done
if [[ "${IS_RELEASE}" == "1" ]]; then
    sed -i "s|f'jaxlib >={_minimum_jaxlib_version}, <={_jax_version}',|f'jaxlib>=0.5.0',|" /opt/jax/setup.py
fi
if (( "${#EXTRA_TARGETS[@]}" > 0 )); then
    bazel build "${BAZEL_ARGS[@]}" --verbose_failures=true "${EXTRA_TARGETS[@]}"
    if [[ -n "${EXTRA_TARGET_DEST}" ]]; then
        mkdir -p ${EXTRA_TARGET_DEST}/bin ${EXTRA_TARGET_DEST}/lib ${EXTRA_TARGET_DEST}/python
        for target in "${EXTRA_TARGETS[@]}"; do
            target_type=$(bazel cquery "${BAZEL_ARGS[@]}" "${target}" --output label_kind 2>/dev/null | cut -f1 -d' ')
            output_files=$(bazel cquery "${BAZEL_ARGS[@]}" "${target}" --output files 2>/dev/null)
            subdir_dest=""
            case ${target_type} in
                "cc_binary")
                    if file ${output_files} | grep -q "shared object"; then
                        # Check for PyInit to see if this is a python module
                        if nm -D ${output_files} | grep PyInit; then
                            subdir_dest="python"
                        else
                            subdir_dest="lib"
                        fi
                    else
                        subdir_dest="bin"
                    fi
                ;;
                "python_library") subdir_dest="python" ;;
                *) echo "warning: unexpected type of extra target: ${target_type} ${target}" ;;
            esac
            cp -v ${output_files} "${EXTRA_TARGET_DEST}/${subdir_dest}"
        done
    fi
fi
popd

## Install the built packages

if [[ "${INSTALL}" == "1" ]]; then
    # Uninstall jaxlib in case this script was used before.
    time pip uninstall -y jax jaxlib jax-cuda${CUDA_MAJOR_VERSION}-pjrt jax-cuda${CUDA_MAJOR_VERSION}-plugin

    # install jax and jaxlib
    time pip --disable-pip-version-check install -e ${BUILD_PATH_JAXLIB}/jaxlib -e ${BUILD_PATH_JAXLIB}/jax_cuda${CUDA_MAJOR_VERSION}_pjrt -e ${BUILD_PATH_JAXLIB}/jax_cuda${CUDA_MAJOR_VERSION}_plugin -e ${SRC_PATH_JAX}

    ## after installation (example)
    # jax                     0.6.1.dev20250425+966578b61 /opt/jax
    # jax-cuda12-pjrt         0.6.1.dev20250425           /opt/jaxlibs/jax_cuda12_pjrt
    # jax-cuda12-plugin       0.6.1.dev20250425           /opt/jaxlibs/jax_cuda12_plugin
    # jaxlib                  0.6.1.dev20250425           /opt/jaxlibs/jaxlib
    pip list | grep ^jax
fi

# Ensure directories are readable by all for non-root users
chmod 755 $BUILD_PATH_JAXLIB/*

## Cleanup
if [[ "$CLEAN" == "1" ]]; then
    clean
fi
