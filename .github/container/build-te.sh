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
    echo "    --ccache                       Use a compiler cache to build TransformerEngine."
    echo "    --no-install                   Only build a wheel; do not install."
    echo "    --src-path-te                  Path to TransformerEngine source code."
    echo "    --src-path-xla                 Path to XLA source code."
    echo "    --sm SM1,SM2,...               Comma-separated list of CUDA SM versions"
    echo "                                   to compile for, e.g. 7.5,8.0 -- PTX will"
    echo "                                   only be emitted for the last one."
    echo "    --sm local                     Compile for the local GPUs."
    echo "    --sm all                       Compile for a default set of SM versions (default)."
    exit $1
}

# Set defaults
CCACHE=0
CLEAN=0
INSTALL=1
SM="all"
SRC_PATH_TE="/opt/transformer-engine"
SRC_PATH_XLA="/opt/xla"

args=$(getopt -o h --long ccache,clean,help,no-install,src-path-te:,src-path-xla:,sm: -- "$@")
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
        --ccache)
            CCACHE=1
            shift 1
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

print_var() {
    echo "$1: ${!1}"
}

clean() {
    pushd "${SRC_PATH_TE}"
    rm -rf build/ .eggs/
    popd
}

# This should standardise on 1.2,3.4,5.6 format
if [[ "$SM" == "all" ]]; then
    if [[ -z "${CUDA_ARCH_LIST}" ]]; then
        echo "CUDA_ARCH_LIST was not set; this is defined by the dl-cuda-base image"
        return 1
    fi
    # Infer the compute capabilities from the CUDA_ARCH_LIST variable if it is set;
    # this is in 1.2 3.4 5.6 format
    SM_LIST=${CUDA_ARCH_LIST// /,}
elif [[ "$SM" == "local" ]]; then
    SM_LIST=$("${SCRIPT_DIR}/local_cuda_arch")
    if [[ -z "${SM_LIST}" ]]; then
        echo "Could not determine the local GPU architecture."
        echo "You should pass --sm when compiling on a machine without GPUs."
        nvidia-smi || true
        exit 1
    fi
else
    SM_LIST=${SM}
fi


# Query nvcc so version-specific
# so compiler-cache workarounds are selected reliably.
CUDA_TOOLKIT_VERSION="${CUDA_VERSION:-}"
if command -v nvcc &> /dev/null; then
    DETECTED_CUDA_TOOLKIT_VERSION="$(
        nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\),.*/\1/p'
    )"
    if [[ -n "${DETECTED_CUDA_TOOLKIT_VERSION}" ]]; then
        CUDA_TOOLKIT_VERSION="${DETECTED_CUDA_TOOLKIT_VERSION}"
    fi
fi

## Print info
echo "=================================================="
echo "                  Configuration                   "
echo "--------------------------------------------------"
print_var CUDA_TOOLKIT_VERSION
print_var CLEAN
print_var INSTALL
print_var SM
print_var SM_LIST
print_var SRC_PATH_TE
print_var SRC_PATH_XLA
echo "=================================================="

# Parse SM_LIST into the format accepted by TransformerEngine's build system
# "1.2,3.4,5.6" -> "12;34;56". In principle we would like to compile SASS-only
# plus PTX for the highest known architecture, but TransformerEngine's build
# system does not currently handle that.
NVTE_CUDA_ARCHS="${SM_LIST//,/;}"
set -x
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS//./}"
# Parallelism within nvcc invocations.
export NVTE_BUILD_THREADS_PER_JOB=8
export NVTE_FRAMEWORK=jax
# TransformerEngine needs FFI headers from XLA
export XLA_HOME=${SRC_PATH_XLA}

pushd ${SRC_PATH_TE}
# Install some build dependencies, but avoid installing everything
# (jax, torch, ...) because we do not want to pull in a released version of
# JAX, or the wheel-based installation of CUDA. Note that when we build TE as
# part of building the JAX containers, JAX and XLA are not yet installed.
python - << EOF
import os, subprocess, sys, tomllib
with open("pyproject.toml", "rb") as ifile:
    data = tomllib.load(ifile)
subprocess.run(
    [sys.executable, "-m", "pip", "install"]
    + [r for r in data["build-system"]["requires"]
       if r.startswith("pybind11") or r.startswith("cmake") or r.startswith("ninja")]
    + [f"nvidia-cudnn-frontend=={os.environ['CUDNN_FRONTEND_VERSION']}"]
)
EOF
# Transformer Engine adds NVTE_CCACHE_BIN as both its C++ and CUDA CMake
# compiler launcher when NVTE_USE_CCACHE is set. Do not also wrap CXX: that
# would produce a recursive "ccache ccache g++" invocation.
if [[ "${CCACHE}" == "1" ]]; then
    NVTE_CCACHE_BIN="${NVTE_CCACHE_BIN:-ccache}"
    CACHE_PROGRAM="$(basename -- "${NVTE_CCACHE_BIN}")"

    if ! command -v "${NVTE_CCACHE_BIN}" &> /dev/null; then
        # Keep direct callers backwards compatible while the Docker build
        # installs its selected cache explicitly.
        CACHE_INSTALLER="${SCRIPT_DIR}/install-compiler-cache.sh"
        if [[ ! -x "${CACHE_INSTALLER}" ]]; then
            CACHE_INSTALLER="$(command -v install-compiler-cache.sh || true)"
        fi
        if [[ -z "${CACHE_INSTALLER}" || ! -x "${CACHE_INSTALLER}" ]]; then
            echo "${NVTE_CCACHE_BIN} is not installed and install-compiler-cache.sh is unavailable"
            exit 1
        fi
        "${CACHE_INSTALLER}" "${NVTE_CCACHE_BIN}"
    fi
    if ! command -v "${NVTE_CCACHE_BIN}" &> /dev/null; then
        echo "Compiler cache installation did not provide ${NVTE_CCACHE_BIN}"
        exit 1
    fi

    export NVTE_USE_CCACHE=1
    export NVTE_CCACHE_BIN
    case "${CACHE_PROGRAM}" in
        sccache)
            # Keep the daemon alive for long CUDA builds and retain an error log
            # so backend failures are visible alongside the cache statistics.
            export SCCACHE_IDLE_TIMEOUT="${SCCACHE_IDLE_TIMEOUT:-0}"
            export SCCACHE_ERROR_LOG="${SCCACHE_ERROR_LOG:-/tmp/sccache-server.log}"
            rm -f -- "${SCCACHE_ERROR_LOG}"
            "${NVTE_CCACHE_BIN}" --start-server
            # WAR needed for CUDA 13.3 because the --simt-only path is not exercised by the TE build system.
            if [[ -n "${CUDA_TOOLKIT_VERSION}" ]] && \
               dpkg --compare-versions "${CUDA_TOOLKIT_VERSION}" ge "13.3"; then
                # Exercise the CUDA 13.3 --simt-only path before starting the
                # expensive TE build. This also catches S3/auth failures early.
                SCCACHE_SMOKE_DIR="$(mktemp -d)"
                SCCACHE_SMOKE_ARCH="${NVTE_CUDA_ARCHS%%;*}"
                # Make the preprocessed source unique so a remote cache hit
                # cannot bypass the compiler path this check is exercising.
                SCCACHE_SMOKE_TOKEN="${RANDOM}${RANDOM}"
                printf '__global__ void k() { unsigned long long token = %sULL; }\n' \
                    "${SCCACHE_SMOKE_TOKEN}" \
                    > "${SCCACHE_SMOKE_DIR}/sccache-smoke.cu"
                if ! "${NVTE_CCACHE_BIN}" "$(command -v nvcc)" \
                    -rdc=true \
                    -gencode \
                    "arch=compute_${SCCACHE_SMOKE_ARCH},code=sm_${SCCACHE_SMOKE_ARCH}" \
                    -c "${SCCACHE_SMOKE_DIR}/sccache-smoke.cu" \
                    -o "${SCCACHE_SMOKE_DIR}/sccache-smoke.o"; then
                    echo "CUDA+sccache smoke compilation failed"
                    "${NVTE_CCACHE_BIN}" --show-stats || true
                    if [[ -s "${SCCACHE_ERROR_LOG}" ]]; then
                        tail -n 200 "${SCCACHE_ERROR_LOG}"
                    fi
                    rm -rf "${SCCACHE_SMOKE_DIR}"
                    exit 1
                fi
                rm -rf "${SCCACHE_SMOKE_DIR}"
            fi
            "${NVTE_CCACHE_BIN}" --zero-stats
            ;;
        ccache)
            export CCACHE_DIR="${CCACHE_DIR:-/root/.cache/ccache}"
            "${NVTE_CCACHE_BIN}" --zero-stats
            ;;
        *)
            echo "Unsupported compiler cache: ${NVTE_CCACHE_BIN}"
            exit 1
            ;;
    esac
    echo "Transformer Engine compiler cache: ${CACHE_PROGRAM}"
fi

# The wheel filename includes the TE commit; if this has changed since the last
# incremental build then we would end up with multiple wheels.
rm -fv dist/*.whl
python setup.py bdist_wheel
popd

CACHE_STATUS=0
if [[ "${CCACHE}" == "1" ]]; then
    case "${CACHE_PROGRAM}" in
        sccache)
            "${NVTE_CCACHE_BIN}" --show-stats || CACHE_STATUS=$?
            if [[ -s "${SCCACHE_ERROR_LOG}" ]]; then
                echo "WARNING: sccache reported backend errors:"
                tail -n 200 "${SCCACHE_ERROR_LOG}"
            fi
            # Stopping the daemon drains any in-flight cache uploads.
            "${NVTE_CCACHE_BIN}" --stop-server || true
            rm -f -- "${SCCACHE_ERROR_LOG}"
            ;;
        ccache)
            "${NVTE_CCACHE_BIN}" --show-stats --verbose || CACHE_STATUS=$?
            ;;
    esac
fi

## Install the built packages
if [[ "${INSTALL}" == "1" ]]; then
    pip uninstall -y transformer_engine
    pip install ${SRC_PATH_TE}/dist/*.whl
    pip list | grep ^transformer_engine
fi

## Cleanup
if [[ "$CLEAN" == "1" ]]; then
    clean
fi
