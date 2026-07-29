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
    echo "                                   Select ccache (default) or sccache with"
    echo "                                   NVTE_CCACHE_BIN. The binary is installed"
    echo "                                   if missing; the caller configures its backend."
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

# Transformer Engine uses separate controls for the number of parallel build
# jobs and the number of threads used by each nvcc invocation. A max-jobs value
# of 0 preserves Transformer Engine's default of using all available jobs.
export NVTE_BUILD_MAX_JOBS="${NVTE_BUILD_MAX_JOBS:-0}"
export NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-8}"
if [[ ! "${NVTE_BUILD_MAX_JOBS}" =~ ^[0-9]+$ ]]; then
    echo "NVTE_BUILD_MAX_JOBS must be a non-negative integer"
    exit 1
fi
if [[ ! "${NVTE_BUILD_THREADS_PER_JOB}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NVTE_BUILD_THREADS_PER_JOB must be a positive integer"
    exit 1
fi

## Print info
echo "=================================================="
echo "                  Configuration                   "
echo "--------------------------------------------------"
print_var CLEAN
print_var INSTALL
print_var NVTE_BUILD_MAX_JOBS
print_var NVTE_BUILD_THREADS_PER_JOB
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
    set +x
    NVTE_CCACHE_BIN="${NVTE_CCACHE_BIN:-ccache}"
    set -x
    CACHE_PROGRAM="$(basename -- "${NVTE_CCACHE_BIN}")"

    if ! command -v "${NVTE_CCACHE_BIN}" &> /dev/null; then
        case "${NVTE_CCACHE_BIN}" in
            sccache)
                SCCACHE_VERSION="${SCCACHE_VERSION:-v0.16.0}"
                if [[ ! "${SCCACHE_VERSION}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                    echo "SCCACHE_VERSION must have the form vX.Y.Z"
                    exit 1
                fi
                case "$(dpkg --print-architecture)" in
                    amd64) SCCACHE_HOST_ARCH=x86_64 ;;
                    arm64) SCCACHE_HOST_ARCH=aarch64 ;;
                    *)
                        echo "No prebuilt sccache binary for $(dpkg --print-architecture)"
                        exit 1
                        ;;
                esac
                SCCACHE_STEM="sccache-${SCCACHE_VERSION}-${SCCACHE_HOST_ARCH}-unknown-linux-musl"
                SCCACHE_URL="https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_STEM}.tar.gz"
                SCCACHE_TMPDIR="$(mktemp -d)"
                SCCACHE_ARCHIVE="${SCCACHE_TMPDIR}/${SCCACHE_STEM}.tar.gz"
                wget -nv --tries=5 --retry-connrefused --waitretry=10 --timeout=60 \
                     --retry-on-http-error=429,500,502,503,504 \
                     -O "${SCCACHE_ARCHIVE}" "${SCCACHE_URL}"
                wget -nv --tries=5 --retry-connrefused -O- "${SCCACHE_URL}.sha256" \
                     | awk -v archive="${SCCACHE_ARCHIVE}" \
                           '{print $1"  "archive}' \
                     | sha256sum -c -
                tar -xzf "${SCCACHE_ARCHIVE}" -C "${SCCACHE_TMPDIR}"
                install -m 755 \
                    "${SCCACHE_TMPDIR}/${SCCACHE_STEM}/sccache" \
                    /usr/local/bin/sccache
                rm -rf "${SCCACHE_TMPDIR}"
                ;;
            ccache)
                apt-get update
                apt-get install -y --no-install-recommends ccache
                ;;
            *)
                echo "${NVTE_CCACHE_BIN} is not installed; automatic installation supports only ccache and sccache"
                exit 1
                ;;
        esac
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
ls dist/
popd

if [[ "${CCACHE}" == "1" ]]; then
    case "${CACHE_PROGRAM}" in
        sccache)
            "${NVTE_CCACHE_BIN}" --show-stats
            if [[ -s "${SCCACHE_ERROR_LOG}" ]]; then
                echo "WARNING: sccache reported backend errors:"
                tail -n 200 "${SCCACHE_ERROR_LOG}"
            fi
            # Stopping the daemon drains any in-flight cache uploads.
            "${NVTE_CCACHE_BIN}" --stop-server || true
            rm -f -- "${SCCACHE_ERROR_LOG}"
            ;;
        ccache)
            "${NVTE_CCACHE_BIN}" --show-stats --verbose
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
