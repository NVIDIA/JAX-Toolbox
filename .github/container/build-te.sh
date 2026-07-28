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
    echo "                                   sccache is used when a remote backend is configured"
    echo "                                   (SCCACHE_BUCKET/SCCACHE_REDIS/SCCACHE_WEBDAV_ENDPOINT),"
    echo "                                   otherwise ccache. Override with NVTE_CCACHE_BIN. The"
    echo "                                   binary is installed if missing; the caller is"
    echo "                                   responsible for configuring the backend."
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
# Compiler cache. TransformerEngine's build system honours NVTE_USE_CCACHE by
# adding -DCMAKE_CXX_COMPILER_LAUNCHER and -DCMAKE_CUDA_COMPILER_LAUNCHER
# (build_tools/build_ext.py), so the nvcc device compilations -- the overwhelming
# majority of this build -- go through the cache too. NVTE_CCACHE_BIN selects
# which launcher binary is used.
#
# Scope: those are CMake variables, so this covers the CMake sub-build only (the
# ~98 ninja edges that dominate the build). TE's 3rdparty/nccl-extensions Makefile
# and the 18 distutils pybind11 .cpp compiles sit outside CMake and stay uncached.
# Do not reach for CC/CXX to cover them: those are process-global, CMake would
# consume them alongside the launcher, and that reintroduces the recursion below.
#
# Do NOT also set CXX="ccache g++" here. Combined with the launcher above that
# yields "ccache ccache g++", and ccache aborts with "Recursive invocation of
# ccache" at the first C++ build edge -- after cmake's configure step has already
# succeeded, so it looks like a TE incompatibility rather than a config error.
if [[ "${CCACHE}" == "1" ]]; then
    # sccache can share its cache over S3, which is what makes caching useful on
    # ephemeral CI runners. Plain ccache is local-only unless the caller has
    # configured remote storage, but it is a sensible default for local rebuilds.
    # xtrace is off for the test itself: it would expand SCCACHE_* into the log.
    # GitHub masks registered secrets in Actions logs, but local and triage runs
    # have no masking, so avoid echoing credential-adjacent values anywhere.
    set +x
    if [[ -z "${NVTE_CCACHE_BIN:-}" ]]; then
        if [[ -n "${SCCACHE_BUCKET:-}${SCCACHE_REDIS:-}${SCCACHE_WEBDAV_ENDPOINT:-}" ]]; then
            NVTE_CCACHE_BIN=sccache
        else
            NVTE_CCACHE_BIN=ccache
        fi
    fi
    set -x

    if ! command -v "${NVTE_CCACHE_BIN}" &> /dev/null; then
        case "${NVTE_CCACHE_BIN}" in
            sccache)
                SCCACHE_VERSION="${SCCACHE_VERSION:-v0.16.0}"
                case "$(dpkg --print-architecture)" in
                    amd64) SCCACHE_HOST_ARCH=x86_64 ;;
                    arm64) SCCACHE_HOST_ARCH=aarch64 ;;
                    *)
                        echo "No sccache build for $(dpkg --print-architecture)"
                        exit 1
                        ;;
                esac
                SCCACHE_STEM="sccache-${SCCACHE_VERSION}-${SCCACHE_HOST_ARCH}-unknown-linux-musl"
                SCCACHE_URL="https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_STEM}.tar.gz"
                # Deliberately not -q: with pipefail a 404 would otherwise surface
                # as "tar: This does not look like a tar archive". This download
                # sits at the head of a ~48 minute stage, so it retries.
                wget -nv --tries=5 --retry-connrefused --waitretry=10 --timeout=60 \
                     --retry-on-http-error=429,500,502,503,504 \
                     -O /tmp/sccache.tgz "${SCCACHE_URL}"
                wget -nv --tries=5 --retry-connrefused -O- "${SCCACHE_URL}.sha256" \
                     | awk '{print $1"  /tmp/sccache.tgz"}' | sha256sum -c -
                tar -xzf /tmp/sccache.tgz -C /tmp
                install -m 755 "/tmp/${SCCACHE_STEM}/sccache" /usr/local/bin/sccache
                rm -rf "/tmp/${SCCACHE_STEM}" /tmp/sccache.tgz
                ;;
            ccache)
                # needs >= 4.1 for Redis remote storage support
                apt-get update && apt-get install -y --no-install-recommends ccache
                ;;
            *)
                echo "${NVTE_CCACHE_BIN} is not installed and cannot be installed automatically"
                exit 1
                ;;
        esac
    fi

    export NVTE_USE_CCACHE=1
    export NVTE_CCACHE_BIN
    # basename, not a bare string compare: NVTE_CCACHE_BIN may be an absolute
    # path (the triage tool forwards arbitrary VAR=val into this script), and
    # taking the ccache branch for a path-valued sccache would run a binary that
    # is not installed.
    if [[ "$(basename "${NVTE_CCACHE_BIN}")" == "sccache" ]]; then
        # Without this the server exits after 10 minutes idle and takes the
        # end-of-build statistics with it.
        export SCCACHE_IDLE_TIMEOUT=0
        # The server's stderr is discarded unless this is set, and read errors
        # are reported as plain cache misses -- without it a broken cache is
        # indistinguishable from a cold one.
        export SCCACHE_ERROR_LOG=/tmp/sccache-server.log
        # Fail soft: an unreachable or misconfigured cache should make this build
        # slow, not red. Without this a bad bucket/credential turns every compiler
        # invocation into an error.
        if ! { sccache --start-server && sccache --zero-stats > /dev/null; }; then
            echo "WARNING: could not start sccache; building without a compiler cache"
            unset NVTE_USE_CCACHE NVTE_CCACHE_BIN
            CCACHE=0
        else
            # The daemon captured the credentials at exec; there is no reason to
            # keep them in the environment of cmake, ninja and ~500 nvcc processes.
            unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
        fi
    else
        # Give ccache a stable directory so a BuildKit cache mount (or a local
        # developer's repeated builds) can actually retain anything.
        export CCACHE_DIR="${CCACHE_DIR:-/root/.cache/ccache}"
        ccache --zero-stats
    fi
fi

# The wheel filename includes the TE commit; if this has changed since the last
# incremental build then we would end up with multiple wheels.
rm -fv dist/*.whl
python setup.py bdist_wheel
ls dist/
popd

if [[ "${CCACHE}" == "1" ]]; then
    if [[ "$(basename "${NVTE_CCACHE_BIN}")" == "sccache" ]]; then
        sccache --show-stats
        # A one-line, greppable summary. This warns rather than asserts, to match
        # the fail-soft behaviour above; put any hard gate in a separate CI step
        # that greps for SCCACHE_SUMMARY.
        sccache --show-stats --stats-format=json > /tmp/sccache-stats.json || true
        python - <<'PY' || true
import json
s = json.load(open("/tmp/sccache-stats.json"))["stats"]
hits = sum(s["cache_hits"]["counts"].values())
misses = sum(s["cache_misses"]["counts"].values())
writes = s["cache_write_errors"]
print(f"SCCACHE_SUMMARY hits={hits} misses={misses} "
      f"write_errors={writes} errors={s['cache_errors']['counts']}")
if hits + misses == 0:
    print("SCCACHE_SUMMARY WARNING: sccache saw no cacheable compilations")
if writes:
    print("SCCACHE_SUMMARY WARNING: cache not fully populated; "
          "check whether the AWS session expired mid-build")
PY
        if [[ -s "${SCCACHE_ERROR_LOG}" ]]; then
            echo "--- sccache server log ---"
            cat "${SCCACHE_ERROR_LOG}"
        fi
        # Also drains any in-flight cache uploads.
        sccache --stop-server || true
    else
        ccache --show-stats --verbose
    fi
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
