#!/bin/bash

## Parse command-line arguments

usage() {
    echo "Run JAX tests in a container"
    echo ""
    echo "Usage: $0 [OPTION]... TESTS"
    echo "  -b, --battery          Specify predefined test batteries to run."
    echo "  --src-path-jax         Path to JAX source."
    echo "  --build-jaxlib         Runs the JAX tests using jaxlib built from source."
    echo "  --cache-test-results   yes|no|auto, passes through to bazel --cache_test_results"
    echo "  --reuse-jaxlib         Runs the JAX tests using preinstalled jaxlib. (DEFAULT)"
    echo "  --disable-x64          Disable 64-bit floating point support in JAX (some tests may fail)"
    echo "  --enable-x64           Enable 64-bit floating point support in JAX (DEFAULT, required for some tests)"
    echo "  --tests-per-gpu        How many test shards should be launched on each GPU in parallel."
    echo "  --visible-gpus         How many GPUs should be made visible to each test."
    echo "  -q, --query            List all tests."
    echo "  -h, --help             Print usage."
    echo ""
    echo "Predefined batteries:"
    echo "  XYZ"
    exit $1
}

## Set default arguments if not provided via command-line
SRC_PATH_JAX="/opt/jax"
BUILD_JAXLIB=0
CACHE_TEST_RESULTS=no
ENABLE_X64=-1
VISIBLE_GPUS=""
JOBS_PER_GPU=""

query_tests() {
    set -e -o pipefail
    cd ${SRC_PATH_JAX}
    # FIXME: this ignores .jax_configure.bazelrc lines
    bazel query tests/... |& grep -F '//tests:'
    exit 0
}

set_default() {
    VAR=$1
    VAL=$2
    if [[ -z "${!VAR}" ]]; then
        export $VAR="${VAL}"
    elif [[ "${!VAR}" != "${VAL}" ]]; then
        echo "WARNING: default value of ${VAR} overriden from ${VAL} to ${!VAR}"
    fi
}

print_var() {
    echo "$1: ${!1}"
}

args=$(getopt -o b:qh --long battery:,build-jaxlib,cache-test-results:,disable-x64,enable-x64,reuse-jaxlib,query,tests-per-gpu:,visible-gpus:,help -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi

## parse arguments
eval set -- "$args"
while [ : ]; do
  case "$1" in
    -b | --battery)
        BATTERY="$2"
        shift 2
        ;;
    --src-path-jax)
        SRC_PATH_JAX=$(realpath "$2")
        shift 2
        ;;
    --build-jaxlib)
        BUILD_JAXLIB=1
        shift 1
        ;;
    --cache-test-results)
        CACHE_TEST_RESULTS="$2"
        shift 2
        ;;
    --enable-x64)
        ENABLE_X64=1
        shift 1
        ;;
    --disable-x64)
        ENABLE_X64=0
        shift 1
        ;;
    --reuse-jaxlib)
        BUILD_JAXLIB=0
        shift 1
        ;;
    --tests-per-gpu)
        JOBS_PER_GPU="$2"
        shift 2
        ;;
    --visible-gpus)
        VISIBLE_GPUS="$2"
        shift 2
        ;;
    -q | --query)
        query_tests
        ;;
    -h | --help)
        usage
        ;;
    --)
        shift;
        break 
        ;;
  esac
done

## Set internal variables
if [[ $# -eq 0 ]] && [[ -z "$BATTERY" ]]; then
    echo "No tests specified. Use '-q/--query' to see a list of all available tests."
    exit 1
fi

readarray -t GPU_MEMORIES < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader)
NGPUS="${#GPU_MEMORIES[@]}"
GPU_MEMORIES_MIB=("${GPU_MEMORIES[@]/ MiB/}")

FLAGS=()

if [[ $ENABLE_X64 != 0 ]]; then
    FLAGS+=("--test_env=JAX_ENABLE_X64=true")
else
    FLAGS+=("--test_env=JAX_ENABLE_X64=false")
fi

if [[ $BUILD_JAXLIB -eq 1 ]]; then
    FLAGS+=("--//jax:build_jaxlib=true")
else
    FLAGS+=("--//jax:build_jaxlib=false")
fi
# Added in https://github.com/jax-ml/jax/pull/28870: do not fetch
# nvidia-*-cu1X wheels from PyPI to run tests, use the local installations
FLAGS+=("--//jaxlib/tools:add_pypi_cuda_wheel_deps=false")

# Default parallelism: at least 10GB per test, no more than 4 tests per GPU.
DEFAULT_JOBS_PER_GPU=$(( GPU_MEMORIES_MIB[0] / 10000))
if (( DEFAULT_JOBS_PER_GPU > 4 )); then DEFAULT_JOBS_PER_GPU=4; fi
set_default JOBS_PER_GPU ${DEFAULT_JOBS_PER_GPU}
FLAGS+=(
    "--cache_test_results=${CACHE_TEST_RESULTS}"
    "--test_timeout=600"
    "--test_env=JAX_SKIP_SLOW_TESTS=1"
    "--test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform"
    "--test_env=XLA_PYTHON_CLIENT_PREALLOCATE=false"
    "--test_env=JAX_EXCLUDE_TEST_TARGETS=PmapTest.testSizeOverflow"
    # Default value of 2048 is not big enough for some tests, e.g.
    # //tests/pallas:mgpu_attention_test_gpu; note that this limit is not
    # respected by all codepaths in XLA.
    "--test_env=TF_PER_DEVICE_MEMORY_LIMIT_MB=$(( GPU_MEMORIES_MIB[0] / JOBS_PER_GPU ))"
    "--test_output=errors"
)

case "${BATTERY}" in
    backend-independent)
        set_default VISIBLE_GPUS 1
        FLAGS+=(
	    "//tests:backend_independent_tests"
	    "//tests/pallas:backend_independent_tests"
	    "//tests/mosaic:backend_independent_tests"
        )
        ;;
    multi-gpu)
        # TODO: unclear if JOBS_PER_GPU>1 will cause deadlocks for any tests in the suite
        set_default VISIBLE_GPUS all
        FLAGS+=(
            "--test_tag_filters=multiaccelerator"
            "//tests:gpu_tests"
            "//tests/pallas:gpu_tests"
            "//tests/mosaic:gpu_tests"
        )
        ;;
    single-gpu)
        set_default VISIBLE_GPUS 1
        FLAGS+=(
            "--test_tag_filters=-multiaccelerator"
            "//tests:gpu_tests"
            "//tests/pallas:gpu_tests"
            "//tests/mosaic:gpu_tests"
        )
        ;;
    "")
        # Default if -b/--battery is not passed
        set_default VISIBLE_GPUS 1
        ;;
    *)
        echo "Unknown battery ${BATTERY}"
        usage 1
        ;;
esac

if [[ ${VISIBLE_GPUS} == "1" ]]; then
    NJOBS=$((NGPUS * JOBS_PER_GPU))
    FLAGS+=(
        "--run_under=${SRC_PATH_JAX}/build/parallel_accelerator_execute.sh"
        "--test_env=JAX_ACCELERATOR_COUNT=${NGPUS}"
        "--test_env=JAX_TESTS_PER_ACCELERATOR=${JOBS_PER_GPU}"
    )
elif [[ ${VISIBLE_GPUS} == "all" ]]; then
    NJOBS="${JOBS_PER_GPU}"
else
  echo "Unsupported --visible-gpus value: ${VISIBLE_GPUS}"
  exit 1
fi

FLAGS+=(
    "--local_test_jobs=${NJOBS}"
    "$@"
)

print_var BATTERY
print_var GPU_MEMORIES_MIB
print_var JOBS_PER_GPU
print_var NGPUS
print_var VISIBLE_GPUS

## Run tests
cd ${SRC_PATH_JAX}
set -ex
bazel test "${FLAGS[@]}"
