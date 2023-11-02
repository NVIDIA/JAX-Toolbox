#!/bin/bash

## Parse command-line arguments

usage() {
    echo "Run JAX tests in a container"
    echo ""
    echo "Usage: $0 [OPTION]... TESTS"
    echo "  -b, --battery          Specify predefined test batteries to run."
    echo "  --build-jaxlib         Runs the JAX tests using jaxlib built form source."
    echo "  --reuse-jaxlib         Runs the JAX tests using preinstalled jaxlib. (DEFAULT)"
    echo "  --disable-x64          Disable 64-bit floating point support in JAX (some tests may fail)"
    echo "  --enable-x64           Enable 64-bit floating point support in JAX (DEFAULT, required for some tests)"
    echo "  -q, --query            List all tests."
    echo "  -h, --help             Print usage."
    echo ""
    echo "Predefined batteries:"
    echo "  XYZ"
    exit $1
}

jax_source_dir() {
    dirname `python -c "import jax; print(*jax.__path__)"`
}

query_tests() {
    cd `jax_source_dir`
    python build/build.py --configure_only
    BAZEL=$(find -type f -executable -name "bazel-*")
    $BAZEL query tests/... 2>&1 | grep -F '//tests:'
    exit
}

print_var() {
    echo "$1: ${!1}"
}

args=$(getopt -o b:qh --long build-jaxlib,disable-x64,enable-x64,reuse-jaxlib,query,help -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi

eval set -- "$args"
while [ : ]; do
  case "$1" in
    -b | --battery)
        BATTERY="$2"
        shift 2
        ;;
    --build-jaxlib)
        BUILD_JAXLIB=1
        shift 1
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

if [[ $# -eq 0 ]] && [[ -z "$BATTERY" ]]; then
    echo "No tests specified. Use '-q/--query' to see a list of all available tests."
    exit 1
fi

## Set default arguments if not provided via command-line

BUILD_JAXLIB=${BUILD_JAXLIB:-0}
ENABLE_X64=${ENABLE_X64:-1}

## Set derived variables

NCPUS=$(grep -c '^processor' /proc/cpuinfo)
NGPUS=$(nvidia-smi -L | grep -c '^GPU')

if [[ $ENABLE_X64 != 0 ]]; then
    export JAX_ENABLE_X64=true
else
    export JAX_ENABLE_X64=false
fi

if [[ $BUILD_JAXLIB -eq 1 ]]; then
    BAZEL_TARGET="--//jax:build_jaxlib=true"
else
    BAZEL_TARGET="--//jax:build_jaxlib=false"
fi

for t in $*; do
    if [[ "$t" != "//tests:"* ]]; then
        t="//tests:${t}"
    fi
    BAZEL_TARGET="${BAZEL_TARGET} $t"
done

COMMON_FLAGS=$(cat << EOF
--test_tag_filters=-multiaccelerator 
--test_env=JAX_SKIP_SLOW_TESTS=1
--test_env=JAX_ACCELERATOR_COUNT=${NGPUS}
--test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform
--test_env=XLA_PYTHON_CLIENT_PREALLOCATE=false
--test_output=errors
--java_runtime_version=remotejdk_11
--run_under `jax_source_dir`/build/parallel_accelerator_execute.sh
EOF
)

case "${BATTERY}" in
    large)
        JOBS_PER_GPU=1
        JOBS=$((NGPUS * JOBS_PER_GPU))
        EXTRA_FLAGS="--local_test_jobs=${JOBS} --test_env=JAX_TESTS_PER_ACCELERATOR=${JOBS_PER_GPU} --test_env=JAX_EXCLUDE_TEST_TARGETS=PmapTest.testSizeOverflow"
        BAZEL_TARGET="${BAZEL_TARGET} //tests:image_test_gpu //tests:scipy_stats_test_gpu"
        ;;
    gpu|backend-independent)
        JOBS_PER_GPU=8
        JOBS=$((NGPUS * JOBS_PER_GPU))
        EXTRA_FLAGS="--local_test_jobs=${JOBS} --test_env=JAX_TESTS_PER_ACCELERATOR=${JOBS_PER_GPU} --test_env=JAX_EXCLUDE_TEST_TARGETS=PmapTest.testSizeOverflow"
        BAZEL_TARGET="${BAZEL_TARGET} //tests:gpu_tests"
        ;;
    "")
        JOBS_PER_GPU=4
        JOBS=$((NGPUS * JOBS_PER_GPU))
        EXTRA_FLAGS="--local_test_jobs=${JOBS} --test_env=JAX_TESTS_PER_ACCELERATOR=${JOBS_PER_GPU}"
        ;;
    *)
        echo "Unknown battery ${BATTERY}"
        usage 1
        ;;
esac

print_var NCPUS
print_var NGPUS
print_var BATTERY
print_var JAX_ENABLE_X64
print_var JOBS_PER_GPU
print_var JOBS
print_var BUILD_JAXLIB
print_var BAZEL_TARGET
print_var COMMON_FLAGS
print_var EXTRA_FLAGS

set -ex

## Install dependencies

pip install -r `jax_source_dir`/build/test-requirements.txt
# Reason for manually installing matplotlib:
# https://github.com/google/jax/commit/6b76937c530bd8ee185cc9e1991b3696bd10e831
# https://github.com/google/jax/blob/6bc74d2a9874e1fe93a45191bb829c07dfee04fa/tests/BUILD#L134
pip install matplotlib

## Run tests

cd `jax_source_dir`
python build/build.py --configure_only
BAZEL=$(find -type f -executable -name "bazel-*")
$BAZEL test ${BAZEL_TARGET} ${COMMON_FLAGS} ${EXTRA_FLAGS}
