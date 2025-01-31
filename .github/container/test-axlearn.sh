#!/bin/bash

set -euo pipefail

usage() {
    echo "Run tests in axlearn with specified options."
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "  OPTIONS                       DESCRIPTION"
    echo "  -d, --directory DIR           Directory to run tests in."
    echo "                                Default: 'axlearn/axlearn/common'."
    echo "  -p, --packages PACKAGES       Space-separated list of packages to install via pip."
    echo "                                Default: 'attrs scikit-learn torch evaluate transformers timm wandb grain'."
    echo "  -c, --cuda-devices DEVICES    CUDA devices to use. Default: '0,1'."
    echo "  -t, --test-files PATTERN      Pattern for test files to run."
    echo "                                Default: '*_test.py'."
    echo "  --test-files-list FILE        File containing the list of test files to run."
    echo "  -o, --output DIRECTORY        Output directory for logs and summary."
    echo "                                Default: 'test_runs/<timestamp>'."
    echo "  -h, --help                    Show this help message and exit."
    exit 1
}

# Default values
DIR='axlearn/axlearn/common'
PACKAGES='attrs scikit-learn torch evaluate transformers timm wandb grain'
CUDNN_VERSION='9.7.0.66' # TODO check the cudnn version on compute
CUDA_DEVICES='0,1'
TEST_FILES_PATTERN='*_test.py'
TEST_FILES_LIST=''
OUTPUT_DIRECTORY=''

# Parse args
args=$(getopt -o d:p:c:t:o:h --long directory:,packages:,cuda-devices:,test-files:,test-files-list:,output:,help -- "$@")
if [ $? -ne 0 ]; then
    usage
    exit 1
fi

eval set -- "$args"

while true; do
    case "$1" in
        -d|--directory)
            DIR="$2"
            shift 2
            ;;
        -p|--packages)
            PACKAGES="$2"
            shift 2
            ;;
        -c|--cuda-devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        -t|--test-files)
            TEST_FILES_PATTERN="$2"
            shift 2
            ;;
        --test-files-list)
            TEST_FILES_LIST="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIRECTORY="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# TODO double check what's the best choice
if [ -z "$OUTPUT_DIRECTORY" ]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIRECTORY="test_runs/${timestamp}"
fi
LOG_DIRECTORY="${OUTPUT_DIRECTORY}/logs"

mkdir -p "${LOG_DIRECTORY}"

# Print out config for sanity check
echo "Configuration:"
echo "  Directory: $DIR"
echo "  Packages: $PACKAGES"
echo "  CUDA Devices: $CUDA_DEVICES"
if [ -n "$TEST_FILES_LIST" ]; then
    echo "  Test Files List: $TEST_FILES_LIST"
else
    echo "  Test Files Pattern: $TEST_FILES_PATTERN"
fi
echo "  Output Directory: $OUTPUT_DIRECTORY"
echo ""


cd "$DIR" || exit 1

# Install all the neeeded packages
echo "Installing packages..."
pip install $PACKAGES

# Set CUDA devices
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
echo "Using CUDA devices: $CUDA_VISIBLE_DEVICES"

echo "Running tests..."

if [ -n "$TEST_FILES_LIST" ]; then
    mapfile -t test_files < "$TEST_FILES_LIST"
else
    shopt -s nullglob
    test_files=($TEST_FILES_PATTERN)
    shopt -u nullglob
fi

if [ "${#test_files[@]}" -eq 0 ]; then
    echo "No test files found to run."
    exit 1
fi

for test_file in "${test_files[@]}"; do
    echo "Running: ${test_file}"
    # Ensure the test file exists
    if [ ! -f "${test_file}" ]; then
        echo "${test_file}: NOT FOUND" >> "${SUMMARY_FILE}"
        echo "Test file not found: ${test_file}"
        ((errors++))
        continue
    fi
    log_file_name=$(echo "${test_file%.py}" | sed 's/\//__/g').log
    log_file="${LOG_DIRECTORY}/${log_file_name}"
    # run the tests and save them as *.log
    pytest "${test_file}" -v --capture=tee-sys | tee "${log_file}"
    # TODO parse the logs
    #echo ${PIPESTATUS[0]}
done
