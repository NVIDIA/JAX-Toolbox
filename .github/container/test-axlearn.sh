#!/bin/bash

set -uo pipefail

usage() {
    echo "Run tests in axlearn with specified options."
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "  OPTIONS                       DESCRIPTION"
    echo "  -d, --directory DIR           Directory to run tests in."
    echo "                                Default: 'axlearn/axlearn/common'."
    echo "  -c, --cuda-devices DEVICES    CUDA devices to use. Default: '0,1,2,3,4,5,6,7'."
    echo "  -t, --test-files FILES        Pattern for test files to run."
    echo "                                Default: '*_test.py'."
    echo "  -o, --output DIRECTORY        Output directory for logs and summary."
    echo "                                Default: 'test_runs/<timestamp>'."
    echo "  -k, --k8s                     Whether to run on a Kubernetes cluster."
    echo "  -h, --help                    Show this help message and exit."
    exit 1
}

# Default values
DIR='axlearn/axlearn/common'
CUDA_DEVICES='0,1,2,3,4,5,6,7'
TEST_FILES=()
OUTPUT_DIRECTORY=''
K8S=false

# Parse args manually
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -d|--directory)
            if [[ -z "$2" ]]; then
                echo "Error: --directory requires an argument."
                usage
            fi
            DIR="$2"
            shift 2
            ;;
        -c|--cuda-devices)
            if [[ -z "$2" ]]; then
                echo "Error: --cuda-devices requires an argument."
                usage
            fi
            CUDA_DEVICES="$2"
            shift 2
            ;;
        -t|--test-files)
            shift
            # Collect all arguments until the next option (starting with '-')
            if [[ $# -eq 0 ]]; then
                echo "Error: --test-files requires at least one file pattern."
                usage
            fi
            echo "Option -t|--test-files with arguments:"
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                echo "  $1"
                TEST_FILES+=("$1")
                shift
            done
            ;;
        -o|--output)
            if [[ -z "$2" ]]; then
                echo "Error: --output requires an argument."
                usage
            fi
            OUTPUT_DIRECTORY="$2"
            shift 2
            ;;
        -k|--k8s)
            K8S=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done


if [ -z "$OUTPUT_DIRECTORY" ]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIRECTORY="test_runs/${timestamp}"
fi
LOG_DIRECTORY="${OUTPUT_DIRECTORY}/logs"

mkdir -p "${LOG_DIRECTORY}"

# Print out config for sanity check
echo "Configuration:"
echo "  Directory: $DIR"
echo "  CUDA Devices: $CUDA_DEVICES"
if [ "${#TEST_FILES[@]}" -gt 0 ]; then
    echo "  Test Files:"
    for f in "${TEST_FILES[@]}"; do
        echo "    $f"
    done
else
    echo "  Test Files Pattern: '*_test.py' (default)"
fi
echo "  Output Directory: $OUTPUT_DIRECTORY"
echo "  Kubernetes mode: $K8S"
echo "" 


cd "$DIR" || exit 1

# Set CUDA devices
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
echo "Using CUDA devices: $CUDA_VISIBLE_DEVICES"

echo "Running tests..."

# If we are on Kubernetes, install torch
if [ "$K8S" = true ]; then
    uname -a 
    python --version
    #pip install torch  # install cpu version 
    #nvidia-cudnn-cu12==9.7.0.66
fi

if [ "${#TEST_FILES[@]}" -eq 0 ]; then
    TEST_FILES=("*_test.py")
fi
expanded_test_files=()
for pattern in "${TEST_FILES[@]}"; do
    # Use globbing to expand pattern
    files=( $pattern )
    if [ "${#files[@]}" -gt 0 ]; then
        expanded_test_files+=( "${files[@]}" )
    else
        echo "Warning: No files matched pattern '$pattern'"
    fi
done


if [ "${#expanded_test_files[@]}" -eq 0 ]; then
    echo "No test files found to run."
    exit 1
fi

echo "These are the test files:"
for f in "${expanded_test_files[@]}"; do
    echo "  $f"
done

# Get the directory where the script is located
#SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EXCLUDE_LIST_FILE="$DIR/exclusion_list.txt"
EXCLUDE_PATTERNS=()

if [ -f "$EXCLUDE_LIST_FILE" ]; then
    echo "Reading exclusion list from '$EXCLUDE_LIST_FILE'"
    mapfile -t EXCLUDE_PATTERNS < "$EXCLUDE_LIST_FILE"
else
    echo "Exclusion list file not found at '$EXCLUDE_LIST_FILE'"
fi
echo "Exclude patterns read:"
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    echo  "$pattern"
done

#expanded_test_files=( "${expanded_test_files[@]:0:10}" )
# we are skipping some tests as there's still wip by Apple
final_test_files=()

for test_file in "${expanded_test_files[@]:0:5}"; do 
    exclude=false 
    #echo $test_file
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do 
        if [[ "$(basename "$test_file")" == "$(basename "$pattern")" ]]; then
            exclude=true 
            break 
        fi 
    done 
    if [ "$exclude" = false ]; then 
        final_test_files+=("$test_file")
    fi 
done

# Initialize counters
failures=0
passed=0
SUMMARY_FILE="${OUTPUT_DIRECTORY}/summary.txt"


for test_file in "${final_test_files[@]}"; do
    echo "Running: ${test_file}"
    # Ensure the test file exists
    if [ ! -f "${test_file}" ]; then
        echo "${test_file}: NOT FOUND" >> "${SUMMARY_FILE}"
        echo "Test file not found: ${test_file}"
        continue
    fi
    log_file_name=$(echo "${test_file%.py}" | sed 's/\//__/g').log
    log_file="${LOG_DIRECTORY}/${log_file_name}"
    # run the tests and save them as *.log
    pytest "${test_file}" --capture=tee-sys | tee "${log_file}"
    # TODO parse the logs?
    exit_code=${PIPESTATUS[0]}
    echo $exit_code
    if [ $exit_code -eq 0 ]; then
        echo "${test_file}: PASSED" >> "${SUMMARY_FILE}"
        ((passed++))
    else
        echo "${test_file}: FAILED (Exit code: $exit_code)" >> "${SUMMARY_FILE}"
        ((failures++))
    fi
    echo ""
done

echo $passed 
echo $failures