#!/bin/bash

set -uo pipefail

# HELPER FUNCTIONS
usage() {
    # Function to handle all the inputs
    echo "Run tests in axlearn with specified options."
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "  OPTIONS                       DESCRIPTION"
    echo "  -d, --directory DIR           Directory to run tests in."
    echo "                                Default: 'axlearn/axlearn/common'."
    echo "  -t, --test-files FILES        Pattern for test files to run."
    echo "                                Default: '*_test.py'."
    echo "  -o, --output DIRECTORY        Output directory for logs and summary."
    echo "                                Default: 'test_runs/<timestamp>'."
    echo "  -h, --help                    Show this help message and exit."
    exit 1
}

run_tests() {
    # Function to run tests for AXLearn
    local env_spec=$1
    local marker=$2
    local suffix=$3

    local junit="log_${suffix}.xml"
    local log="log_${suffix}.log"

    cmd="${env_spec:+${env_spec} }pytest -m \"${marker}\" ${final_test_files[@]}\
    --capture=tee-sys -v \
    --junit-xml=${LOG_DIRECTORY}/${junit} | tee ${LOG_DIRECTORY}/${log}"
    echo "Running command ${cmd}"
    eval "${cmd}"
}

# DEFAULT VALUES
DIR='axlearn/axlearn/common'
TEST_FILES=()
OUTPUT_DIRECTORY=''

# INPUT PARSING
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

echo "Configuration:"
echo "  Directory: $DIR"
if [ "${#TEST_FILES[@]}" -gt 0 ]; then
    echo "  Test Files:"
    for f in "${TEST_FILES[@]}"; do
        echo "    $f"
    done
else
    echo "  Test Files Pattern: '*_test.py' (default)"
fi

cd "$DIR" || exit 1

# DEPENDENCIES
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm transformers scikit-learn tensorflow_io grain evaluate wandb
echo "Downloading input data..."
mkdir -p /opt/axlearn/axlearn/data/tokenizers/sentencepiece
mkdir -p /opt/axlearn/axlearn/data/tokenizers/bpe
curl https://huggingface.co/t5-base/resolve/main/spiece.model -o /opt/axlearn/axlearn/data/tokenizers/sentencepiece/t5-base
curl https://huggingface.co/FacebookAI/roberta-base/raw/main/merges.txt -o /opt/axlearn/axlearn/data/tokenizers/bpe/roberta-base-merges.txt
curl https://huggingface.co/FacebookAI/roberta-base/raw/main/vocab.json -o /opt/axlearn/axlearn/data/tokenizers/bpe/roberta-base-vocab.json

# RETRIEVE TEST FILES
if [ "${#TEST_FILES[@]}" -eq 0 ]; then
    TEST_FILES=("*_test.py")
fi

expanded_test_files=()
for pattern in "${TEST_FILES[@]}"; do
    # retrieve all the files
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

EXCLUDE_PATTERNS=("array_serialization_test.py"
    "host_array_test.py"
    "utils_test.py"
    "inference_test.py" # rocm test
    "t5_test.py" # tensorflow bug
    "input_t5_test.py"
    "layers_test.py" # tensorflow bug
    "checkpointer_orbax_test.py"
    "checkpointer_test.py"
    "input_glue_test.py"
    )
final_test_files=()

for test_file in "${expanded_test_files[@]}"; do
    exclude=false
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

# RUN TESTS
runs=(
  "JAX_PLATFORMS='cuda'|not (gs_login or tpu or high_cpu or fp64 or for_8_devices)|base"
  "JAX_PLATFORMS='gpu' JAX_ENABLE_X64=1|fp64|fp64"
  "JAX_PLATFORMS='gpu'|for_8_devices|8dev"
)

for spec in "${runs[@]}"; do
    IFS='|' read -r env_spec marker suffix <<< "${spec}"
    echo "Running tests with ${env_spec}, ${marker}, ${suffix}"
    run_tests "${env_spec}" "${marker}" "${suffix}"
    echo "Test run"
done

# SUMMARY STATUS
passed=0
failed=0
skipped=0
for log in ${LOG_DIRECTORY}/log_*.log; do
    count_pass=$(grep -Eo '[0-9]+ passed' "${log}" | awk '{print $1}' || true)
    count_fail=$(grep -Eo '[0-9]+ failed' "${log}" | awk '{print $1}' || true)
    count_skipped=$(grep -Eo '[0-9]+ skipped' "${log}" | awk '{print $1}' || true)
    # in case of None
    count_pass=${count_pass:-0}
    count_fail=${count_fail:-0}
    count_skipped=${count_skipped:-0}
    # count all the tests
    (( passed += count_pass ))
    (( failed += count_fail ))
    (( skipped += count_skipped ))
done

echo "Total number of passed tests ${passed}"
echo "Total number of failed tests ${failed}"
echo "Total number of skipped tests ${skipped}"
# add those to summary.txt and we're using it for extracting values
echo "PASSED: ${passed} FAILED: ${failed} SKIPPED: ${skipped}" >> ${LOG_DIRECTORY}/summary.txt
