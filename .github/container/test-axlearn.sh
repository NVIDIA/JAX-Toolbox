#!/bin/bash

set -uo pipefail

# HELPER FUNCTIONS
usage() {
    echo "Run tests in axlearn with specified options."
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "  -d, --directory DIR           Directory to run tests in."
    echo "                                Default: '/opt/axlearn/axlearn/common'."
    echo "  -t, --test-files FILES        Pattern for test files to run."
    echo "                                Default: '*_test.py'."
    echo "  -o, --output DIRECTORY        Output directory for logs and summary."
    echo "                                Default: 'output/<timestamp>'."
    echo "  -h, --help                    Show this help message and exit."
    exit 1
}

run_tests() {
    local env_spec=$1
    local marker=$2
    local suffix=$3
    shift 3
    local -a test_files=("$@")

    local junit="log_${suffix}.xml"
    local log="log_${suffix}.log"

    # Build --deselect args for individual test cases we want to skip
    local deselect_args=""
    for d in "${DESELECT_TESTS[@]}"; do
        deselect_args+=" --deselect=${d}"
    done

    cmd="${env_spec:+${env_spec} }pytest -m \"${marker}\" ${test_files[@]} \
    ${deselect_args} \
    --capture=tee-sys -v \
    --junit-xml=${LOG_DIRECTORY}/${junit} | tee ${LOG_DIRECTORY}/${log}"
    echo "Running command: ${cmd}"
    eval "${cmd}"
}

# DEFAULT VALUES
DIR='/opt/axlearn/axlearn/common'
TEST_FILES=()
OUTPUT_DIRECTORY=''

# INPUT PARSING
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -d|--directory)
            [[ -z "${2:-}" ]] && { echo "Error: --directory requires an argument."; usage; }
            DIR="$2"; shift 2 ;;
        -t|--test-files)
            shift
            [[ $# -eq 0 ]] && { echo "Error: --test-files requires at least one file pattern."; usage; }
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                TEST_FILES+=("$1"); shift
            done ;;
        -o|--output)
            [[ -z "${2:-}" ]] && { echo "Error: --output requires an argument."; usage; }
            OUTPUT_DIRECTORY="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

cd "$DIR"

if [ -z "$OUTPUT_DIRECTORY" ]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIRECTORY="output/${timestamp}"
fi
LOG_DIRECTORY="${OUTPUT_DIRECTORY}/logs"
mkdir -p "${LOG_DIRECTORY}"

# DEPENDENCIES
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm transformers scikit-learn grain evaluate prefixed wandb

echo "Downloading input data..."
mkdir -p /opt/axlearn/axlearn/data/tokenizers/sentencepiece
mkdir -p /opt/axlearn/axlearn/data/tokenizers/bpe
curl https://huggingface.co/t5-base/resolve/main/spiece.model \
    -o /opt/axlearn/axlearn/data/tokenizers/sentencepiece/t5-base
curl https://huggingface.co/FacebookAI/roberta-base/raw/main/merges.txt \
    -o /opt/axlearn/axlearn/data/tokenizers/bpe/roberta-base-merges.txt
curl https://huggingface.co/FacebookAI/roberta-base/raw/main/vocab.json \
    -o /opt/axlearn/axlearn/data/tokenizers/bpe/roberta-base-vocab.json

# RETRIEVE TEST FILES
if [ "${#TEST_FILES[@]}" -eq 0 ]; then
    TEST_FILES=("*_test.py")
fi

expanded_test_files=()
for pattern in "${TEST_FILES[@]}"; do
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

# ============================================================
# EXCLUSIONS — file-level (matched by basename)
# ============================================================
# Each group documents WHY a test is excluded so we can revisit later.
EXCLUDE_PATTERNS=(
    # --- Pre-existing exclusions (TF / optax / orbax bugs, manual-only, etc.) ---
    "array_serialization_test.py"
    "t5_test.py"                          # tensorflow bug
    "loss_test.py"                        # optax bug
    "input_t5_test.py"
    "layers_test.py"                      # tensorflow bug
    "checkpointer_orbax_test.py"
    "checkpointer_orbax_emergency_test.py"
    "checkpointer_test.py"
    "input_glue_test.py"
    "deberta_test.py"
    "quantizer_test.py"
    "test_utils_test.py"
    "update_transformation_test.py"
    "env_test.py"
    "causal_lm_test.py"
    "gradient_accumulation_test.py"
    "file_system_test.py"
    "compiler_options_test.py"            # tpu only
    "metrics_correlation_test.py"         # manual only
    "metrics_glue_test.py"
    "ssm_test.py"
    "summary_test.py"                     # wandb test
    "param_converter_test.py"
    "attention_test.py"                   # assertion errors to fix
    # --- for_8_devices tests (we no longer run that suite) ---
    "gda_test.py"
    "input_base_test.py"
    "input_dispatch_test.py"
    "trainer_test.py"
    # NOTE: cloud/gcp/runners/utils_test.py and common/utils_test.py share
    # basename "utils_test.py". We excluded utils_test.py for the runners
    # case below, which also drops the for_8_devices common/utils_test.py.
    "utils_test.py"
    # --- Missing optional packages (prometheus_client, ml_goodput_measurement) ---
    # utils_test.py is already covered above (basename collision).
    "measurement_test.py"
    # --- JAX API breakage (jax.ad_checkpoint.checkpoint removed; jnp.clip a_max) ---
    "pipeline_test.py"
    "rnn_test.py"
    # --- transformers 5.x API incompatibilities ---
    "bert_test.py"
    "convolution_test.py"
    "distilbert_test.py"
    "dit_test.py"
    "embedding_test.py"
    "encoder_decoder_test.py"
    "encoder_test.py"
    "lora_test.py"
    "repeat_test.py"
    "splade_test.py"
    "utils_text_dual_encoder_test.py"
    "input_text_test.py"                  # BasicTokenizer import broken
    "neural_retrieval_test.py"            # modeling_flax_utils missing
    # --- sklearn 1.8 private API breakage ---
    "metrics_classification_test.py"
    # --- Roberta tokenizer config schema mismatch (vocab_file -> vocab/merges) ---
    "input_reading_comprehension_test.py"
)

# ============================================================
# DESELECTIONS — individual test cases (kept here for reference,
# currently empty since we drop whole files above. Add entries like:
#   "axlearn/common/measurement_test.py::UtilsTest::test_initialize_gcp_goodput_recorder"
# if you ever want finer-grained exclusion instead of dropping the file.)
# ============================================================
DESELECT_TESTS=()

final_test_files=()
for test_file in "${expanded_test_files[@]}"; do
    exclude=false
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        [[ -z "$pattern" ]] && continue
        if [[ "$(basename "$test_file")" == "$(basename "$pattern")" ]]; then
            exclude=true
            break
        fi
    done
    if [ "$exclude" = false ]; then
        final_test_files+=("$test_file")
    fi
done

echo "Running ${#final_test_files[@]} test files after exclusions."

# ============================================================
# TEST RUNS
# ============================================================
runs=(
  "|not (gs_login or tpu or high_cpu or fp64 or for_8_devices)|base"
  "JAX_ENABLE_X64=1|fp64|fp64"
)
crashed=0
for spec in "${runs[@]}"; do
    IFS='|' read -r env_spec marker suffix <<< "${spec}"
    echo "Running tests with env='${env_spec}', marker='${marker}', suffix='${suffix}'"
    run_tests "${env_spec}" "${marker}" "${suffix}" "${final_test_files[@]}"
    status=$?
    # pytest exits 0 (all passed) or 1 (test failures, counted from the logs
    # below). Anything else (2=interrupted, 3=internal error, 4=usage error,
    # 5=no tests collected, >128=crash/signal) never prints a summary line, so
    # the grep-based counts below would report a false green.
    if [ "${status}" -gt 1 ]; then
        echo "Test run '${suffix}' exited with unexpected status ${status}."
        crashed=1
    fi
    echo "Test run '${suffix}' done."
done

# ============================================================
# SUMMARY
# ============================================================
passed=0
failed=0
skipped=0
for log in ${LOG_DIRECTORY}/log_*.log; do
    count_pass=$(grep -Eo '[0-9]+ passed'  "${log}" | awk '{s+=$1} END{print s+0}')
    count_fail=$(grep -Eo '[0-9]+ failed'  "${log}" | awk '{s+=$1} END{print s+0}')
    count_err=$( grep -Eo '[0-9]+ error'   "${log}" | awk '{s+=$1} END{print s+0}')
    count_skip=$(grep -Eo '[0-9]+ skipped' "${log}" | awk '{s+=$1} END{print s+0}')
    (( passed  += count_pass ))
    (( failed  += count_fail ))
    (( failed  += count_err  ))
    (( skipped += count_skip ))
done

echo "Total passed:  ${passed}"
echo "Total failed:  ${failed}"
echo "Total skipped: ${skipped}"
echo "PASSED: ${passed} FAILED: ${failed} SKIPPED: ${skipped}" >> "${LOG_DIRECTORY}/summary.txt"

if [ ${failed} -gt 0 ] || [ ${crashed} -ne 0 ]; then
    echo "Some tests failed or crashed. Check the logs in ${LOG_DIRECTORY} for details."
    exit 1
else
    echo "All tests passed successfully."
fi
