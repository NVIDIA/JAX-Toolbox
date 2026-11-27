#!/usr/bin/env bash
# Diagnose ONE kernel with Nsight Compute.
#
# The target must call cudaProfilerStart()/Stop() or cuProfilerStart()/Stop() around a warmed-up region.
# Run this on a machine with GPU performance-counter access.
# If access is unavailable, ncu reports ERR_NVGPUCTRPERM.
#
# Usage:
#   scripts/ncu_kernel.sh [TARGET_PY] [KERNEL_REGEX] [SET] [OUT]
# Examples:
#   scripts/ncu_kernel.sh                                   # defaults below
#   scripts/ncu_kernel.sh profile_artifacts/target.py gemm basic
#   scripts/ncu_kernel.sh my_app.py "regex_matching_your_kernel" full reports/mykernel
set -euo pipefail

TARGET="${1:-profile_artifacts/target.py}"   # python script to profile
KREGEX="${2:-gemm}"                          # profile only kernels whose name matches this regex
SET="${3:-full}"                             # start with basic; use detailed or full when more context is needed
OUT="${4:-profile_artifacts/step_gemm}"      # report path; ncu appends .ncu-repz or .ncu-rep

# Only reports written during this capture are eligible; older reports in the
# other format may still exist alongside them.
CAPTURE_MARKER="$(mktemp)"
trap 'rm -f "${CAPTURE_MARKER}"' EXIT

# Start disabled so compilation/autotuning is excluded.
# Warmup finishes before the profiler-start call; capture 2 matching launches.
# -k "regex:$KREGEX"               : restrict profiling to matching kernels.
ncu --target-processes all \
    --force-overwrite \
    --profile-from-start off --launch-skip 0 --launch-count 2 \
    -k "regex:${KREGEX}" \
    --set "${SET}" \
    -o "${OUT}" \
    python "${TARGET}"

# Current versions export compressed reports; older versions use .ncu-rep.
REPORT_CANDIDATES=("${OUT}.ncu-repz" "${OUT}.ncu-rep")
case "${OUT}" in
    *.ncu-repz|*.ncu-rep) REPORT_CANDIDATES+=("${OUT}") ;;
esac
REPORTS=()
for candidate in "${REPORT_CANDIDATES[@]}"; do
    if [[ -f "${candidate}" && "${candidate}" -nt "${CAPTURE_MARKER}" ]]; then
        REPORTS+=("${candidate}")
    fi
done
if (( ${#REPORTS[@]} != 1 )); then
    echo "Expected one newly written .ncu-repz or .ncu-rep report for ${OUT}; found ${#REPORTS[@]}." >&2
    exit 1
fi
REPORT="${REPORTS[0]}"

echo
echo "Report written: ${REPORT}"
echo "  open in GUI : ncu-ui \"${REPORT}\""
echo "  open in CLI : ncu --import \"${REPORT}\" --page details"
