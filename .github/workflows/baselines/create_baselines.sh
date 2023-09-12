#!/bin/bash

set -euo pipefail
usage() {
    echo -e "Usage: ${0} pax|t5x WORKFLOW_IDS..."
    exit 1
}

[ "$#" -ge "2" ] || usage

TYPE=$1
if [[ "$TYPE" == "pax" ]]; then
    CONFIGS=("1DP1TP1PP" "8DP1TP1PP" "2DP1TP4PP" "16DP1TP1PP")
elif [[ "$TYPE" == "t5x" ]]; then
    CONFIGS=("1G1N" "1G2N" "1P1G" "1P2G" "1P4G" "1P8G" "2G1N" "2G2N" "4G1N" "4G2N" "8G1N" "8G2N")
else
    usage
fi
ALL_WF_RUNS=(${@:2})

# call download artifacts from this  script's dir
UTIL_DIR="$(dirname "$(readlink --canonicalize -- "${BASH_SOURCE[0]}")")"
bash ${UTIL_DIR}/download_artifacts.sh ${ALL_WF_RUNS[@]}

URLS=()
for WORKFLOW_RUN in ${ALL_WF_RUNS[@]}; do
  pushd ${WORKFLOW_RUN}
  for CFG in ${CONFIGS[@]}; do
    if [[ $TYPE == "pax" ]]; then
        python3 ${UTIL_DIR}/summarize_metrics.py ${CFG}
    elif [[ $TYPE == "t5x" ]]; then
        python3 ${UTIL_DIR}/summarize_metrics.py ${CFG} --perf_summary_name "timing/steps_per_second"
    fi
  done
  popd
  URLS+=("\"https://api.github.com/repos/NVIDIA/JAX-Toolbox/actions/runs/${WORKFLOW_RUN}/artifacts\"")
done

for CFG in ${CONFIGS[@]}; do
  # Average metrics data for this config
  python3 ${UTIL_DIR}/average_baselines.py ${CFG} ${ALL_WF_RUNS[@]}
  
  # Append date and workflow sources
  cat <<< $(jq -rc '. += {"run_urls":['$(IFS=, ; echo "${URLS[*]}")'], "date":"'$(date +%Y-%m-%d)'"}' "${CFG}.json") > ${CFG}.json
done
