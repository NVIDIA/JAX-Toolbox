#!/bin/bash

usage() {
    echo -e "Usage: ${0} WORKFLOW_IDS..."
    exit 1
}

[ "$#" -ge "1" ] || usage

CONFIGS=("1DP1TP1PP" "8DP1TP1PP" "2DP1TP4PP" "16DP1TP1PP")
ALL_WF_RUNS=($*)

# call download artifacts from this  script's dir
UTIL_DIR="$(dirname "$(readlink --canonicalize -- "${BASH_SOURCE[0]}")")"
bash ${UTIL_DIR}/download_artifacts.sh ${ALL_WF_RUNS[@]}

URLS=()
for WORKFLOW_RUN in ${ALL_WF_RUNS[@]}; do
  pushd ${WORKFLOW_RUN}
  for CFG in ${CONFIGS[@]}; do
    python3 ${UTIL_DIR}/summarize_metrics.py ${CFG}
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
