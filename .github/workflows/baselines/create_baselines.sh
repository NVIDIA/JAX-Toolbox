#!/bin/bash

set -euo pipefail
usage() {
    echo -e "Usage: ${0} upstream-pax|pax|upstream-t5x|t5x|maxtext WORKFLOW_IDS..."
    exit 1
}

[ "$#" -ge "2" ] || usage

TYPE=$1
if [[ "$TYPE" == "upstream-pax" ]]; then
    CONFIGS=(
        "16DP1FSDP1TP1PP"
        "1DP1FSDP1TP1PP"
        "1DP2FSDP4TP1PP_single_process"
        "1DP8FSDP1TP1PP"
        "2DP1FSDP1TP4PP"
        "2DP1FSDP2TP4PP"
        "4DP1FSDP2TP1PP"
        "8DP1FSDP1TP1PP"
        "8DP1FSDP1TP1PP_eval"
        "8DP1FSDP1TP1PP_single_process"
    )
    # OUTPUT_DIR should be relative to this script
    OUTPUT_DIR=PAX_MGMN/upstream
elif [[ "$TYPE" == "pax" ]]; then
    CONFIGS=(
        "16DP1FSDP1TP1PP_TE"
        "1DP1FSDP1TP1PP_TE"
        "1DP2FSDP4TP1PP_single_process_TE"
        "1DP8FSDP1TP1PP_TE"
        "2DP1FSDP1TP4PP"
        "2DP1FSDP2TP4PP"
        "4DP1FSDP2TP1PP"
        "4DP1FSDP2TP1PP_TE"
        "8DP1FSDP1TP1PP"
        "8DP1FSDP1TP1PP_TE"
        "8DP1FSDP1TP1PP_eval_TE"
        "8DP1FSDP1TP1PP_single_process_TE"
    )
    OUTPUT_DIR=PAX_MGMN/rosetta
elif [[ "$TYPE" == "upstream-t5x" ]]; then
    CONFIGS=(
        "1G1N"
        "1G2N"
        "1P1G"
        "1P2G"
        "1P4G"
        "1P8G"
        "2G1N"
        "2G2N"
        "4G1N"
        "4G2N"
        "8G1N"
        "8G2N"
    )
    OUTPUT_DIR=T5X_MGMN/upstream
elif [[ "$TYPE" == "t5x" ]]; then
    CONFIGS=(
        "1N1G-te-1"
        "1N8G-te-1"
        "1P1G_te-0"
        "1P1G_te-1"
        "1P8G_te-1"
        "2N2G_te-0"
        "2N8G-te-1"
        "VIT1G1N"
        "VIT1G2N"
        "VIT1P8G"
        "VIT8G1N"
        "VIT8G2N"
    )
    OUTPUT_DIR=T5X_MGMN/rosetta
elif [[ "$TYPE" == "maxtext" ]]; then
    CONFIGS=("1DP1FSDP1TP1PP" "1DP1FSDP8TP1PP" "1DP2FSDP4TP1PP_single_process" "1DP4FSDP2TP1PP" "1DP8FSDP1TP1PP" "2DP2FSDP2TP1PP" "4DP2FSDP2TP1PP")
    OUTPUT_DIR=MAXTEXT/upstream
else
    usage
fi
ALL_WF_RUNS=(${@:2})

# call download artifacts from this  script's dir
UTIL_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
bash ${UTIL_DIR}/download_artifacts.sh ${ALL_WF_RUNS[@]}

URLS=()
for WORKFLOW_RUN in ${ALL_WF_RUNS[@]}; do
  pushd ${WORKFLOW_RUN}
  for CFG in ${CONFIGS[@]}; do
    if [[ $(find . -mindepth 1 -maxdepth 2 -type d -name $CFG | wc -l) -ne 1 ]]; then
      echo "Expected one artifact to have a '$CFG' dir under '$PWD', but found $(find . -mindepth 1 -maxdepth 2 -type d -name $CFG)"
      exit 1
    fi
    # There should only be one directory with this config so this glob should match
    pushd $(dirname */$CFG)
    if [[ $TYPE == "upstream-pax" || $TYPE == "pax" ]]; then
        python3 ${UTIL_DIR}/summarize_metrics.py ${CFG}
    elif [[ $TYPE == "upstream-t5x" || $TYPE == "t5x" ]]; then
        python3 ${UTIL_DIR}/summarize_metrics.py ${CFG} --perf_summary_name "timing/steps_per_second"
    elif [[ $TYPE == "maxtext" ]]; then
        python3 ${UTIL_DIR}/summarize_metrics.py ${CFG} --loss_summary_name "learning/loss" --perf_summary_name "perf/step_time_seconds"
    fi
    popd
  done
  popd
  URLS+=("\"https://api.github.com/repos/NVIDIA/JAX-Toolbox/actions/runs/${WORKFLOW_RUN}/artifacts\"")
done

for CFG in ${CONFIGS[@]}; do
  # Average metrics data for this config
  run_dir_paths=""
  for run_id in ${ALL_WF_RUNS[@]}; do
    # Even though it's a glob, there should be only one b/c of the check above
    run_dir_paths+=" $(dirname $run_id/*/$CFG)"
  done

  python3 ${UTIL_DIR}/average_baselines.py --config ${CFG} --run_dirs $run_dir_paths --output_dir $OUTPUT_DIR
  
  # Append date and workflow sources
  cat <<< $(jq -rc '. += {"run_urls":['$(IFS=, ; echo "${URLS[*]}")'], "date":"'$(date +%Y-%m-%d)'"}' "${CFG}.json") > $OUTPUT_DIR/${CFG}.json
done
