#!/bin/bash

## Parse command-line arguments

usage() {
    echo "Usage: $0 [OPTION]..."
    echo "  -f, --files=PATH       Relative path of pytest reportlogs (glob allowed/space sep allowed). Assumes all artifacts are downloaded in current working directory"
    echo "  -j, --jsonout=PATH     Output json file with badge metadata"
    echo "  -l, --label=PATH       Badge label name"
    echo "  -h, --help             Print usage."
    exit $1
}

args=$(getopt -o f:j:l:h --long files:,jsonout:,label:,help -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi

eval set -- "$args"
while [ : ]; do
  case "$1" in
    -f | --files)
        INPUT_LOG_FILES="$2"
        shift 2
        ;;
    -j | --jsonout)
        JSON_OUTPUT_PATH="$2"
        shift 2
        ;;
    -l | --label)
        BADGE_LABEL="$2"
        shift 2
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

if [[ $# -ge 1 ]]; then
    echo "Un-recognized argument: $*" && echo
    usage 1
fi

set -eou pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/helpers.env

if [[ -z "${INPUT_LOG_FILES+x}" ]]; then
  eecho Need to set -f/--files
  usage 1
elif [[ -z "${JSON_OUTPUT_PATH+x}" ]]; then
  eecho Need to set -j/--jsonout
  usage 1
elif [[ -z "${BADGE_LABEL+x}" ]]; then
  eecho Need to set -l/--label
  usage 1
fi

decho "--files="$INPUT_LOG_FILES

all_outcomes() {
  cat $INPUT_LOG_FILES | jq -r '. | select((.["$report_type"] == "TestReport") and (.when == "call")) | .outcome'
}
cnt_type() {
  cat $INPUT_LOG_FILES | jq '. | select((.["$report_type"] == "TestReport") and (.when == "call") and (.outcome | contains("'${1}'"))) | .outcome' | wc -l
}
SKIPPED_TESTS=$(cnt_type skipped)
FAILED_TESTS=$(cnt_type failed)
PASSED_TESTS=$(cnt_type passed)
TOTAL_TESTS=$(all_outcomes | wc -l)
decho "Unit test breakdown:"
all_outcomes | sort | uniq -c
if [[ $FAILED_TESTS -eq 0 ]] && [[ $TOTAL_TESTS -gt 0 ]]; then
  BADGE_COLOR=brightgreen
else
  if [[ $PASSED_TESTS -eq 0 ]]; then
    BADGE_COLOR=red
  else
    BADGE_COLOR=yellow
  fi
fi
(
cat << EOF
{
  "schemaVersion": 1,
  "label": "${BADGE_LABEL}",
  "message": "${PASSED_TESTS}/${SKIPPED_TESTS}/${FAILED_TESTS} pass/skip/fail",
  "color": "${BADGE_COLOR}"
}
EOF
) | tee $JSON_OUTPUT_PATH
