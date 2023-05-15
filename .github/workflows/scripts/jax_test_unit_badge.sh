#!/bin/bash

## Parse command-line arguments

usage() {
    echo "Usage: $0 [OPTION]..."
    echo "  -f, --files=PATH       Relative path of jax test logs (glob allowed/space sep allowed). Assumes all artifacts are downloaded in current working directory"
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

FAILED_TESTS=$(cat $INPUT_LOG_FILES | grep -c 'FAILED in' || true)
PASSED_TESTS=$(cat $INPUT_LOG_FILES | grep -c 'PASSED in' || true)
TOTAL_TESTS=$((FAILED_TESTS + PASSED_TESTS))
if [[ $FAILED_TESTS == 0 ]]; then
  BADGE_COLOR=brightgreen
else
  if [[ $FAILED_TESTS < $TOTAL_TESTS ]]; then
    BADGE_COLOR=yellow
  else
    BADGE_COLOR=red
  fi
fi
(
cat << EOF
{
  "schemaVersion": 1,
  "label": "${BADGE_LABEL}",
  "message": "${PASSED_TESTS}/${TOTAL_TESTS} passed",
  "color": "${BADGE_COLOR}"
}
EOF
) | tee $JSON_OUTPUT_PATH

