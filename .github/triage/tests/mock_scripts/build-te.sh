#!/bin/bash
set -euo pipefail
CCACHE=0
args=$(getopt -o h --long ccache -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi
eval set -- "$args"
while [ : ]; do
    case "$1" in
        --ccache)
            CCACHE=1
            shift 1
            ;;
        --)
            shift;
            break
            ;;
        *)
            echo "UNKNOWN OPTION $1"
            exit 2
    esac
done
if [[ "${JAX_TOOLBOX_TRIAGE_BUILD_TE_POISON_PILL-0}" == "1" ]]; then
    exit 3
fi
if [[ "${CCACHE}" == "1" ]]; then
    if [[ "${CCACHE_SENTINEL}" != "42" ]]; then
        exit 4
    fi
fi
exit 0
