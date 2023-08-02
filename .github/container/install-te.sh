#!/bin/bash

## Parse command-line arguments

usage() {
    echo "Usage: $0 [OPTION]..."
    echo '  --defer                When passed, will defer the installation of the main package. Can be installed afterwards with `pip install -r requirements-defer.txt` and any deferred cleanup commands can be run with `bash cleanup.sh`' 
    echo "  -d, --dir=PATH         Path to store TE source. Defaults to /opt/transformer-engine"
    echo "  -f, --from=URL         URL of the TE repo. Defaults to https://github.com/NVIDIA/TransformerEngine.git"
    echo "  -h, --help             Print usage."
    echo "  -r, --ref=REF          Git commit hash or tag name that specifies the version of TE to install. Defaults to HEAD."
    exit $1
}

args=$(getopt -o d:f:hr: --long defer,dir:,from:,help,ref: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi

eval set -- "$args"
while [ : ]; do
  case "$1" in
    --defer)
        DEFER=true
        shift
        ;;
    -d | --dir)
        INSTALL_DIR="$2"
        shift 2
        ;;
    -f | --from)
        TE_REPO="$2"
        shift 2
        ;;
    -h | --help)
        usage
        ;;
    -r | --ref)
        TE_REF="$2"
        shift 2
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

## Set default arguments if not provided via command-line

DEFER=${DEFER:-false}
TE_REF="${TE_REF:-HEAD}"
TE_REPO="${TE_REPO:-https://github.com/NVIDIA/TransformerEngine.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt/transformer-engine}"

echo "Installing TE $TE_REF from $TE_REPO to $INSTALL_DIR"

maybe_defer_pip_install() {
  if [[ "$DEFER" = true ]]; then
    echo "Deferring installation of 'pip install $*'"
    echo "$*" >> /opt/requirements-defer.txt
  else
    pip install $@
  fi
}

set -ex

## Install dependencies

pip install --no-cache-dir pybind11 ninja packaging

## Install TE

git clone ${TE_REPO} ${INSTALL_DIR}
cd ${INSTALL_DIR}
git checkout ${TE_REF}
git submodule init
git submodule update --recursive
maybe_defer_pip_install -e ${INSTALL_DIR}
