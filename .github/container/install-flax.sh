#!/bin/bash

## Parse command-line arguments

usage() {
    echo "Usage: $0 [OPTION]..."
    echo '  --defer                When passed, will defer the installation of the main package. Can be installed afterwards with `pip install -r requirements-defer.txt` and any deferred cleanup commands can be run with `bash cleanup.sh`' 
    echo "  -d, --dir=PATH         Path to store flax source. Defaults to /opt/flax"
    echo "  -f, --from=URL         URL of the flax repo. Defaults to https://github.com/google/flax.git"
    echo "  -h, --help             Print usage."
    echo "  -r, --ref=REF          Git commit hash or tag name that specifies the version of flax to install. Defaults to HEAD."
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
        FLAX_REPO="$2"
        shift 2
        ;;
    -h | --help)
        usage
        ;;
    -r | --ref)
        FLAX_REF="$2"
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
FLAX_REF="${FLAX_REF:-HEAD}"
FLAX_REPO="${FLAX_REPO:-https://github.com/google/flax.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt/flax}"

echo "Installing flax $FLAX_REF from $FLAX_REPO to $INSTALL_DIR"

maybe_defer_pip_install() {
  if [[ "$DEFER" = true ]]; then
    echo "Deferring installation of 'pip install $*'"
    echo "$*" >> /opt/requirements-defer.txt
  else
    pip install $@
  fi
}

set -ex

## Install flax

git clone ${FLAX_REPO} ${INSTALL_DIR}
cd ${INSTALL_DIR}
git checkout ${FLAX_REF}
maybe_defer_pip_install ${INSTALL_DIR}