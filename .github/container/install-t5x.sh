#!/bin/bash

## Parse command-line arguments

usage() {
    echo "Usage: $0 [OPTION]..."
    echo '  --defer                When passed, will defer the installation of the main package. Can be installed afterwards with `pip install -r requirements-defer.txt` and any deferred cleanup commands can be run with `bash cleanup.sh`' 
    echo "  -d, --dir=PATH         Path to store T5X source. Defaults to /opt"
    echo "  -f, --from=URL         URL of the T5X repo. Defaults to https://github.com/google-research/t5x.git"
    echo "  -h, --help             Print usage."
    echo "  -r, --ref=REF          Git commit hash or tag name that specifies the version of T5X to install. Defaults to HEAD."
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
        T5X_REPO="$2"
        shift 2
        ;;
    -h | --help)
        usage
        ;;
    -r | --ref)
        T5X_REF="$2"
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
T5X_REF="${T5X_REF:-HEAD}"
T5X_REPO="${T5X_REPO:-https://github.com/google-research/t5x.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt}"

echo "Installing T5X $T5X_REF from $T5X_REPO to $INSTALL_DIR"

maybe_defer_cleanup() {
  if [[ "$DEFER" = true ]]; then
    echo "# Cleanup from: $0"
    echo "$*" >> /opt/cleanup.sh
  else
    $@
  fi
}

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

apt-get update
apt-get install -y \
    build-essential \
    cmake \
    clang \
    git

## Install T5X

T5X_INSTALLED_DIR=${INSTALL_DIR}/t5x

git clone ${T5X_REPO} ${T5X_INSTALLED_DIR}
cd ${T5X_INSTALLED_DIR}
git checkout ${T5X_REF}
# We currently require installing editable (-e) to build a distribution since
# we edit the source in place and do not re-install
maybe_defer_pip_install -e ${T5X_INSTALLED_DIR}[gpu]

maybe_defer_cleanup apt-get autoremove -y
maybe_defer_cleanup apt-get clean
maybe_defer_cleanup rm -rf /var/lib/apt/lists/*
maybe_defer_cleanup rm -rf ~/.cache/pip/
