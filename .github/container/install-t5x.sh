#!/bin/bash

## Parse command-line arguments

usage() {
    echo "Usage: $0 [OPTION]..."
    echo "  -d, --dir=PATH         Path to store T5X source. Defaults to /opt"
    echo "  -f, --from=URL         URL of the T5X repo. Defaults to https://github.com/google-research/t5x.git"
    echo "  -h, --help             Print usage."
    echo "  -r, --ref=REF          Git commit hash or tag name that specifies the version of T5X to install. Defaults to HEAD."
    exit $1
}

args=$(getopt -o d:f:hr: --long dir:,from:,help,ref: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi

eval set -- "$args"
while [ : ]; do
  case "$1" in
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

T5X_REF="${T5X_REF:-HEAD}"
T5X_REPO="${T5X_REPO:-https://github.com/google-research/t5x.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt}"

echo "Installing T5X $T5X_REF from $T5X_REPO to $INSTALL_DIR"

set -ex

## Install dependencies

apt-get update
apt-get install -y \
    build-essential \
    cmake \
    clang \
    git

## Install T5X

git clone ${T5X_REPO} ${INSTALL_DIR}/t5x
cd ${INSTALL_DIR}/t5x
git checkout ${T5X_REF}
# The dependency chain for t5x is t5x -> seqio[any] -> tensorflow-text[any] -> tensorflow[any]
# and tensorflow 2.13.0 has a constraint on typing_extensions that makes it
# incompatible with transformer_engine which depends on pydantic and by
# extension, typing_extensions: https://github.com/tensorflow/tensorflow/pull/60688
# As the top level application, we will just exclude it since typing_extensions appears
# to have been relaxed at the head of tensorflow.
pip install -e .[gpu] 'tensorflow-text!=2.13.*'

apt-get autoremove -y
apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf ~/.cache/pip/
