#!/bin/bash

## Parse command-line arguments

usage() {
    echo "Usage: $0 [OPTION]..."
    echo "  -d, --dir=PATH         Path to store Pax source. Defaults to /opt"
    echo "  --from_paxml=URL       URL of the Paxml repo. Defaults to https://github.com/google/paxml.git"
    echo "  --from_praxis=URL      URL of the Praxis repo. Defaults to https://github.com/google/praxis.git"
    echo "  -h, --help             Print usage."
    echo "  --ref_paxml=REF        Git commit hash or tag name that specifies the version of Paxml to install. Defaults to HEAD."
    echo "  --ref_praxis=REF       Git commit hash or tag name that specifies the version of Praxis to install. Defaults to HEAD."
    exit $1
}

args=$(getopt -o d:h --long dir:,from_paxml:,from_praxis:,help,ref_paxml:,ref_praxis: -- "$@")
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
    --from_paxml)
        PAXML_REPO="$2"
        shift 2
        ;;
    --from_praxis)
        PRAXIS_REPO="$2"
        shift 2
        ;;
    -h | --help)
        usage
        ;;
    --ref_paxml)
        PAXML_REF="$2"
        shift 2
        ;;
    --ref_praxis)
        PRAXIS_REF="$2"
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

PAXML_REF="${PAXML_REF:-HEAD}"
PAXML_REPO="${PAXML_REPO:-https://github.com/google/paxml.git}"
PRAXIS_REF="${PRAXIS_REF:-HEAD}"
PRAXIS_REPO="${PRAXIS_REPO:-https://github.com/google/praxis.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt}"

echo "Installing Paxml $PAXML_REF from $PAXML_REPO and $PRAXIS_REF from $PRAXIS_REPO to $INSTALL_DIR"

set -ex

## Install dependencies

apt-get update
apt-get upgrade -y
apt-get install -y git

## Install Praxis

git clone ${PRAXIS_REPO} ${INSTALL_DIR}/praxis
pushd ${INSTALL_DIR}/praxis
git checkout ${PRAXIS_REF}
pip install -e .
popd

## Install Paxml

git clone ${PAXML_REPO} ${INSTALL_DIR}/paxml
pushd ${INSTALL_DIR}/paxml
git checkout ${PAXML_REF}
pip install -e .[gpu]
popd

apt-get autoremove -y
apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf ~/.cache/pip/

