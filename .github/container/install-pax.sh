#!/bin/bash

## Parse command-line arguments

usage() {
    echo "Usage: $0 [OPTION]..."
    echo '  --defer                When passed, will defer the installation of the main package. Can be installed afterwards with `pip install -r requirements-defer.txt` and any deferred cleanup commands can be run with `bash cleanup.sh`' 
    echo "  -d, --dir=PATH         Path to store Pax source. Defaults to /opt"
    echo "  --from_paxml=URL       URL of the Paxml repo. Defaults to https://github.com/google/paxml.git"
    echo "  --from_praxis=URL      URL of the Praxis repo. Defaults to https://github.com/google/praxis.git"
    echo "  -h, --help             Print usage."
    echo "  --ref_paxml=REF        Git commit hash or tag name that specifies the version of Paxml to install. Defaults to HEAD."
    echo "  --ref_praxis=REF       Git commit hash or tag name that specifies the version of Praxis to install. Defaults to HEAD."
    exit $1
}

args=$(getopt -o d:h --long defer,dir:,from_paxml:,from_praxis:,help,ref_paxml:,ref_praxis: -- "$@")
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

DEFER=${DEFER:-false}
PAXML_REF="${PAXML_REF:-HEAD}"
PAXML_REPO="${PAXML_REPO:-https://github.com/google/paxml.git}"
PRAXIS_REF="${PRAXIS_REF:-HEAD}"
PRAXIS_REPO="${PRAXIS_REPO:-https://github.com/google/praxis.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt}"

echo "Installing Paxml $PAXML_REF from $PAXML_REPO and $PRAXIS_REF from $PRAXIS_REPO to $INSTALL_DIR"

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
    for arg in $@; do
      if [[ $arg == "-e" ]]; then
        echo -n "$arg " >>/opt/requirements-defer.txt
      else
        echo "$arg" >> /opt/requirements-defer.txt
      fi
    done
  else
    pip install $@
  fi
}

set -ex

## Install dependencies

apt-get update
apt-get install -y git

## Install Praxis

PRAXIS_INSTALLED_DIR=${INSTALL_DIR}/praxis
git clone ${PRAXIS_REPO} ${PRAXIS_INSTALLED_DIR}
pushd ${PRAXIS_INSTALLED_DIR}
git checkout ${PRAXIS_REF}
popd

## Install Paxml

PAXML_INSTALLED_DIR=${INSTALL_DIR}/paxml
git clone ${PAXML_REPO} ${PAXML_INSTALLED_DIR}
pushd ${PAXML_INSTALLED_DIR}
git checkout ${PAXML_REF}
popd

# SKIP_HEAD_INSTALLS avoids having to install jax from Github source so that
# we do not overwrite the jax that was already installed. Jax at head is
# required by both praxis and paxml, and fiddle at head is only required by
# praxis
HEAD_PACKAGES="jax fiddle"
SKIP_HEAD_INSTALLS=true maybe_defer_pip_install -e ${PRAXIS_INSTALLED_DIR} -e ${PAXML_INSTALLED_DIR}[gpu] $HEAD_PACKAGES

maybe_defer_cleanup apt-get autoremove -y
maybe_defer_cleanup apt-get clean
maybe_defer_cleanup rm -rf /var/lib/apt/lists/*
maybe_defer_cleanup rm -rf ~/.cache/pip/

