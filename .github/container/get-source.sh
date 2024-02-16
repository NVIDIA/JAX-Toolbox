#!/bin/bash

## Parse command-line arguments

usage() {
  cat <<EOF
  Clones a git repo and write the pip-compile directive to stdout

  Usage: $0 [OPTION]...
    -c, --checkout-dir            Directory to check out the package.
    -p, --pip-directive-file      Write the pip directive for installing the source package to file
    -u, --urlref                  URL and ref to clone, in the form "URL#REF"
    -h, --help                    Print usage.

  Example:
    get-source.sh -b /opt -r https://github.com/google/jax.git#v0.4.24
  Output:
    -e /opt/jax
EOF

  exit $1
}

args=$(getopt -o c:p:u:h --long checkout-dir:,pip-directive-file:,urlref:,help -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi

## Set default arguments

CHECKOUT_DIR=""
URLREF=""
PIP_DIRECTIVE_FILE=""

eval set -- "$args"
while [ : ]; do
  case "$1" in
    -c | --checkout-dir)
        CHECKOUT_DIR=$(readlink -f "$2")
        shift 2
        ;;
    -p | --pip-directive-file)
        PIP_DIRECTIVE_FILE="$2"
        shift 2
        ;;
    -u | --urlref)
        URLREF="$2"
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
    echo "Un-recognized argument: $*"
    usage 1
fi

if [[ -z "${CHECKOUT_DIR}" ]]; then
    echo "Check out path not specified."
    usage 1
fi

if [[ -z "${URLREF}" ]]; then
    echo "Git URL/ref not specified."
    usage 1
fi

GIT_REPO=$(echo $URLREF | cut -d# -f1)
GIT_REF=$(echo $URLREF | cut -d# -f2)

echo "Fetching $GIT_REPO at $GIT_REF to $CHECKOUT_DIR"

set -exu -o pipefail

git clone ${GIT_REPO} ${CHECKOUT_DIR}
pushd ${CHECKOUT_DIR}
git checkout ${GIT_REF}
git submodule update --init --recursive
popd

if [[ -n "${PIP_DIRECTIVE_FILE}" ]]; then
  echo "Appending to ${PIP_DIRECTIVE_FILE}:"
  echo "-e file://${CHECKOUT_DIR}" | tee -a ${PIP_DIRECTIVE_FILE}
fi
