#!/bin/bash

## Parse command-line arguments

usage() {
cat <<EOF
Clones a git repo from the manifest and write the pip-compile input to stdout

Usage: $0 [OPTION]...
  -b, --base-dir DIR     Directory to install package under. Default /opt
  -h, --help             Print usage.
  -l, --library LIB      The library to clone, e.g., jax, flax, t5x
  -m, --manifest FILE    The JAX-Toolbox manifest yaml file
  -o, --out-requirements Create a pip manifest file if specified

Example:
  get-source.sh -m manifest.yaml -l flax
Output:
  -e /opt/flax

EOF
    exit $1
}

args=$(getopt -o b:hl:m:o: --long base-dir:,help,library:,manifest:,out-requirements: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi

## Set default arguments

BASE_INSTALL_DIR="/opt"
MANIFEST=""
OUT_REQUIREMENTS_FILE=""

eval set -- "$args"
while [ : ]; do
  case "$1" in
    -b | --base-dir)
        BASE_INSTALL_DIR=$(readlink -f "$2")
        shift 2
        ;;
    -h | --help)
        usage
        ;;
    -l | --library)
        LIBRARY="$2"
        shift 2
        ;;
    -m | --manifest)
        MANIFEST="$2"
        shift 2
        ;;
    -o | --out-requirements)
        OUT_REQUIREMENTS_FILE="$2"
        shift 2
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

if [[ -z "${LIBRARY}" ]]; then
    echo "Library not specified."
    usage 1
fi

if [[ -z "${MANIFEST}" ]]; then
    echo "Manifest not specified."
    usage 1
fi

## check out the source
PACKAGE_MODE=$(yq e ".${LIBRARY}.mode" $MANIFEST)
if [[ "${PACKAGE_MODE}" != "git-clone" ]]; then
  echo "--library=${LIBRARY} mode is ${PACKAGE_MODE} which is not meant to be cloned. Update mode to \"git-clone\" if this repo should be cloned"
  exit 1
fi

GIT_REPO=$(yq e ".${LIBRARY}.url" $MANIFEST)
GIT_REF=$(yq e ".${LIBRARY}.latest_verified_commit" $MANIFEST)
INSTALL_DIR=${BASE_INSTALL_DIR}/$LIBRARY

echo "Fetching $GIT_REPO#$GIT_REF to $INSTALL_DIR"

set -ex -o pipefail

git clone ${GIT_REPO} ${INSTALL_DIR}
pushd ${INSTALL_DIR}
git checkout ${GIT_REF}
git submodule update --init --recursive
popd

echo "Writing to ${OUT_REQUIREMENTS_FILE}:"
echo "-e file://${INSTALL_DIR}" | tee -a ${OUT_REQUIREMENTS_FILE}
