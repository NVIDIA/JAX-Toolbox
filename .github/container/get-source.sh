#!/bin/bash
## Clone a git repo and write the pip-compile input to stdout
## Example:
## get-source.sh -m manifest.yaml -l flax
## Output:
## -e /opt/flax

## Parse command-line arguments

usage() {
    echo "Usage: $0 [OPTION]..."
    echo "  -h, --help             Print usage."
    echo "  -l, --library LIB      The library to clone, e.g., jax, flax, t5x"
    echo "  -m, --manifest FILE    The JAX-Toolbox manifest yaml file"
    echo "  -o, --out-requirements Create a pip manifest file if specified"
    echo
    exit $1
}

args=$(getopt -o hl:m:o: --long help,library:,manifest:,out-requirements: -- "$@")
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
GIT_REF=$(yq e ".${LIBRARY}.ref" $MANIFEST)
INSTALL_DIR=${BASE_INSTALL_DIR}/$LIBRARY

echo "Fetching $GIT_REPO#$GIT_REF to $INSTALL_DIR"

set -ex -o pipefail

git clone ${GIT_REPO} ${INSTALL_DIR}
pushd ${INSTALL_DIR}
git checkout ${GIT_REF}
git submodule init
git submodule update --recursive
popd

echo "Writing to ${OUT_REQUIREMENTS_FILE}:"
echo "-e file://${INSTALL_DIR}" | tee -a ${OUT_REQUIREMENTS_FILE}
