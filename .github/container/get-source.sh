#!/bin/bash
## Fetch a Python package from a git repo and write the pip-tools input manifest to stdout
## Example:
## get-source.sh -f https://github.com/google/flax.git -r main -d /opt/flax
## Output:
## -e /opt/flax

## Parse command-line arguments

usage() {
    echo "Usage: $0 [OPTION]..."
    echo "  -d, --dir PATH         [Required] Local path to check out the source code."
    echo "  -f, --from URL         [Required] URL of the source repo."
    echo "  -h, --help             Print usage."
    echo "  -m, --manifest FILE    Create a pip manifest file if specified"
    echo "  -r, --ref REF          Git commit SHA, branch name, or tag name to checkout. Uses default branch if not specified."
    echo
    exit $1
}

args=$(getopt -o d:f:hm:r: --long dir:,from:,help,manifest:,ref: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi

## Set default arguments

GIT_REPO=""
GIT_REF="${GIT_REF:-HEAD}"
INSTALL_DIR=""
MANIFEST_FILE=""

eval set -- "$args"
while [ : ]; do
  case "$1" in
    -d | --dir)
        INSTALL_DIR="$2"
        shift 2
        ;;
    -f | --from)
        GIT_REPO="$2"
        shift 2
        ;;
    -h | --help)
        usage
        ;;
    -m | --manifest)
        MANIFEST_FILE="$2"
        shift 2
        ;;
    -r | --ref)
        GIT_REF="$2"
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

if [[ ! -n "${GIT_REPO}" ]]; then
    echo "Source repository not speicified." && echo
    usage 1
fi

if [[ ! -n "${INSTALL_DIR}" ]]; then
    echo "Check out destination not specified." && echo
    usage 1
fi

## check out the source

echo "Fetching $GIT_REPO#$GIT_REF to $INSTALL_DIR"

set -ex -o pipefail

git clone ${GIT_REPO} ${INSTALL_DIR}
pushd ${INSTALL_DIR}
git checkout ${GIT_REF}
git submodule init
git submodule update --recursive
popd

echo "Writing to ${MANIFEST_FILE}:"
echo "-e file://${INSTALL_DIR}" | tee -a ${MANIFEST_FILE}
