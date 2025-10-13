#!/bin/bash

## Parse command-line arguments

usage() {
cat <<EOF
Clones a git repo at a specific ref and update manifest file with the commit hash.

Usage: $0 [OPTION]... GIT_URL#REF PATH
  -h, --help             Print usage.
  -m, --manifest FILE    The manifest yaml file to which the downloaded library is documented.
                         Default is /opt/manifest.d/git-clone.yaml

Example:
  # clone JAX's main branch at /opt/jax and update manifest file at the default location
  git-clone.sh https://github.com/google/jax.git#main /opt/jax
  # clone JAX at jax-v0.4.26 and update manifest file at an alternative location
  git-clone.sh -m /root/manifest.yaml https://github.com/google/jax.git#jax-v0.4.26 /opt/jax
EOF
    exit $1
}

args=$(getopt -o hm: --long help,manifest: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi

## Set default arguments

MANIFEST="/opt/manifest.d/git-clone.yaml"

eval set -- "$args"
while [ : ]; do
  case "$1" in
    -h | --help)
        usage
        ;;
    -m | --manifest)
        MANIFEST="$2"
        shift 2
        ;;
    --)
        shift;
        break 
        ;;
  esac
done

GIT_URLREF="$1"; shift
DESTINATION="$1"; shift

if [[ $# -ge 1 ]]; then
    echo "Un-recognized argument(s): $*"
    usage 1
fi

if [[ -z "${GIT_URLREF}" ]]; then
    echo "Git url/ref not specified."
    usage 1
fi

if [[ -z "${DESTINATION}" ]]; then
    echo "Destination path not specified."
    usage 1
fi

## check out the source
GIT_REPO=$(cut -d# -f1 <<< $GIT_URLREF)
GIT_REF=$(cut -d# -f2- <<< $GIT_URLREF)

echo "Fetching $GIT_REPO#$GIT_REF to $DESTINATION"

set -ex -o pipefail

git clone ${GIT_REPO} ${DESTINATION}
pushd ${DESTINATION}
git checkout ${GIT_REF}
COMMIT_SHA=$(git rev-parse HEAD)
git submodule update --init --recursive
popd

## update the manifest file

mkdir -p $(dirname ${MANIFEST})
touch ${MANIFEST}
PACKAGE=$(basename "${DESTINATION}")
yq eval --inplace ". += {\"${PACKAGE}\": {\"url\": \"${GIT_REPO}\", \"tracking_ref\": \"${GIT_REF}\", \"latest_verified_commit\": \"${COMMIT_SHA}\", \"mode\": \"git-clone\"}}" ${MANIFEST}
