#!/bin/bash

## Parse command-line arguments

usage() {
cat <<EOF
This script is a utility for updating the commit reference for openxla-triton
in a manifest YAML file used to build JAX-Toolbox images. The commit is derived
from the commit for xla contained in the manifest.

Usage: $0 [OPTION]...
  -h, --help      Print usage.
  --manifest PATH The YAML manifest file specifying the world state. Updated in-place.
EOF
exit $1
}

args=$(getopt -o h --long help,manifest: -- "$@")
if [[ $? -ne 0 ]]; then
  echo
  usage 1
fi

eval set -- "$args"
while [ : ]; do
  case "$1" in
    --manifest)
        MANIFEST=$(readlink -f "$2")
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
    echo
    usage 1
fi

if [[ -z "${MANIFEST:-}" ]]; then
  echo "Need to provide a value for --manifest"
  usage 1
fi

set -eou pipefail

xla_url=$(yq e ".xla.url" $MANIFEST)
xla_tracking_ref=$(yq e ".xla.tracking_ref" $MANIFEST)
xla_repo=$(mktemp -d /tmp/xla.XXXXXX)
xla_commit=$(yq e ".xla.latest_verified_commit" $MANIFEST)
git clone --branch "${xla_tracking_ref}" --single-branch "${xla_url}" "${xla_repo}"
(cd "${xla_repo}" && git checkout "${xla_commit}")
# Extract the openxla/triton tag used by XLA. Even though it is called
# TRITON_COMMIT it is a tag. In principle we should also account for the
# patches in this .bzl file, but skip that for now.
openxla_triton_tag=$(sed -n -e 's#\s\+TRITON_COMMIT = "\(cl[0-9]\+\)"#\1#p' "${xla_repo}/third_party/triton/workspace.bzl")
rm -rf "${xla_repo}"
openxla_triton_url=$(yq e ".openxla-triton.url" $MANIFEST)
openxla_triton_repo=$(mktemp -d /tmp/openxla-triton.XXXXXX)
git clone --branch "${openxla_triton_tag}" --single-branch --depth 3 "${openxla_triton_url}" "${openxla_triton_repo}"
# Undo two changes that Google always apply to the openxla/triton@llvm-head
# branch that these tags are based off, because they remove the Python
# bindings that we need.
openxla_triton_commit=$(cd "${openxla_triton_repo}" && git rev-parse --verify HEAD~2)
rm -rf "${openxla_triton_repo}"
yq e ".openxla-triton.latest_verified_commit = \"${openxla_triton_commit}\"" -i $MANIFEST
