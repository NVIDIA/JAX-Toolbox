#!/bin/bash

## Parse command-line arguments

usage() {
cat <<EOF
This script is a utility for updating the commit reference for openxla-triton
in a manifest YAML file used to build JAX-Toolbox images. The commit is derived
from the commit for xla contained in the manifest, along with the patches.

Usage: $0 [OPTION]...
  -h, --help            Print usage.
  --base-patch-dir PATH Where generated patch files are written.
  --manifest PATH       The YAML manifest file specifying the world state. Updated in-place.
EOF
exit $1
}

args=$(getopt -o h --long base-patch-dir:,help,manifest: -- "$@")
if [[ $? -ne 0 ]]; then
  echo
  usage 1
fi

eval set -- "$args"
while [ : ]; do
  case "$1" in
    --base-patch-dir)
        BASE_PATCH_DIR=$(readlink -f "$2")
        shift 2
        ;;
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

if [[ -z "${BASE_PATCH_DIR:-}" ]]; then
  echo "Need to provide a value for --base-patch-dir"
  usage 1
fi

if [[ -z "${MANIFEST:-}" ]]; then
  echo "Need to provide a value for --manifest"
  usage 1
fi

set -eou pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

xla_url=$(yq e ".xla.url" $MANIFEST)
xla_tracking_ref=$(yq e ".xla.tracking_ref" $MANIFEST)
xla_repo=$(mktemp -d /tmp/xla.XXXXXX)
xla_commit=$(yq e ".xla.latest_verified_commit" $MANIFEST)
git clone --branch "${xla_tracking_ref}" --single-branch "${xla_url}" "${xla_repo}"
(cd "${xla_repo}" && git checkout "${xla_commit}")
# Extract the openxla/triton tag used by XLA. Even though it is called
# TRITON_COMMIT it is a tag.
workspace_file="${xla_repo}/third_party/triton/workspace.bzl"
openxla_triton_tag=$(sed -n -e 's#\s\+TRITON_COMMIT = "\(cl[0-9]\+\)"#\1#p' "${workspace_file}")
# Extract Triton patch files applied by XLA
patch_files=$(python3 -c 'import ast, sys; tree = ast.parse(sys.stdin.read()); print(" ".join(elem.value.removeprefix("//third_party/triton:").removesuffix(".patch") for node in ast.walk(tree) if isinstance(node, ast.keyword) and node.arg == "patch_file" for elem in node.value.elts))' < "${workspace_file}")
i=0
for patch_file in ${patch_files}; do
  cp -v "${xla_repo}/third_party/triton/${patch_file}.patch" "${BASE_PATCH_DIR}/openxla-triton/${i}_${patch_file}.patch"
  i=$((i+1))
done
rm -rf "${xla_repo}"
yq e ".openxla-triton.latest_verified_commit = \"${openxla_triton_tag}\"" -i $MANIFEST
