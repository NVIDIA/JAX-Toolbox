#!/bin/bash

## Parse command-line arguments

usage() {
cat <<EOF
This script is a utility for updating patches when building JAX-Toolbox images.

Usage: $0 [OPTION]...
  -b, --base-patch-dir  PATH     Where generated patch files are written. Default is $SCRIPT_DIR/patches
  -i, --input-manifest  PATH     The YAML manifest file specifying the world state. Updated in-place unless --output-manifest is provided
  -h, --help                     Print usage.
  -o, --output-manifest PATH     Path to output manifest. Use this if you don't want to update manifest in-place

Note: patches are always updated in-place

EOF
exit $1
}

args=$(getopt -o b:i:ho:s --long base-patch-dir:,input-manifest:,help,output-manifest:,skip-bump-refs -- "$@")
if [[ $? -ne 0 ]]; then
  echo
  usage 1
fi

eval set -- "$args"
while [ : ]; do
  case "$1" in
    -b | --base-patch-dir)
        BASE_PATCH_DIR=$(readlink -f "$2")
        shift 2
        ;;
    -i | --input-manifest)
        MANIFEST_IN=$(readlink -f "$2")
        shift 2
        ;;
    -h | --help)
        usage
        ;;
    -o | --output-manifest)
        MANIFEST_OUT=$(readlink -f "$2")
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
    echo
    usage 1
fi

set -eou pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

BASE_PATCH_DIR=${BASE_PATCH_DIR:-$SCRIPT_DIR/patches}

if [[ -z "${MANIFEST_IN:-}" ]]; then
  echo "Need to provide a value for -i/--input-manifest"
  usage 1
fi

if [[ -z "${MANIFEST_OUT:-}" ]]; then
  # Perform the update in place
  MANIFEST_OUT=$MANIFEST_IN
else
  # Write to a new file
  cp $MANIFEST_IN $MANIFEST_OUT
fi

for pkg in $(yq e 'keys | .[]' $MANIFEST_OUT); do
    has_patches=$(yq e ".${pkg} | has(\"patches\")" $MANIFEST_OUT)
    if [[ $has_patches == "true" ]]; then
        url=$(yq e ".${pkg}.url" $MANIFEST_OUT)
        repo_tmp=$(mktemp -d /tmp/${pkg}.XXXXXX)
        git clone $url $repo_tmp
        # Skip apply to defer to allow building upstream t5x and rosetta t5x
        $SCRIPT_DIR/create-distribution.sh \
          --base-patch-dir $BASE_PATCH_DIR \
          --manifest $MANIFEST_OUT \
          --override_dir $repo_tmp \
          --package ${pkg} \
          --skip-apply
        rm -rf $repo_tmp
    fi
done
