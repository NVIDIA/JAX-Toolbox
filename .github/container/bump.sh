#!/bin/bash

## Parse command-line arguments

usage() {
cat <<EOF
This script is a utility for updating source references in a manifest YAML file for building 
JAX-Toolbox images. It either updates the 'latest_verified_commit' for each package in the 
manifest based on its current tracking reference, or, if specified, creates local patches that
freeze git-refs (which can point to different SHAs).

Usage: $0 [OPTION]...
  -i, --input-manifest  PATH     The YAML manifest file specifying the world state. Updated in-place unless --output-manifest is provided
  -h, --help                     Print usage.
  -o, --output-manifest PATH     Path to output manifest. Use this if you don't want to update manifest in-place
  -s, --skip-bump-refs           If provided, update patch files and the patchlist in the manifest, but skip bumping refs

Note: patches are always updated in-place

EOF
exit $1
}

args=$(getopt -o i:ho:s --long input-manifest:,help,output-manifest:,skip-bump-refs -- "$@")
if [[ $? -ne 0 ]]; then
  echo
  usage 1
fi

eval set -- "$args"
while [ : ]; do
  case "$1" in
    -s | --skip-bump-refs)
        SKIP_BUMP_REFS=1
        shift 1
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

SKIP_BUMP_REFS=${SKIP_BUMP_REFS:-0}

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
    mode=$(yq e ".${pkg}.mode" $MANIFEST_OUT)
    if [[ $mode == git-clone || $mode == pip-vcs ]] && [[ $SKIP_BUMP_REFS -eq 0 ]]; then
        url=$(yq e ".${pkg}.url" $MANIFEST_OUT)
        tracking_ref=$(yq e ".${pkg}.tracking_ref" $MANIFEST_OUT)
        if ! new_ref=$(git ls-remote --exit-code $url $tracking_ref | awk '{print $1}'); then
	    echo "Could not fetch $tracking_ref from $url"
	    exit 1
	fi
        yq e ".${pkg}.latest_verified_commit = \"$new_ref\"" -i $MANIFEST_OUT
    fi

    has_patches=$(yq e ".${pkg} | has(\"patches\")" $MANIFEST_OUT)
    if [[ $mode == git-clone && $has_patches == "true" ]]; then
        url=$(yq e ".${pkg}.url" $MANIFEST_OUT)
        repo_tmp=$(mktemp -d /tmp/${pkg}.XXXXXX)
        git clone $url $repo_tmp
        # Skip apply to defer to allow building upstream t5x and rosetta t5x
        $SCRIPT_DIR/create-distribution.sh \
          --manifest $MANIFEST_OUT \
          --override_dir $repo_tmp \
          --package ${pkg} \
          --skip-apply
        rm -rf $repo_tmp
    fi
done

# unfortunately the openxla-triton commit must be derived from the content of
# the xla repository
if [[ $SKIP_BUMP_REFS -eq 0 ]]; then
    "${SCRIPT_DIR}/bump-openxla-triton.sh" --manifest "${MANIFEST_OUT}"
fi
