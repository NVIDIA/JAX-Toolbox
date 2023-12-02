#!/bin/bash

## Parse command-line arguments

usage() {
cat <<EOF
Usage: $0 [OPTION]...
  -b, --only-bump-patches        [Optional] If provided, update patch files and the patchlist in the manifest, but skip bumping refs
  -i, --input-manifest  PATH     If set, will clean the patch dir. Default is not to clean
  -h, --help                     [Optional] Print usage.
  -o, --output-manifest PATH     [Optional] Use this if you don't want to update manifest in-place

EOF
exit $1
}

args=$(getopt -o bi:ho: --long only-bump-patches,input-manifest:,help,output-manifest: -- "$@")
if [[ $? -ne 0 ]]; then
  echo
  usage 1
fi

eval set -- "$args"
while [ : ]; do
  case "$1" in
    -b | --only-bump-patches)
        ONLY_BUMP_PATCHES=1
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

ONLY_BUMP_PATCHES=${ONLY_BUMP_PATCHES:-0}

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
    if [[ $mode == git-clone || $mode == pip-vcs ]] && [[ $ONLY_BUMP_PATCHES -eq 0 ]]; then
        url=$(yq e ".${pkg}.url" $MANIFEST_OUT)
        tracking_ref=$(yq e ".${pkg}.tracking_ref" $MANIFEST_OUT)
        new_ref=$(git ls-remote $url $tracking_ref | awk '{print $1}')
        yq e ".${pkg}.ref = \"$new_ref\"" -i $MANIFEST_OUT
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
