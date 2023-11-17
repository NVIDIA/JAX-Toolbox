#!/bin/bash

set -eou pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

MANIFEST_IN=$1
MANIFEST_OUT=${1}.bump

cp $MANIFEST_IN $MANIFEST_OUT

for pkg in $(yq e 'keys | .[]' $MANIFEST_OUT); do
    mode=$(yq e ".${pkg}.mode" $MANIFEST_OUT)
    if [[ $mode == root ]]; then
        new_ref=$(git rev-parse HEAD)
        yq e ".${pkg}.ref = \"$new_ref\"" -i $MANIFEST_OUT
    elif [[ $mode == git-clone || $mode == pip-vcs ]]; then
        url=$(yq e ".${pkg}.url" $MANIFEST_OUT)
        tracking_ref=$(yq e ".${pkg}.tracking_ref" $MANIFEST_OUT)
        new_ref=$(git ls-remote $url $tracking_ref | awk '{print $1}')
        yq e ".${pkg}.ref = \"$new_ref\"" -i $MANIFEST_OUT
    fi

    has_patches=$(yq e ".${pkg} | has(\"patches\")" $MANIFEST_OUT)
    if [[ $mode == git-clone && $has_patches == "true" ]]; then
        url=$(yq e ".${pkg}.url" $MANIFEST_OUT)
        mirror_url=$(yq e ".${pkg}.mirror_url" $MANIFEST_OUT)
        repo_tmp=$(mktemp -d /tmp/${pkg}.XXXXXX)
        git clone $url $repo_tmp
        # Skip apply since
        $SCRIPT_DIR/create-distribution.sh \
          --manifest $MANIFEST_OUT \
          --override_dir $repo_tmp \
          --package ${pkg} \
          --skip-apply
        rm -rf $repo_tmp
    fi
done