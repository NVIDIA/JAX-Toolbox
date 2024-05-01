#!/bin/bash

# This tests creating a distribution using branches from a nvjax-svc-0 mirror and
# validates that the commits were applied correctly.
#
# This tests patches of the form:
#
#    patches:
#      mirror/<some/branch/name>: file://a.patch

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

set -eou pipefail

# Version should work on Linux || Darwin
tmp_base=$(mktemp -d 2>/dev/null || mktemp -d -t 'upstream')
workspace_tmp=$(mktemp -d 2>/dev/null || mktemp -d -t 'workspace')
manifest_tmp=$(mktemp /tmp/manifest.yaml.XXXXXX)

LIBRARY=t5x
# Commit was taken just before PR-1372
DISTRIBUTION_BASE_REF=22117ce5a3606706ba9519ccdd77b532ad8ff7b2
repo_tmp=$tmp_base/$LIBRARY

cat <<EOF >> $manifest_tmp
t5x:
  url: https://github.com/google-research/t5x.git
  mirror_url: https://github.com/nvjax-svc-0/t5x.git
  tracking_ref: main
  patches:
    mirror/pull/4/head: null
EOF
bash ../../.github/container/git-clone.sh https://github.com/google-research/t5x.git#$DISTRIBUTION_BASE_REF $repo_tmp

cp ../../.github/container/create-distribution.sh $workspace_tmp/
base_cmd() {
  bash $workspace_tmp/create-distribution.sh \
    --manifest $manifest_tmp \
    --override_dir $repo_tmp \
    --package $LIBRARY \
    $@
}
base_cmd --skip-apply
base_cmd

# TESTS
EXPECTED_HEAD_COMMIT_MSG="*TEST - DELETE README"
EXPECTED_PENULTIMATE_COMMIT_MSG="$DISTRIBUTION_BASE_REF*Update calls to clu metrics to pass jnp.ndarrays instead of ints."

HEAD_COMMIT_MSG=$(git -C $repo_tmp show --quiet --pretty=oneline HEAD)
PENULTIMATE_COMMIT_MSG=$(git -C $repo_tmp show --quiet --pretty=oneline HEAD^)
if [[ "$HEAD_COMMIT_MSG" == "$EXPECTED_HEAD_COMMIT_MSG" ]]; then
  echo "Expected head commit msg: $HEAD_COMMIT_MSG"
  echo "Head commit msg:          $EXPECTED_HEAD_COMMIT_MSG"
  echo "TEST FAIL"
  exit 1
elif [[ "$PENULTIMATE_COMMIT_MSG" == "$EXPECTED_PENULTIMATE_COMMIT_MSG" ]]; then
  echo "Expected penultimate commit msg: $PENULTIMATE_COMMIT_MSG"
  echo "Penultimate commit msg:          $EXPECTED_PENULTIMATE_COMMIT_MSG"
  echo "TEST FAIL"
  exit 1
fi

rm -rf $repo_tmp $manifest_tmp $workspace_tmp
echo "TEST SUCCESS"
