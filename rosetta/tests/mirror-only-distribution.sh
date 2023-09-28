#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

set -eou pipefail

# Version should work on Linux || Darwin
#repo_tmp=$(mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir')
repo_tmp=/tmp/t5x
patchlist_tmp=$(mktemp /tmp/patchlist.txt.XXXXXX)

UPSTREAM_URL=https://github.com/google-research/t5x.git
MIRROR_URL=https://github.com/nvjax-svc-0/t5x.git
# Commit was taken just before PR-1372
DISTRIBUTION_BASE_REF=22117ce5a3606706ba9519ccdd77b532ad8ff7b2

git clone $UPSTREAM_URL $repo_tmp
echo "mirror/pull/4/head" >> $patchlist_tmp

bash ../create-distribution.sh \
    -r $DISTRIBUTION_BASE_REF \
    -d $repo_tmp \
    -m $MIRROR_URL \
    -p $patchlist_tmp

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

rm -rf $repo_tmp $patchlist_tmp
echo "TEST SUCCESS"
