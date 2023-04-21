#!/bin/bash

set -euox pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [[ $# -ne 1 ]]; then
  echo "[ERROR]: Need to pass a patchlist file: ./$0 <patchlist.txt>"
  exit 1
fi

PATCH_LIST=$(readlink -f $1)

# This is the git directory of t5x we will build the distribution in
INSTALLED_T5X_DIR=${INSTALLED_T5X_DIR:-}
# If provided, this is the commit that we'll base the distribution on
DISTRIBUTION_BASE_REF=${DISTRIBUTION_BASE_REF:-}
INTERNAL_T5X_SUBMODULE_DIR=$SCRIPT_DIR/t5x
UPSTREAM_GIT_URL=https://github.com/google-research/t5x.git

if [[ -z "$INSTALLED_T5X_DIR" ]]; then
  echo "[ERROR]: Need to specify the env INSTALLED_T5X_DIR="
  exit 1
fi

cd $INSTALLED_T5X_DIR
git config user.email "jax@nvidia.com"
git config user.name "NVIDIA"

if [[ -z "$DISTRIBUTION_BASE_REF" ]]; then
  # Note: not origin/main
  DISTRIBUTION_BASE_REF=$(git rev-parse main)
fi

echo "[INFO]: Basing distribution on t5x commit: $DISTRIBUTION_BASE_REF"
# Switch to main so other branches can be forced updated
git switch main
# Create a local branch to mark the base commit
git branch --force distribution-base $DISTRIBUTION_BASE_REF
# Create a local branch for the distribution that starts from the base
git branch --force rosetta-distribution distribution-base
git switch rosetta-distribution

##############################
# Update NV Internal patches #
##############################
# Create local branches for all remote branches since remotes aren't recognized
# when adding local repos as remotes. Excludes origin/HEAD and origin/test_\d+
function git-nv-t5x {
  git -C $INTERNAL_T5X_SUBMODULE_DIR "$@"
}
NV_REMOTE_NAME=nv-t5x
if [[ -d $INTERNAL_T5X_SUBMODULE_DIR ]]; then
  for remote_branch in $(git-nv-t5x branch --remotes | grep -v 'origin/HEAD' | egrep -v 'origin/test_[0-9]+'); do
    # Try creating a local tracking branch, but if already exists, then update it to match remote
    git-nv-t5x checkout --track $remote_branch || git-nv-t5x reset --hard $remote_branch
  done
  
  if git remote show $NV_REMOTE_NAME &>/dev/null; then
    git remote remove $NV_REMOTE_NAME
  fi
  git remote add -f $NV_REMOTE_NAME $INTERNAL_T5X_SUBMODULE_DIR
fi

#################
# Apply patches #
#################
UPSTREAM_REMOTE_NAME=upstream
if git remote show $UPSTREAM_REMOTE_NAME &>/dev/null; then
  git remote remove $UPSTREAM_REMOTE_NAME
fi
git remote add -f $UPSTREAM_REMOTE_NAME $UPSTREAM_GIT_URL
IFS=$'\n'
for line in $(cat $PATCH_LIST); do
  if [[ "$line" =~ ^[[:blank:]]*$ ]] || [[ "$line" =~ ^[[:blank:]]*\# ]]; then
    continue
  fi
  git_ref=$(awk '{print $1}' <<< "${line}")
  if [[ "$git_ref" =~ ^pull/ ]]; then
    REMOTE_NAME=$UPSTREAM_REMOTE_NAME
    PR_ID=$(cut -d/ -f2 <<<"$git_ref")
    branch=PR-${PR_ID}
    git fetch $REMOTE_NAME $git_ref:$branch
  elif [[ "$git_ref" =~ ^patch/ ]]; then
    if [[ ! -d $INTERNAL_T5X_SUBMODULE_DIR ]]; then
      echo "[INFO]: INTERNAL_T5X_SUBMODULE_DIR=$INTERNAL_T5X_SUBMODULE_DIR does not exist so skipping $git_ref"
      continue
    fi
    REMOTE_NAME=$NV_REMOTE_NAME
    # Fetch both the feature branch and main so that we can cherry pick the entire branch
    branch=$REMOTE_NAME/$git_ref
  else
    echo "[ERROR]: Unrecognized git_ref=$git_ref in $1. Should start with patch/ or pull/"
    exit 1
  fi
  fork_point=$(git merge-base $REMOTE_NAME/main $branch)
  ret_code=0
  git cherry-pick ${fork_point}..$branch || ret_code=$?
  if [[ $ret_code -ne 0 ]]; then
    cat <<EOF
[ERROR]: Tried patching commits from $fork_point..$branch errored. Some possibilities include:

    1. Merge conflict encountered; need to resolve.
    2. It's possible the branch=$branch wasn't fetched, doesn't exist, or was deleted.

Note: $fork_point=\$(git merge-base $REMOTE_NAME/main $branch)

==== git status ====
$(git status)
==== git diff ====
$(git diff)
==================
EOF
    exit 1
  fi
done

# Cleanup
for remote in $UPSTREAM_REMOTE_NAME $NV_REMOTE_NAME; do
  if git remote show $remote &>/dev/null; then
    git remote remove $remote
  fi
done
