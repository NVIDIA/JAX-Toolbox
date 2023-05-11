#!/bin/bash

## Parse command-line arguments

usage() {
  echo "Usage: $0 [OPTION]..."
  echo "  -p, --patchlist=PATH   Path to patchlist.txt with feature PRs"
  echo "  -u, --url=URL          Git url of the upstream repo to obtain pull request refs"
  echo "  -d, --dir=PATH         Path of installed base library. Defaults to /opt/t5x"
  echo "  -r, --ref=REF          Git commit hash or tag name that specifies the base of the t5x distribution. Defaults to main (not origin/main)"
  echo "  -e, --extra=PATH       Path to an additional mirror of the base library to retrieve git-refs to patch upstream. Should be a fork/mirror of \
      the repo passed to -d/--dir. Helpful if you want to test your patch before PR-ing against upstream"
  echo "  -h, --help             Print usage."
  exit $1
}

args=$(getopt -o p:u:d:r:e:h --long patchlist:,url:,dir:,ref:,extra:,help -- "$@")
if [[ $? -ne 0 ]]; then
  echo
  usage 1
fi

eval set -- "$args"
while [ : ]; do
  case "$1" in
    -p | --patchlist)
        PATCH_LIST=$(readlink -f "$2")
        shift 2
        ;;
    -u | --url)
        UPSTREAM_GIT_URL="$2"
        shift 2
        ;;
    -d | --dir)
        INSTALLED_DIR="$2"
        shift 2
        ;;
    -r | --ref)
        DISTRIBUTION_BASE_REF="$2"
        shift 2
        ;;
    -e | --extra)
        EXTRA_MIRROR_DIR="$2"
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

set -euox pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

INSTALLED_DIR=${INSTALLED_DIR:-/opt/t5x}
DISTRIBUTION_BASE_REF=${DISTRIBUTION_BASE_REF:-main}
UPSTREAM_GIT_URL=${UPSTREAM_GIT_URL:-https://github.com/google-research/t5x.git}

if [[ -z "$INSTALLED_DIR" ]]; then
  echo "[ERROR]: Need to specify the env INSTALLED_DIR="
  exit 1
fi

cd $INSTALLED_DIR
git config user.email "jax@nvidia.com"
git config user.name "NVIDIA"

echo "[INFO]: Basing distribution on commit: $DISTRIBUTION_BASE_REF"
# Switch to main so other branches can be forced updated
git switch main
# Create a local branch to mark the base commit
git branch --force distribution-base $DISTRIBUTION_BASE_REF
# Create a local branch for the distribution that starts from the base
git branch --force rosetta-distribution distribution-base
git switch rosetta-distribution

####################################
# Branch patches from a local repo #
####################################
# Create local branches for all remote branches since remotes aren't recognized
# when adding local repos as remotes. Excludes origin/HEAD and origin/test_\d+
if [[ -n "${EXTRA_MIRROR_DIR+x}" ]] && [[ -d $EXTRA_MIRROR_DIR ]]; then
  EXTRA_REMOTE_NAME=extra
  function git-extra {
    git -C $EXTRA_MIRROR_DIR "$@"
  }
  for remote_branch in $(git-extra branch --remotes | grep -v 'origin/HEAD' | egrep -v 'origin/test_[0-9]+'); do
    # Try creating a local tracking branch, but if already exists, then update it to match remote.
    # appending -tmp-rosetta in case there's already a local tracking branch with that name.
    git-extra branch --track --force $(sed 's/origin\///' <<<$remote_branch)-tmp-rosetta $remote_branch
  done
  
  if git remote show $EXTRA_REMOTE_NAME &>/dev/null; then
    git remote remove $EXTRA_REMOTE_NAME
  fi
  git remote add -f $EXTRA_REMOTE_NAME $EXTRA_MIRROR_DIR
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
  else
    if [[ -n "${EXTRA_MIRROR_DIR+x}" ]] && [[ -d $EXTRA_MIRROR_DIR ]]; then
      echo "[ERROR]: EXTRA_MIRROR_DIR=$EXTRA_MIRROR_DIR does not exist so cannot cherry-pick $git_ref"
      continue
    fi
    REMOTE_NAME=$EXTRA_REMOTE_NAME
    # Fetch both the feature branch and main so that we can cherry pick the entire branch
    branch=$REMOTE_NAME/${git_ref}-tmp-rosetta
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
for remote in $UPSTREAM_REMOTE_NAME ${EXTRA_REMOTE_NAME:-}; do
  if git remote show $remote &>/dev/null; then
    git remote remove $remote
  fi
done
if [[ -n "${EXTRA_REMOTE_NAME+x}" ]]; then
  git-extra branch -d --list '*-tmp-rosetta'
fi
