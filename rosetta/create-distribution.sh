#!/bin/bash

## Parse command-line arguments

usage() {
cat <<EOF
Usage: $0 [OPTION]...
  -d, --dir=PATH         Path of installed upstream base library. Defaults to /opt/t5x
  -e, --extra-dir=PATH   Path to an additional git mirror of the base library to retrieve git-refs to patch upstream.
                           Should be a fork/mirror of the repo passed to -d/--dir. Helpful if you want to test your
                           patch before PR-ing against upstream
  -h, --help             Print usage.
  -m, --mirror-url=URL   Git url of a mirror/fork of the upstream repo located at --dir
  -p, --patchlist=PATH   Path to patchlist.txt with feature PRs
  -r, --ref=REF          Git commit hash or tag name that specifies the base of the t5x distribution. Defaults to main (not origin/main)

Relationship between --dir, --extra-dir, and --mirror-url repo args:
  --dir: The upstream repo, locally cloned
  --mirror-url: A mirror of the upstream repo
  --extra-dir: A locally cloned mirror of the upstream repo. Helpful to incorporate changes from private repos.

Patches in the --patchlist will be applied from the repos above according to the following rules:

  --dir:
    * ^pull/.*
  --mirror-url:
    * ^mirror/.*
    * ^mirror/pull/.*
  --extra-dir:
    * Anything else

EOF
exit $1
}

args=$(getopt -o d:e:hm:p:r: --long dir:,extra-dir:,help,mirror-url:,patchlist:,ref: -- "$@")
if [[ $? -ne 0 ]]; then
  echo
  usage 1
fi

eval set -- "$args"
while [ : ]; do
  case "$1" in
    -d | --dir)
        INSTALLED_DIR="$2"
        shift 2
        ;;
    -e | --extra-dir)
        EXTRA_DIR="$2"
        shift 2
        ;;
    -h | --help)
        usage
        ;;
    -m | --mirror-url)
        MIRROR_GIT_URL="$2"
        shift 2
        ;;
    -p | --patchlist)
        PATCH_LIST=$(readlink -f "$2")
        shift 2
        ;;
    -r | --ref)
        DISTRIBUTION_BASE_REF="$2"
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

set -euox pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

INSTALLED_DIR=${INSTALLED_DIR:-/opt/t5x}
DISTRIBUTION_BASE_REF=${DISTRIBUTION_BASE_REF:-HEAD}
MIRROR_GIT_URL=${MIRROR_GIT_URL:-https://github.com/nvjax-svc-0/t5x.git}

if [[ -z "${INSTALLED_DIR}" ]]; then
  echo "[ERROR]: Need to specify -d/--dir"
  usage 1
fi

cd ${INSTALLED_DIR}

for git_command in cherry-pick rebase merge revert; do
  RESERVED_INPROGRESS_REF=$(echo $git_command | tr 'a-z' 'A-Z' | sed 's/-/_/g')_HEAD
  if git rev-parse --verfiy $RESERVED_INPROGRESS_REF >/dev/null 2>&1; then
    echo -e "[ERROR]: There is an inprogress $git_command. If you'd like to abort then run:\n"
    echo "  git -C ${INSTALLED_DIR} $git_command --abort"
    exit 1
  fi
done

# Situation:
#    A - B(main,origin/main) - C
#                               \
#                                 __ D (PR || patch)
#
# Since PR may be rooted on a future commit on main (C), if we run "git merge-base origin/main D", we will get B
# instead of C, which would cause us to cherry-pick future upstream commits. So, we will fetch the latest
# upstream main to prevent this.
git fetch origin main

echo "[INFO]: Basing distribution on git-ref: ${DISTRIBUTION_BASE_REF} ($(git rev-parse ${DISTRIBUTION_BASE_REF}))"
# previous-HEAD's purpose is to point to the state of the repo before any changes are made whereas
# distribution-base is to point to the commit where we want to begin building the distribution on.
# Most of the time it will be the same, but it can be different.
if ! git rev-parse --verify previous-HEAD >/dev/null 2>&1; then
  git branch --force previous-HEAD HEAD
else
  git switch previous-HEAD
fi
# Create a local branch to mark the base commit
git branch --force distribution-base ${DISTRIBUTION_BASE_REF}
# Create a local branch for the distribution that starts from the base
git branch --force rosetta-distribution distribution-base
git switch rosetta-distribution

####################################
# Branch patches from a local repo #
####################################
# Create local branches for all remote branches since remotes aren't recognized
# when adding local repos as remotes. Excludes origin/HEAD and origin/test_\d+
TMP_BRANCH_SUFFIX='-tmp-rosetta'
if [[ -n "${EXTRA_DIR+x}" ]] && [[ -d ${EXTRA_DIR} ]]; then
  EXTRA_REMOTE_NAME=extra
  git+extra() {
    git -C ${EXTRA_DIR} $@
  }
  # Removing 'origin/test_[0-9]+' as this is a common remote for forks and will pull in many branches we never use
  for remote_branch in $(git+extra branch --remotes | grep -v 'origin/HEAD' | egrep -v 'origin/test_[0-9]+' | cut -c3-); do
    # Try creating a local tracking branch, but if already exists, then update it to match remote.
    # appending -tmp-rosetta in case there's already a local tracking branch with that name.
    git+extra branch --track --force $(sed 's/origin\///' <<<${remote_branch})${TMP_BRANCH_SUFFIX} ${remote_branch}
  done
  # Now create a tracking branch for all local branches that don't already have a temporary branch
  for local_branch in $(git+extra branch | egrep -v -- ${TMP_BRANCH_SUFFIX}'$' | cut -c3- | egrep -v '^\('); do
    if git rev-parse --verify ${local_branch}${TMP_BRANCH_SUFFIX} >/dev/null 2>&1; then
      # Skipping refs already created from previous loop
      continue
    fi
    # To deal with the anomolous situation where a local_branch matches another git-ref like a tag
    # We will use refs/heads/$local_branch instead of $local_branch
    git+extra branch --force ${local_branch}${TMP_BRANCH_SUFFIX} refs/heads/$local_branch
  done
  
  if git remote show ${EXTRA_REMOTE_NAME} &>/dev/null; then
    git remote remove ${EXTRA_REMOTE_NAME}
  fi
  git remote add -f ${EXTRA_REMOTE_NAME} ${EXTRA_DIR}
fi

#################
# Apply patches #
#################
fork-point() {
  main=$1
  feat_branch=$2
  # 1. Try to find the common base commit assuming the refs are un-merged
  base=$(git merge-base ${main} ${feat_branch})
  if [[ $base != $(git rev-parse ${feat_branch}) ]]; then
    echo $base
    return
  fi
  # 2. The refs are merged somehow b/c feat_branch is an ancestor of main; need to find the merge commit
  echo "[WARNING] One of these two refs ('$main' or '$feat_branch') were merged into the other. Trying to come up with fork-point assuming possible merge-commit." >&2
  merge_commit=$(git rev-list --ancestry-path ${feat_branch}..${main}  | tail -n1)
  git merge-base ${merge_commit}^ ${feat_branch}^
}
apply-patches() {
  from=$1
  to=$2
  # Normally we'd apply the changes with git cherry-pick, but we need to check if there are merge commits
  num_merge_commits=$(git rev-list --min-parents=2 --count $from..$to)
  if [[ $num_merge_commits -gt 0 ]]; then
    echo "[WARNING] There are merge commits between ${from}..${to}. Linearizing history before cherry-picking to remove merge-commits" >&2
    # Make a tmp branch for the linear history
    git checkout -b tmp-linear-tmp $to
    # This will create a linear history
    git rebase $from
    # switch back to the rosetta-distribution branch
    git checkout -
    to=tmp-linear-tmp
  fi
  git cherry-pick ${from}..${to}
  ret_code=$?
  if [[ $to == tmp-linear-tmp ]]; then
    git branch -D tmp-linear-tmp
  fi
  return $ret_code
}
MIRROR_REMOTE_NAME=mirror
if git remote show ${MIRROR_REMOTE_NAME} &>/dev/null; then
  git remote remove ${MIRROR_REMOTE_NAME}
fi
git remote add -f ${MIRROR_REMOTE_NAME} ${MIRROR_GIT_URL}
IFS=$'\n'
for line in $(cat ${PATCH_LIST}); do
  if [[ "${line}" =~ ^[[:blank:]]*$ ]] || [[ "${line}" =~ ^[[:blank:]]*\# ]]; then
    continue
  fi
  git_ref=$(awk '{print $1}' <<< "${line}")
  if [[ "${git_ref}" =~ ^pull/ ]]; then
    REMOTE_NAME=origin
    PR_ID=$(cut -d/ -f2 <<<"${git_ref}")
    branch=PR-${PR_ID}
    git fetch ${REMOTE_NAME} ${git_ref}:${branch}
    main_branch=${REMOTE_NAME}/main
  elif [[ "${git_ref}" =~ ^mirror/pull/ ]]; then
    REMOTE_NAME=${MIRROR_REMOTE_NAME}
    PR_ID=$(cut -d/ -f3 <<<"${git_ref}")
    branch=PR-${PR_ID}
    git fetch ${REMOTE_NAME} $(cut -d/ -f2- <<<${git_ref}):${branch}
    main_branch=${REMOTE_NAME}/main
  elif [[ "${git_ref}" =~ ^mirror/ ]]; then
    REMOTE_NAME=${MIRROR_REMOTE_NAME}
    # REMOTE_NAME not needed b/c git_ref already prefixed
    branch=${git_ref}
    main_branch=${REMOTE_NAME}/main
  else
    if [[ -z "${EXTRA_DIR+x}" ]] || [[ ! -d ${EXTRA_DIR} ]]; then
      echo "[WARNING]: EXTRA_DIR=${EXTRA_DIR} does not exist so cannot cherry-pick ${git_ref}"
      continue
    fi
    REMOTE_NAME=${EXTRA_REMOTE_NAME}
    # Fetch both the feature branch and main so that we can cherry pick the entire branch
    branch=${REMOTE_NAME}/${git_ref}${TMP_BRANCH_SUFFIX}
    # Use main-tmp-rosetta instead of main b/c remote branches may have been updated and the local main is stale
    main_branch=${REMOTE_NAME}/main${TMP_BRANCH_SUFFIX}
  fi
  fork_point=$(fork-point ${main_branch} ${branch})
  ret_code=0
  apply-patches ${fork_point} ${branch} || ret_code=$?
  if [[ ${ret_code} -ne 0 ]]; then
    cat <<EOF
[ERROR]: Tried patching commits from ${fork_point}..${branch} errored. Some possibilities include:

    1. Merge conflict encountered; need to resolve.
    2. It's possible the branch=${branch} wasn't fetched, doesn't exist, or was deleted.

Note: ${fork_point}=\$(git merge-base ${main_branch} ${branch})

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
for remote in ${MIRROR_REMOTE_NAME} ${EXTRA_REMOTE_NAME:-}; do
  if git remote show ${remote} &>/dev/null; then
    git remote remove ${remote}
  fi
done
if [[ -n "${EXTRA_REMOTE_NAME+x}" ]]; then
  git+extra branch --list "*${TMP_BRANCH_SUFFIX}" | xargs -I@ git -C ${EXTRA_DIR} branch -d @
fi