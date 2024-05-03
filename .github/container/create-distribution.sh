#!/bin/bash

## Parse command-line arguments

usage() {
cat <<EOF
$0 is a utility script that creates patches from remotes and applies them to an upstream library
that has been locally cloned. The end result is a "distribution" of the library that includes features
or fixes from the patches. This script does not build or install the library, but creates a local branch
that includes all of the patches.

Usage: $0 [OPTION]...
  -b, --base-patch-dir      Where generated patch files are written. Default is $SCRIPT_DIR/patches
  -c, --clean               If set, will clean the patch dir. Default is not to clean
  -h, --help                Print usage.
  -m, --manifest=PATH       Path to the manifest. Updates it in-place
  -o, --override_dir=PATH   Use this if there is a custom location of the upstream clone. If not specified, uses /opt/\${PACKAGE}
  -p, --package=KEY         The package name in the manifest to use, e.g., t5x, paxml
  -s, --skip-apply          If provided, will only create patches, update manifest, and skip applying. When not provided, applies local patches.

--------------

This script has two modes of operation:
  1. $0 --skip-apply ...
  2. $0 ...

Assuming you have:
t5x:
  patches:
   pull/3340/head: file://patches/t5x/pull-3340-head.patch

(1) looks at the tracking-refs (pull/3340/head) of the patch and updates the local patch and the filename in the manifest (file://patches/t5x/pull-3340-head.patch)
(2) looks only at the filename value (file://patches/t5x/pull-3340-head.patch) and applies it

--------------

The manifest can contain three versions of the repo:
  url: The upstream repo, locally cloned
  mirror_url: A miror of the upstream repo
  extra_dir: Absolute path of locally cloned mirror of the upstream repo. Helpful to incorporate changes from private repos

This script will in-place replace the patches in the --manifest with local patches.
Patches will be applied from the repos (if --skip-apply not set) above according to the following rules:

  Local patches (relative to this file):
    * ^file://.*
  url:
    * ^pull/.*
  mirror_url:
    * ^mirror/.*
    * ^mirror/pull/.*
  extra_dir:
    * Anything else

EOF
exit $1
}

args=$(getopt -o b:chm:p:s --long base-patch-dir:,clean,help,manifest:,override_dir:,package:,skip-apply -- "$@")
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
    -c | --clean)
        CLEAN_PATCHES=1
        shift 1
        ;;
    -h | --help)
        usage
        ;;
    -m | --manifest)
        MANIFEST=$(readlink -f "$2")
        shift 2
        ;;
    -o | --override_dir)
        OVERRIDE_INSTALL_DIR="$2"
        shift 2
        ;;
    -p | --package)
        PACKAGE="$2"
        shift 2
        ;;
    -s | --skip-apply)
        SKIP_APPLY=1
        shift 1
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

set -eoux pipefail
# readlink -f $(pwd) is cross-platform way to ensure /tmp gets resolved correctly on macos
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && readlink -f $(pwd) )

if [[ -z "$MANIFEST" || -z "$PACKAGE" ]]; then
  echo "--manifest and --package must be provided"
  usage 1
fi

BASE_PATCH_DIR=${BASE_PATCH_DIR:-$SCRIPT_DIR/patches}
CLEAN_PATCHES=${CLEAN_PATCHES:-0}
UPSTREAM_URL=$(yq e ".${PACKAGE}.url" $MANIFEST)
# The tracking_ref is interpreted as the default "main" branch and all patches are
# assumed to be rooted on a sha on the tracking_ref's history
TRACKING_REF=$(yq e ".${PACKAGE}.tracking_ref" $MANIFEST)
INSTALLED_DIR=${OVERRIDE_INSTALL_DIR:-/opt/${PACKAGE}}
MIRROR_GIT_URL=$(yq e ".${PACKAGE}.mirror_url // \"\"" $MANIFEST)
EXTRA_DIR=$(yq e ".${PACKAGE}.extra_dir // \"\"" $MANIFEST)

SKIP_APPLY=${SKIP_APPLY:-0}
GEN_PATCH_DIR=$BASE_PATCH_DIR/$PACKAGE
# Associative arrays aren't available before bash <4.0, so maintaining separate key/value arrays
PATCH_KEYS=()
PATCH_VALUES=()
if [[ $CLEAN_PATCHES -eq 1 ]]; then
  echo "--clean provided, so deleting $GEN_PATCH_DIR"
  rm -rf $GEN_PATCH_DIR
fi
mkdir -p $GEN_PATCH_DIR

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
git fetch origin $TRACKING_REF

# previous-HEAD's purpose is to point to the state of the repo before any distribution changes are made
# We do not rely on the manifest.yaml's .${library}.latest_verified_commit because local commits may be made on top by the upstream docker builds
if ! git rev-parse --verify previous-HEAD >/dev/null 2>&1; then
  echo "[INFO]: Basing distribution on HEAD ($(git rev-parse HEAD)) and marking that with the local branch: previous-HEAD"
  git branch --force previous-HEAD HEAD
else
  echo "[INFO]: Basing distribution on ref: previous-HEAD ($(git rev-parse previous-HEAD))"
  git switch previous-HEAD
fi
# Create a local branch to mark the base commit
git branch --force distribution-base previous-HEAD
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

####################
# Helper Functions #
####################
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
# Applies git-am and returns the local patch URI
maybe-apply-local-patch() {
  # This is the associated array key used to update the patchlist
  patch_name=$1
  # Canonicalize path to remove extra slashes or dot syntax
  patch_path=$(readlink -f $2)
  if [[ ! $patch_path =~ ^${SCRIPT_DIR} ]]; then
    echo "[ERROR]: patch_path=$patch_path should start with $SCRIPT_DIR"
    exit 1
  fi
  # Create a new generated patchlist (for reproducibility)
  PATCH_KEYS+=($patch_name)
  PATCH_VALUES+=("file://${patch_path#$SCRIPT_DIR/}")
  if [[ "$SKIP_APPLY" -eq 1 ]]; then
    echo "[INFO]: Skipping patch application: $patch_path"
    return
  fi

  # Apply the patch
  git am --3way <$patch_path || ret_code=$?
  if [[ ${ret_code:-0} -ne 0 ]]; then
    cat <<EOF
[ERROR]: Tried patching commits from $patch_path, but failed:
==== git status ====
$(git status)
==== git diff ====
$(git diff)
==================
EOF
    exit 1
  fi
}
create-and-maybe-apply-ref-patches() {
  patch_name=$1
  from=$2
  to=$3
  # Normally we'd apply the changes with git cherry-pick, but we need to check if there are merge commits
  num_merge_commits=$(git rev-list --min-parents=2 --count $from..$to)
  if [[ $num_merge_commits -gt 0 ]]; then
    echo "[WARNING] There are merge commits between ${from}..${to}. Linearizing history before cherry-picking to remove merge-commits" >&2
    # Make a tmp branch for the linear history
    to_linear=${to}.linearized
    git checkout -b ${to_linear} $to
    # This will create a linear history
    git rebase $from
    # switch back to the rosetta-distribution branch
    git checkout -
    to=${to_linear}
  fi
  # Make the patch
  patch_fname=$(tr '/' '-' <<< "$to").patch
  git format-patch --stdout ${from}..${to} >$GEN_PATCH_DIR/$patch_fname
  if [[ -n "${to_linear:-}" ]]; then
    git branch -D ${to_linear}
  fi
  # Apply the patch
  maybe-apply-local-patch $patch_name $GEN_PATCH_DIR/$patch_fname
}
if [[ -n "${MIRROR_GIT_URL}" ]] ; then
  MIRROR_REMOTE_NAME=mirror
  if git remote show ${MIRROR_REMOTE_NAME} &>/dev/null; then
    git remote remove ${MIRROR_REMOTE_NAME}
  fi
  git remote add -f ${MIRROR_REMOTE_NAME} ${MIRROR_GIT_URL}
fi

#################
# Apply patches #
#################
IFS=$'\n'
for git_ref in $(yq e ".${PACKAGE}.patches | keys | .[]" $MANIFEST); do
  echo '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
  echo "@@ Processing git_ref=$git_ref"
  echo '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
  if [[ $SKIP_APPLY -eq 0 ]]; then
    # If we apply, then use the value, not the key
    patch_uri=$(yq e ".${PACKAGE}.patches.\"${git_ref}\"" $MANIFEST)
    patch_path=$SCRIPT_DIR/${patch_uri#file://}
    if [[ ! -f $patch_path ]]; then
      echo "[ERROR]: ${git_ref} refers to $patch_path which does not exist"
      exit 1
    fi
    maybe-apply-local-patch $git_ref $patch_path
    continue
  elif [[ "${git_ref}" =~ ^file:// ]]; then
    # Getting here means the manifest has a patches entry that looks like {file://a.patch: file://b.patch}
    # So the patch does not need to be created and we'll use whatever the value is. In this case file://b.patch
    continue
  elif [[ "${git_ref}" =~ ^pull/ ]]; then
    REMOTE_NAME=origin
    PR_ID=$(cut -d/ -f2 <<<"${git_ref}")
    branch=PR-${PR_ID}
    git fetch ${REMOTE_NAME} ${git_ref}:${branch}
    main_branch=${REMOTE_NAME}/${TRACKING_REF}
  elif [[ "${git_ref}" =~ ^mirror/pull/ ]]; then
    if [[ -z "${MIRROR_GIT_URL}" ]] ; then
      echo "[Error]: MIRROR_GIT_URL not provided so cannot apply patch=${git_ref}"
      exit 1
    fi
    REMOTE_NAME=${MIRROR_REMOTE_NAME}
    PR_ID=$(cut -d/ -f3 <<<"${git_ref}")
    branch=PR-${PR_ID}
    git fetch ${REMOTE_NAME} $(cut -d/ -f2- <<<${git_ref}):${branch}
    main_branch=${REMOTE_NAME}/${TRACKING_REF}
  elif [[ "${git_ref}" =~ ^mirror/ ]]; then
    if [[ -z "${MIRROR_GIT_URL}" ]] ; then
      echo "[Error]: MIRROR_GIT_URL not provided so cannot apply patch=${git_ref}"
      exit 1
    fi
    REMOTE_NAME=${MIRROR_REMOTE_NAME}
    # REMOTE_NAME not needed b/c git_ref already prefixed
    branch=${git_ref}
    main_branch=${REMOTE_NAME}/${TRACKING_REF}
  else
    if [[ -z "${EXTRA_DIR+x}" ]] || [[ ! -d ${EXTRA_DIR} ]]; then
      echo "[Error]: EXTRA_DIR=${EXTRA_DIR} does not exist so cannot apply patch=${git_ref}"
      exit 1
    fi
    REMOTE_NAME=${EXTRA_REMOTE_NAME}
    # Fetch both the feature branch and main so that we can cherry pick the entire branch
    branch=${REMOTE_NAME}/${git_ref}${TMP_BRANCH_SUFFIX}
    # Use main-tmp-rosetta instead of main b/c remote branches may have been updated and the local main is stale
    main_branch=${REMOTE_NAME}/${TRACKING_REF}${TMP_BRANCH_SUFFIX}
  fi
  fork_point=$(fork-point ${main_branch} ${branch})
  create-and-maybe-apply-ref-patches ${git_ref} ${fork_point} ${branch} || ret_code=$?
  if [[ ${ret_code:-0} -ne 0 ]]; then
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

# Update the patches
for i in "${!PATCH_KEYS[@]}"; do
  yq e ".${PACKAGE}.patches.\"${PATCH_KEYS[$i]}\" = \"${PATCH_VALUES[$i]}\"" -i $MANIFEST
done

###########
# Cleanup #
###########
for remote in ${MIRROR_REMOTE_NAME} ${EXTRA_REMOTE_NAME:-}; do
  if git remote show ${remote} &>/dev/null; then
    git remote remove ${remote}
  fi
done
if [[ -n "${EXTRA_REMOTE_NAME+x}" ]]; then
  git+extra branch --list "*${TMP_BRANCH_SUFFIX}" | xargs -I@ git -C ${EXTRA_DIR} branch -d @
fi
