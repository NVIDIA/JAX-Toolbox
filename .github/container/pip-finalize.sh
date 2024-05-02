#!/bin/bash

set -eoux pipefail

pushd /opt/pip-tools.d

# First pip-compile gathers all reqs, but we are care only about VCS installs
# It's possible there are 2nd degree transitive dependencies that are VCS, so
# this is more robust to gather VCS requirements at the cost of pip-compiling
# twice
pip-compile -o requirements.pre $(ls requirements-*.in)

# Find the VCS installs, which are of the form
# PACKAGE @ git+GIT_REPO_URL
for line in $(sed -n -e 's/^\([^#].*\) @ git+\(.*\)$/\1=\2/p' requirements.pre); do
  PACKAGE="${line%=*}"
  REPO_URL="${line#*=}"
  ref=$(yq e ".${PACKAGE}.commit" ${MANIFEST_FILE})
  if [[ "${ref}" == "null" ]]; then
    # If a commit wasn't pinned in the manifest, get the latest version of the
    # default branch of $REPO_URL, pin it, and write it to the manifest.
    ref=$(git ls-remote --exit-code "${REPO_URL}" HEAD | awk '{ print $1 }')
    touch /opt/manifest.d/pip-finalize.yaml
    yq -i e ".${PACKAGE}.commit = \"${ref}\"" /opt/manifest.d/pip-finalize.yaml
    yq -i e ".${PACKAGE}.mode = \"pip-vcs\"" /opt/manifest.d/pip-finalize.yaml
    yq -i e ".${PACKAGE}.url = \"${REPO_URL}\"" /opt/manifest.d/pip-finalize.yaml
  fi
  echo "${PACKAGE} @ git+${REPO_URL}@${ref}"
done | tee requirements.vcs

# Second pip-compile includes one more requirements file that pins all vcs installs
# Uses a special env var to let our custom pip impl know to treat the following as
# equivalent:
#
# fiddle @ git+https://github.com/google/fiddle
# fiddle @ git+https://github.com/google/fiddle@cd4497e4c09bdf95dcccaa1e138c2c125d32d39f
#
# JAX_TOOLBOX_VCS_EQUIVALENCY is an environment variable enabling custom logic in pip
# that treats the above as equivalent and prefers the URI wit the SHA
JAX_TOOLBOX_VCS_EQUIVALENCY=true pip-compile -o requirements.txt requirements.vcs $(ls requirements-*.in)

# If there are unpinned VCS dependencies, error since these should be included in the manifest
unpinned_vcs_dependencies=$(cat requirements.txt | egrep '^[^#].+ @ git\+' | egrep -v '^[^#].+ @ git\+.+@' || true)
if [[ $(echo -n "$unpinned_vcs_dependencies" | wc -l) -gt 0 ]]; then
  echo "Unpinned VCS installs found in $(readlink -f requirements.txt):"
  echo "$unpinned_vcs_dependencies"
  exit 1
fi

# --no-deps is required since conflicts can still appear during pip-sync
pip-sync --pip-args '--no-deps --src /opt' requirements.txt

rm -rf ~/.cache/*
