#!/bin/bash

set -eoux pipefail

pushd /opt/pip-tools.d

# First pip-compile gathers all reqs, but we are care only about VCS installs
# It's possible there are 2nd degree transitive dependencies that are VCS, so
# this is more robust to gather VCS requirements at the cost of pip-compiling
# twice
pip-compile -o requirements.pre $(ls requirements-*.in)

IFS=$'\n'
for line in $(cat requirements.pre | egrep '^[^#].+ @ git\+' || true); do
  # VCS installs are of the form "PACKAGE @ git+..."
  PACKAGE=$(echo "$line" | awk '{print $1}')
  ref=$(yq e ".${PACKAGE}.latest_verified_commit" ${MANIFEST_FILE})
  if [[ "$line" == *"#subdirectory="* ]]; then
    # This is required b/c git-refs/commits cannot come after
    # the subdirectory fragment.
    # An example of an install that is of this form is:
    # 'orbax-checkpoint @ git+https://github.com/google/orbax/#subdirectory=checkpoint'
    echo "${line}" | sed "s/#subdirectory=/@${ref}#subdirectory=/"
  else
    echo "${line}@${ref}"
  fi
done | tee requirements.vcs
unset IFS

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

# Execute post-install hooks
for post_install in $(ls /opt/pip-tools-post-install.d/*); do
  if [[ -x "${post_install}" ]]; then
    "${post_install}"
  fi
done
