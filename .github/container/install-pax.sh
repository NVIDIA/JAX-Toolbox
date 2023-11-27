set -ex

## Install Praxis

PRAXIS_INSTALLED_DIR=${INSTALL_DIR}/praxis
git clone ${PRAXIS_REPO} ${PRAXIS_INSTALLED_DIR}
pushd ${PRAXIS_INSTALLED_DIR}
git checkout ${PRAXIS_REF}
if [[ $(uname -m) == "aarch64" ]]; then
  # These dependencies are broken on ARM64 right now, we handle them separately
  # sed -i 's/^tensorflow/#tensorflow/' praxis/pip_package/requirements.txt  requirements.in
  # sed -i 's/^lingvo/#lingvo/' praxis/pip_package/requirements.txt  requirements.in
  sed -i 's/^scikit-learn/#scikit-learn/' praxis/pip_package/requirements.txt  requirements.in
fi
popd

## Install Paxml

PAXML_INSTALLED_DIR=${INSTALL_DIR}/paxml
git clone ${PAXML_REPO} ${PAXML_INSTALLED_DIR}
pushd ${PAXML_INSTALLED_DIR}
git checkout ${PAXML_REF}
if [[ $(uname -m) == "aarch64" ]]; then
  # These dependencies are broken on ARM64 right now, we handle them separately
  pip install chex==0.1.7
  # sed -i 's/^tensorflow/#tensorflow/'  paxml/pip_package/requirements.txt requirements.in
  # sed -i 's/^lingvo/#lingvo/' paxml/pip_package/requirements.txt requirements.in
  sed -i 's/^scikit-learn/#scikit-learn/' paxml/pip_package/requirements.txt requirements.in
  sed -i 's/^t5/#t5/' paxml/pip_package/requirements.txt requirements.in
  sed -i 's/^jax/#jax/' paxml/pip_package/requirements.txt requirements.in
  sed -i 's/^protobuf/#protobuf/' paxml/pip_package/requirements.txt requirements.in
  sed -i 's/^numpy/#numpy/' paxml/pip_package/requirements.txt requirements.in
fi
popd

# SKIP_HEAD_INSTALLS avoids having to install jax from Github source so that
# we do not overwrite the jax that was already installed. Jax at head is
# required by both praxis and paxml, and fiddle at head is only required by
# praxis
HEAD_PACKAGES="jax fiddle"
# We currently require installing editable (-e) to build a distribution since
# we edit the source in place and do not re-install
SKIP_HEAD_INSTALLS=true maybe_defer_pip_install -e ${PRAXIS_INSTALLED_DIR}
SKIP_HEAD_INSTALLS=true maybe_defer_pip_install -e ${PAXML_INSTALLED_DIR}[gpu] $HEAD_PACKAGES

if [[ $(uname -m) == "aarch64" ]]; then
  # array-record pip package is broken on ARM64 right now, we build it from source here.
  cd ${INSTALL_DIR}
  pip uninstall -y array-record || true
  git clone http://github.com/google/array_record.git
  pushd array_record
  oss/build_whl.sh
  pip install  /tmp/array_record/all_dist/array_record-*.whl --force-reinstall
  popd
  rm -Rf array_record /tmp/array_record
fi

maybe_defer_cleanup apt-get autoremove -y
maybe_defer_cleanup apt-get clean
maybe_defer_cleanup rm -rf /var/lib/apt/lists/*
maybe_defer_cleanup rm -rf ~/.cache/
maybe_defer_cleanup rm -rf /tmp/*
