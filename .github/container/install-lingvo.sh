#!/bin/bash -exu
set -o pipefail
INSTALL_DIR="${INSTALL_DIR:-/opt}"
LINGVO_REF="${LINGVO_REF:-HEAD}"
LINGVO_REPO="${LINGVO_REPO:-https://github.com/tensorflow/lingvo.git}"

## Download lingvo early to fail fast
LINGVO_INSTALLED_DIR=${INSTALL_DIR}/lingvo

[[ -d ${LINGVO_INSTALLED_DIR} ]] || git clone ${LINGVO_REPO} ${LINGVO_INSTALLED_DIR}

pushd ${LINGVO_INSTALLED_DIR}
# Local patches, two PR waiting to be merged + one custom patch
# git fetch origin pull/326/head:pr326  ## merged upstream
# git fetch origin pull/328/head:pr328  ## merged upstream
git fetch origin pull/329/head:pr329
git config user.name "JAX Toolbox"
git config user.email "jax@nvidia.com"
# git cherry-pick --allow-empty pr326 pr328 pr329  ## pr326 pr328 merged
git cherry-pick --allow-empty pr329
sed -i 's/tensorflow\(.*\)2.13/tensorflow\12.14/' ${LINGVO_INSTALLED_DIR}/WORKSPACE
sed -i 's/from tensorflow.python.training.tracking /from tensorflow.python.trackable /' ${LINGVO_INSTALLED_DIR}/lingvo/compat.py
sed -i 's/tensorflow.python.training.tracking.data_structures/tensorflow.python.trackable.data_structures/' ${LINGVO_INSTALLED_DIR}/lingvo/core/nested_map.py
sed -i 's/tensorflow\(.*\)2.13/tensorflow\12.14/' ${LINGVO_INSTALLED_DIR}/pip_package/setup.py

# Disable 2 flaky tests here
patch -p1 < /opt/lingvo.patch


## Install lingvo
sed -i 's/tensorflow=/#tensorflow=/'  docker/dev.requirements.txt
sed -i 's/tensorflow-text=/#tensorflow-text=/'  docker/dev.requirements.txt
sed -i 's/dataclasses=/#dataclasses=/'  docker/dev.requirements.txt
pip install -r docker/dev.requirements.txt
pip install protobuf==3.20.3
pip install patchelf

# Some tests are flaky right now (see the patch abovbe), if needed we can skip
# running the tests entirely by uncommentin the following line.
# SKIP_TEST=1
PYTHON_MINOR_VERSION=10 pip_package/build.sh
pip install /tmp/lingvo/dist/lingvo*linux_aarch64.whl
popd
rm -Rf *lingvo*
rm -Rf /root/.cache
