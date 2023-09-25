#!/bin/bash -exu

INSTALL_DIR="${INSTALL_DIR:-/opt}"
LINGVO_REF="${LINGVO_REF:-HEAD}"
LINGVO_REPO="${LINGVO_REPO:-https://github.com/tensorflow/lingvo.git}"

## Install tensorflow-text
cd ${INSTALL_DIR}
pip install tensorflow_datasets==4.9.2 # force a recent version to have latest protobuf dep
pip install auditwheel
pip install tensorflow==2.13.0
git clone http://github.com/tensorflow/text.git
pushd text
git checkout v2.13.0
./oss_scripts/run_build.sh
find * | grep '.whl$'
pip install ./tensorflow_text-*.whl
popd
rm -Rf text

## Install lingvo
LINGVO_INSTALLED_DIR=${INSTALL_DIR}/lingvo

[[ -d lingvo ]] || git clone ${LINGVO_REPO} ${LINGVO_INSTALLED_DIR}

pushd ${LINGVO_INSTALLED_DIR}
# Local patches, two PR waiting to be merged + one custom patch
git fetch origin pull/326/head:pr326
git fetch origin pull/328/head:pr328
git fetch origin pull/329/head:pr329
git config user.name "JAX Toolbox"
git config user.email "jax@toolbox"
git cherry-pick pr326 pr328 pr329

# Disable 2 flaky tests here
patch -p1 < /opt/lingvo.patch

sed -i 's/tensorflow=/#tensorflow=/'  docker/dev.requirements.txt
sed -i 's/tensorflow-text=/#tensorflow-text=/'  docker/dev.requirements.txt
sed -i 's/dataclasses=/#dataclasses=/'  docker/dev.requirements.txt
pip install -r docker/dev.requirements.txt
pip install protobuf==3.20
pip install patchelf

# Some tests are flaky right now (see the patch abovbe), if needed we can skip
# running the tests entirely by uncommentin the following line.
# SKIP_TEST=1
PYTHON_MINOR_VERSION=10 pip_package/build.sh
pip install /tmp/lingvo/dist/lingvo*linux_aarch64.whl
popd
rm -Rf *lingvo*
rm -Rf /root/.cache
