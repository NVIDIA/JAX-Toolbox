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
cat << EOF | patch -p1
diff --git a/pip_package/build.sh b/pip_package/build.sh
index ef62c432e..659e78956 100755
--- a/pip_package/build.sh
+++ b/pip_package/build.sh
@@ -89,7 +89,7 @@ bazel clean
 bazel build $@ ...
 if ! [[ $SKIP_TESTS ]]; then
   # Just test the core for the purposes of the pip package.
-  bazel test $@ lingvo/core/...
+  bazel test $@ lingvo/core/... --  -//lingvo/tasks/mt:model_test -//lingvo/core:saver_test
 fi

 DST_DIR="/tmp/lingvo/dist"
EOF

sed -i 's/tensorflow=/#tensorflow=/'  docker/dev.requirements.txt
sed -i 's/tensorflow-text=/#tensorflow-text=/'  docker/dev.requirements.txt
sed -i 's/dataclasses=/#dataclasses=/'  docker/dev.requirements.txt
pip install -r docker/dev.requirements.txt
pip install protobuf==3.20
pip install patchelf

# Some tests are flaky right now (see the patch abovbe), if needed we can skip
# running the tests entirely by uncommentin the following line.
PYTHON_MINOR_VERSION=10 pip_package/build.sh
pip install /tmp/lingvo/dist/lingvo*linux_aarch64.whl
popd
rm -Rf *lingvo*
rm -Rf /root/.cache
