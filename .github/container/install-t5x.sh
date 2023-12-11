#/bin/bash -ex
#DOCKER_BUILDKIT=1 docker build --network=host -f Dockerfile.t5x -t t5x --build-arg BASE_IMAGE=jax --build-arg T5X_REF=$REF_T5X --build-arg REF_TE=$REF_TE .

#/opt/jax/build/bazel-6.1.2-linux-arm64
pip install chex==0.1.7
cd /opt
git clone https://github.com/google-research/t5x.git
cd t5x
export T5X_INSTALLED_DIR=/opt/t5x

sed -i 's/tensorflow/#tensorflow/' ${T5X_INSTALLED_DIR}/setup.py
sed -i 's/t5=/#t5=/' ${T5X_INSTALLED_DIR}/setup.py
sed -i 's/^jax/#jax/' ${T5X_INSTALLED_DIR}/setup.py

sed -i "s/f'jax/#f'jax/" ${T5X_INSTALLED_DIR}/setup.py
sed -i "s/'tpu/#'tpu/" ${T5X_INSTALLED_DIR}/setup.py

sed -i 's/protobuf/#protobuf/' ${T5X_INSTALLED_DIR}/setup.py
sed -i 's/numpy/#numpy/' ${T5X_INSTALLED_DIR}/setup.py
cat ${T5X_INSTALLED_DIR}/setup.py

pip install tensorflow_datasets==4.9.2 # force a recent version to have latest protobuf dep
pip install auditwheel tensorflow==2.13.0

cd /opt/jax/build/
ln -s bazel-6.1.2-linux-arm64 bazel
export PATH=$PATH:/opt/jax/build/

# Need as specific bazel version
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-arm64 -O /usr/bin/bazel ;
chmod a+x /usr/bin/bazel

cd /opt
git clone http://github.com/tensorflow/text.git
pushd text
git checkout v2.13.0
./oss_scripts/run_build.sh
find * | grep '.whl$'
pip install ./tensorflow_text-*.whl
popd


# Install T5 now, Pip will build the wheel from source, it needs Rust.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > /tmp/rustup.sh && \
    echo "be3535b3033ff5e0ecc4d589a35d3656f681332f860c5fd6684859970165ddcc /tmp/rustup.sh" | sha256sum --check && \
    bash /tmp/rustup.sh -y && \
    export PATH=$PATH:/root/.cargo/bin && \
    pip install t5 && \
    rm -Rf /root/.cargo /root/.rustup && \
    mv /root/.profile /root/.profile.save && \
    grep -v cargo /root/.profile.save > /root/.profile && \
    rm /root/.profile.save && \
    mv /root/.bashrc /root/.bashrc.save && \
    grep -v cargo /root/.bashrc.save > /root/.bashrc && \
    rm /root/.bashrc.save && \
        rm -Rf /root/.cache /tmp/*
