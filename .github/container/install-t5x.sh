#!/bin/bash

## Parse command-line arguments

usage() {
    echo "Usage: $0 [OPTION]..."
    echo '  --defer                When passed, will defer the installation of the main package. Can be installed afterwards with `pip install -r requirements-defer.txt` and any deferred cleanup commands can be run with `bash cleanup.sh`' 
    echo "  -d, --dir=PATH         Path to store T5X source. Defaults to /opt"
    echo "  -f, --from=URL         URL of the T5X repo. Defaults to https://github.com/google-research/t5x.git"
    echo "  -h, --help             Print usage."
    echo "  -r, --ref=REF          Git commit hash or tag name that specifies the version of T5X to install. Defaults to HEAD."
    exit $1
}

args=$(getopt -o d:f:hr: --long defer,dir:,from:,help,ref: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi

eval set -- "$args"
while [ : ]; do
  case "$1" in
    --defer)
        DEFER=true
        shift
        ;;
    -d | --dir)
        INSTALL_DIR="$2"
        shift 2
        ;;
    -f | --from)
        T5X_REPO="$2"
        shift 2
        ;;
    -h | --help)
        usage
        ;;
    -r | --ref)
        T5X_REF="$2"
        shift 2
        ;;
    --)
        shift;
        break 
        ;;
  esac
done

if [[ $# -ge 1 ]]; then
    echo "Un-recognized argument: $*" && echo
    usage 1
fi

## Set default arguments if not provided via command-line

DEFER=${DEFER:-false}
T5X_REF="${T5X_REF:-HEAD}"
T5X_REPO="${T5X_REPO:-https://github.com/google-research/t5x.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt}"

echo "Installing T5X $T5X_REF from $T5X_REPO to $INSTALL_DIR"

maybe_defer_cleanup() {
  if [[ "$DEFER" = true ]]; then
    echo "# Cleanup from: $0"
    echo "$*" >> /opt/cleanup.sh
  else
    $@
  fi
}

maybe_defer_pip_install() {
  if [[ "$DEFER" = true ]]; then
    echo "Deferring installation of 'pip install $*'"
    echo "$*" >> /opt/requirements-defer.txt
  else
    pip install $@
  fi
}

set -ex

## Install dependencies

apt-get update
apt-get install -y \
    build-essential \
    cmake \
    clang \
    git

## Install T5X

T5X_INSTALLED_DIR=${INSTALL_DIR}/t5x

git clone ${T5X_REPO} ${T5X_INSTALLED_DIR}
cd ${T5X_INSTALLED_DIR}
git checkout ${T5X_REF}

if [[ $(uname -m) == "aarch64" ]]; then
    # WAR some aarch64 issues by pre-installing some packages
    echo "AARCH64 WARs"
    pip install chex==0.1.7
    sed -i 's/tensorflow/#tensorflow/' ${T5X_INSTALLED_DIR}/setup.py
    sed -i 's/t5=/#t5=/' ${T5X_INSTALLED_DIR}/setup.py
    sed -i 's/^jax/#jax/' ${T5X_INSTALLED_DIR}/setup.py

    sed -i "s/f'jax/#f'jax/" ${T5X_INSTALLED_DIR}/setup.py
    sed -i "s/'tpu/#'tpu/" ${T5X_INSTALLED_DIR}/setup.py
    
    sed -i 's/protobuf/#protobuf/' ${T5X_INSTALLED_DIR}/setup.py
    sed -i 's/numpy/#numpy/' ${T5X_INSTALLED_DIR}/setup.py
    cat ${T5X_INSTALLED_DIR}/setup.py
    # Manuall install troublesome dependency.

    ## Install tensorflow-text
#    pip install tensorflow_datasets==4.9.2 # force a recent version to have latest protobuf dep
#    pip install auditwheel
#    pip install tensorflow==2.13.0
    # git clone http://github.com/tensorflow/text.git
    # pushd text
    # git checkout v2.13.0
    # ./oss_scripts/run_build.sh
    # find * | grep '.whl$'
    # pip install ./tensorflow_text-*.whl
    # popd
    # rm -Rf text


    
    # Install T5 now, Pip will build the wheel from source, it needs Rust.
    # curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > /tmp/rustup.sh && \
    # echo "be3535b3033ff5e0ecc4d589a35d3656f681332f860c5fd6684859970165ddcc /tmp/rustup.sh" | sha256sum --check && \
    # bash /tmp/rustup.sh -y && \
    # export PATH=$PATH:/root/.cargo/bin && \
    # pip install t5 && \
    # rm -Rf /root/.cargo /root/.rustup && \
    # mv /root/.profile /root/.profile.save && \
    # grep -v cargo /root/.profile.save > /root/.profile && \
    # rm /root/.profile.save && \
    # mv /root/.bashrc /root/.bashrc.save && \
    # grep -v cargo /root/.bashrc.save > /root/.bashrc && \
    # rm /root/.bashrc.save && \
    # rm -Rf /root/.cache /tmp/*
fi

# We currently require installing editable (-e) to build a distribution since
# we edit the source in place and do not re-install
maybe_defer_pip_install -e ${T5X_INSTALLED_DIR}[gpu]

maybe_defer_cleanup apt-get autoremove -y
maybe_defer_cleanup apt-get clean
maybe_defer_cleanup rm -rf /var/lib/apt/lists/*
maybe_defer_cleanup rm -rf ~/.cache/pip/
