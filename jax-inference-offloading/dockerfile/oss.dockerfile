# syntax=docker/dockerfile:1-labs
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:26.05-cuda13.2-devel-ubuntu24.04
ARG URL_JIO=https://github.com/NVIDIA/JAX-Toolbox.git
ARG REF_JIO=main
ARG BASE_PATH_JIO=/opt/jtbx
ARG SUB_PATH_JIO=jax-inference-offloading

###############################################################################
## Install system dependencies and pip-tools
###############################################################################

FROM ${BASE_IMAGE} AS base

ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN <<"EOF" bash -ex
export DEBIAN_FRONTEND=noninteractive
export TZ=UTC
apt_packages=(curl git python-is-python3 python3-pip rsync vim wget jq zip)
apt-get update
apt-get install -y ${apt_packages[@]}
apt-get clean
apt-get autoremove -y
rm -rf /var/lib/apt/lists/*

pip install pip-tools
rm -rf ~/.cache/pip
EOF

###############################################################################
## Download source and configure pip-tools
###############################################################################

FROM base AS mealkit
ARG URL_JIO
ARG REF_JIO
ARG BASE_PATH_JIO
ARG SUB_PATH_JIO

ENV SRC_PATH_JIO=${BASE_PATH_JIO}/${SUB_PATH_JIO}

# Check out source code
RUN --mount=type=ssh <<"EOF" bash -ex -o pipefail
mkdir -p -m 0700 ~/.ssh
ssh-keyscan github.com >> ~/.ssh/known_hosts
git clone --no-checkout ${URL_JIO} ${BASE_PATH_JIO}
pushd ${BASE_PATH_JIO}
git sparse-checkout init --cone
git sparse-checkout set ${SUB_PATH_JIO}
git fetch origin ${REF_JIO}
git checkout FETCH_HEAD
popd
install -D -m 0755 ${SRC_PATH_JIO}/dockerfile/cuda_package_skiplist.py /usr/local/bin/cuda-package-skiplist
EOF

# Aggregate requirements for pip-tools
RUN <<"EOF" bash -ex -o pipefail
mkdir -p /opt/pip-tools.d
pip freeze | grep wheel >> /opt/pip-tools.d/overrides.in
echo "jax[cuda13-local,k8s]>=0.8.3,<0.9" >> /opt/pip-tools.d/requirements.in
echo "-e file://${SRC_PATH_JIO}[checkpoint]" >> /opt/pip-tools.d/requirements.in
echo "setuptools>=77.0.3,<81.0.0" >> /opt/pip-tools.d/requirements.in
EOF

###############################################################################
## install Python packages
###############################################################################

FROM mealkit AS final

# Finalize installation
RUN <<"EOF" bash -ex -o pipefail
export PIP_INDEX_URL=https://download.pytorch.org/whl/cu130
export PIP_EXTRA_INDEX_URL="https://flashinfer.ai/whl/cu130 https://pypi.org/simple"
pushd /opt/pip-tools.d
pip-compile --allow-unsafe -o requirements.txt $(ls requirements*.in) --constraint overrides.in
cuda-package-skiplist filter \
  --input requirements.txt \
  --output requirements.runtime.txt \
  --skipped cuda-wheel-payload-skipped.csv
pip install --no-deps --src /opt -r requirements.runtime.txt
cuda-package-skiplist mark-installed --skipped cuda-wheel-payload-skipped.csv
pip check
popd
rm -rf ~/.cache/*
EOF


WORKDIR ${SRC_PATH_JIO}/examples
