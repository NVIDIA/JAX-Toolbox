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

ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04
ARG URL_JIO=https://github.com/NVIDIA/JAX-Toolbox.git
ARG REF_JIO=main
ARG URL_TUNIX=https://github.com/google/tunix.git
ARG REF_TUNIX=main
ARG BASE_PATH_JIO=/opt/jtbx
ARG SUB_PATH_JIO=jax-inference-offloading
ARG SRC_PATH_TUNIX=/opt/tunix

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

pip install pip-tools pip-mark-installed
rm -rf ~/.cache/pip
EOF

###############################################################################
## Download source and configure pip-tools
###############################################################################

FROM base AS mealkit
ARG URL_JIO
ARG REF_JIO
ARG URL_TUNIX
ARG REF_TUNIX
ARG BASE_PATH_JIO
ARG SUB_PATH_JIO
ARG SRC_PATH_TUNIX

ENV SRC_PATH_JIO=${BASE_PATH_JIO}/${SUB_PATH_JIO}

# Check out source code
RUN <<"EOF" bash -ex -o pipefail
git clone --no-checkout ${URL_JIO} ${BASE_PATH_JIO}
pushd ${BASE_PATH_JIO}
git sparse-checkout init --cone
git sparse-checkout set ${SUB_PATH_JIO}
git fetch origin ${REF_JIO}
git checkout FETCH_HEAD
ls -lR ${SUB_PATH_JIO}
popd
git clone --branch ${REF_TUNIX} ${URL_TUNIX} ${SRC_PATH_TUNIX}
EOF

# Aggregate requirements for pip-tools
# RUN <<"EOF" bash -ex -o pipefail
# mkdir -p /opt/pip-tools.d
# pip freeze | grep wheel >> /opt/pip-tools.d/overrides.in
# echo "jax[cuda12_local]" >> /opt/pip-tools.d/requirements.in
# echo "-e file://${SRC_PATH_JIO}" >> /opt/pip-tools.d/requirements.in
# echo "-e file://${SRC_PATH_TUNIX}" >> /opt/pip-tools.d/requirements.in
# cat "${SRC_PATH_JIO}/examples/requirements.in" >> /opt/pip-tools.d/requirements.in
# EOF
RUN mkdir -p /opt/pip-tools.d
RUN pip freeze | grep wheel >> /opt/pip-tools.d/overrides.in
RUN echo "jax[cuda12_local]" >> /opt/pip-tools.d/requirements.in
RUN echo "-e file://${SRC_PATH_JIO}" >> /opt/pip-tools.d/requirements.in
RUN echo "-e file://${SRC_PATH_TUNIX}" >> /opt/pip-tools.d/requirements.in
RUN cat "${SRC_PATH_JIO}/examples/requirements.in" >> /opt/pip-tools.d/requirements.in


###############################################################################
## install Python packages
###############################################################################

FROM mealkit AS final

# Finalize installation
RUN <<"EOF" bash -ex -o pipefail
export PIP_INDEX_URL=https://download.pytorch.org/whl/cu129
export PIP_EXTRA_INDEX_URL=https://pypi.org/simple
pushd /opt/pip-tools.d
pip-compile -o requirements.txt $(ls requirements*.in) --constraint overrides.in
# remove cuda wheels from install list since the container already has them
sed -i 's/^nvidia-/# nvidia-/g' requirements.txt
pip install --no-deps --src /opt -r requirements.txt
# make pip happy about the missing torch dependencies
pip-mark-installed nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 \
  nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 \
  nvidia-cufft-cu12 nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 nvidia-nvtx-cu12
popd
rm -rf ~/.cache/*
EOF

WORKDIR ${SRC_PATH_JIO}/examples
