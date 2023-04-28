#!/bin/bash

set -ex

export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles

apt-get update
apt-get install -y \
    libcudnn8 \
    libcudnn8-dev

apt-get clean
rm -rf /var/lib/apt/lists/*
