#!/bin/bash

set -ex -o pipefail

sed -i "s|flax @ git+https://github.com/google/flax#egg=flax||g" /opt/pip-tools.d/manifest.*

pip-compile $(ls /opt/pip-tools.d/manifest.*) -o /opt/pip-tools.d/requirements.txt

sed -i "s|flax @ git+https://github.com/google/flax.git|-e /opt/flax|g" /opt/pip-tools.d/requirements.txt

pip-sync /opt/pip-tools.d/requirements.txt

rm -rf ~/.cache/*
