#!/bin/bash

set -ex -o pipefail

pushd /opt/pip-tools.d

pip-compile $(ls manifest.*) -o requirements.txt

pip-sync --pip-args '--src /opt' requirements.txt

rm -rf ~/.cache/*
