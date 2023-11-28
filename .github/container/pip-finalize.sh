#!/bin/bash

set -exo pipefail

pushd /opt/pip-tools.d

pip-compile -o requirements.txt $(ls requirements-*.in)

pip-sync --pip-args '--src /opt' requirements.txt

rm -rf ~/.cache/*
