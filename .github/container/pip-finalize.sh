#!/bin/bash

set -ex -o pipefail

pushd /opt/pip-tools.d

pip-compile -o requirements.txt $(ls manifest.*)

pip-sync --pip-args '--src /opt' requirements.txt

rm -rf ~/.cache/*
