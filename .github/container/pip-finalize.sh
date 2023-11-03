#!/bin/bash

set -ex -o pipefail

pip-compile $(ls /opt/pip-tools.d/manifest.*) -o /opt/pip-tools.d/requirements.txt

pip install --src /opt -r /opt/pip-tools.d/requirements.txt

rm -rf ~/.cache/*
