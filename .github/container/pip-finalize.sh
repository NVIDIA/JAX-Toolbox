#!/bin/bash

set -ex -o pipefail

pip-compile $(ls /opt/pip-tools.d/manifest.*) -o /opt/pip-tools.d/requirements.txt

pip-sync /opt/pip-tools.d/requirements.txt

rm -rf ~/.cache/*
