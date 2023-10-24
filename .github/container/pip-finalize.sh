#!/bin/bash

pip-compile /opt/pip-tools.d/*.in -o /opt/pip-tools.d/requirements.txt

pip-sync /opt/pip-tools.d/requirements.txt

rm -rf ~/.cache/*
