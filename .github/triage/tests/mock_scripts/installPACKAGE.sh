#!/bin/bash
new_version=$1
echo "Setting PACKAGE_VERSION to $new_version"
cat ${JAX_TOOLBOX_TRIAGE_PREFIX}.env
sed -i "s|^PACKAGE_VERSION=.*$|PACKAGE_VERSION=$new_version|" ${JAX_TOOLBOX_TRIAGE_PREFIX}.env
cat ${JAX_TOOLBOX_TRIAGE_PREFIX}.env
