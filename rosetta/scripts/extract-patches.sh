#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Copies the patches from within an image to the GIT_ROOT/rosetta/patches dir"
  echo
  echo "Usage: $0 <image> <patch dir in image: default /opt/manifest.d/patches> <rosetta dir: default GIT_ROOT/rosetta/>"
  exit 1
fi

IMAGE=$1
IMAGE_PATCH_DIR=${2:-"/opt/manifest.d/patches"}
ROSETTA_DIR=${3:-$(readlink -f ../)}

container_id=$(docker create $IMAGE)
docker cp $container_id:$IMAGE_PATCH_DIR $ROSETTA_DIR
docker rm -v $container_id
