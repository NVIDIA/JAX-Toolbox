#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Copies the patches from within an image to the GIT_ROOT/rosetta/patches dir"
  echo
  echo "Usage: $0 <image> <rosetta dir: default GIT_ROOT/rosetta/>"
  exit 1
fi

IMAGE=$1
ROSETTA_DIR=${2:-$(readlink -f ../)}

container_id=$(docker create $IMAGE)
docker cp $container_id:/opt/rosetta/patches $ROSETTA_DIR
docker rm -v $container_id
