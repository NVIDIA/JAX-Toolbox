#!/bin/bash

source $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/inspect_remote_img.sh
get_remote_labels() {
  if [[ $# -ne 4 ]]; then
    echo '[get_remote_labels](./get_remote_labels.sh) $GH_TOKEN $IMAGE $OS $ARCH'
    echo 'Example: get_remote_labels XXXXXXXXXXXX ghcr.io/nvidia/upstream-t5x:latest linux amd64'
    echo 'Returns the opencontainer annotation labels of a tagged remote image (no download)'
    return 1
  fi
  inspect_remote_img $@ | jq .config.Labels
}
