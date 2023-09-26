#!/bin/bash

source $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/inspect_remote_img.sh
get_remote_labels() {
  if [[ $# -ne 2 ]]; then
    echo '[get_remote_labels](./get_remote_labels.sh) $GH_TOKEN $IMAGE'
    echo 'Example: get_remote_labels XXXXXXXXXXXX ghcr.io/nvidia/t5x:latest'
    echo 'Returns the opencontainer annotation labels of a tagged remote image (no download)'
    return 1
  fi
  inspect_remote_img $@ | jq .config.Labels
}
