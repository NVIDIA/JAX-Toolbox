#!/bin/bash

source $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/inspect_remote_img.sh
get_build_date() {
  if [[ $# -ne 2 ]]; then
    echo 'get_build_date $GH_TOKEN $IMAGE'
    echo 'Example: get_build_date XXXXXXXXXXXX ghcr.io/nvidia/t5x:latest'
    echo 'Returns the BUILD_DATE of a tagged remote image (no download)'
    return 1
  fi
  inspect_remote_img $@ | jq -r '.config.Labels["org.opencontainers.image.created"]'
}
