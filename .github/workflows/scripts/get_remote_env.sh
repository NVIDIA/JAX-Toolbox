#!/bin/bash

source $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/inspect_remote_img.sh
get_remote_env() {
  if [[ $# -ne 4 ]]; then
    echo 'get_remote_env $GH_TOKEN $IMAGE $OS $ARCH'
    echo 'Example: get_remote_env XXXXXXXXXXXX ghcr.io/nvidia/upstream-t5x:latest linux amd64'
    echo 'Returns the ENV of a tagged remote image (no download)'
    echo 'Useful to inspect CUDA env vars'
    return 1
  fi
  inspect_remote_img $@ | jq .config.Env
}
