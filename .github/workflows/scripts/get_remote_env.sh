#!/bin/bash
get_remote_env() {
  if [[ $# -ne 3 ]]; then
    echo 'get_remote_env $IMAGE $OS $ARCH'
    echo 'Example: get_remote_env ghcr.io/nvidia/upstream-t5x:latest linux amd64'
    echo 'Returns the ENV of a tagged remote image (no download)'
    echo 'Useful to inspect CUDA env vars'
    return 1
  fi
  skopeo inspect --override-arch "${3}" --override-os "${2}" "docker://${1}"  | jq -r '.Env'
}
