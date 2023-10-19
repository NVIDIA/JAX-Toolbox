#!/bin/bash
get_remote_labels() {
  if [[ $# -ne 3 ]]; then
    echo 'get_remote_labels $IMAGE $OS $ARCH'
    echo 'Example: get_remote_labels ghcr.io/nvidia/upstream-t5x:latest linux amd64'
    echo 'Returns the opencontainer annotation labels of a tagged remote image (no download)'
    return 1
  fi
  skopeo inspect --override-arch "${3}" --override-os "${2}" "docker://${1}"  | jq -r '.Labels'
}
