#!/bin/bash
get_build_date() {
  if [[ $# -ne 3 ]]; then
    echo 'get_build_date $IMAGE $OS $ARCH'
    echo 'Example: get_build_date ghcr.io/nvidia/upstream-t5x:latest linux amd64'
    echo 'Returns the BUILD_DATE of a tagged remote image (no download)'
    return 1
  fi
  skopeo inspect --override-arch "${3}" --override-os "${2}" "docker://${1}"  | jq -r '.Labels["org.opencontainers.image.created"]'
}
