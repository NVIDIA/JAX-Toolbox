#!/bin/bash

inspect_remote_img() {
  if [[ $# -ne 4 ]]; then
    echo 'inspect_remote_img $GH_TOKEN $IMAGE $OS $ARCH'
    echo 'Example: inspect_remote_img XXXXXXXXXXXX ghcr.io/nvidia/upstream-t5x:latest linux amd64'
    echo 'Returns the metadata of a tagged remote image (no download)'
    return 1
  fi
  GH_TOKEN=$1
  IMAGE=$2
  OS=$3
  ARCH=$4

  IMAGE_REPO=$(echo $IMAGE | rev | cut -d: -f2- | rev)

  PACKAGE=$(echo $IMAGE_REPO | rev | cut -d/ -f1 | rev)
  ORG=$(echo $IMAGE_REPO | rev | cut -d/ -f2 | rev)

  config_digest=$(docker manifest inspect -v $IMAGE | jq -r ".[] | select(.Descriptor.platform == {\"architecture\": \"$ARCH\", \"os\": \"$OS\"}) | .OCIManifest.config.digest + .SchemaV2Manifest.config.digest")
  curl -s -L -H "Authorization: Bearer $(echo $GH_TOKEN | base64)" "https://ghcr.io/v2/$ORG/$PACKAGE/blobs/$config_digest"
}
export -f inspect_remote_img
