#!/bin/bash

inspect_remote_img() {
  GH_TOKEN=$1
  IMAGE=$2

  TAG=$(echo $IMAGE | rev | cut -d: -f1 | rev)
  IMAGE_REPO=$(echo $IMAGE | rev | cut -d: -f2- | rev)

  PACKAGE=$(echo $IMAGE_REPO | rev | cut -d/ -f1 | rev)
  ORG=$(echo $IMAGE_REPO | rev | cut -d/ -f2 | rev)

  top_manifest_digest=$(curl -s -H "Authorization: Bearer $(echo $GH_TOKEN | base64)" "https://ghcr.io/v2/$ORG/$PACKAGE/manifests/$TAG" | jq -r .manifests[0].digest)
  config_digest=$(curl -s -H "Authorization: Bearer $(echo $GH_TOKEN | base64)" "https://ghcr.io/v2/$ORG/$PACKAGE/manifests/$top_manifest_digest" | jq -r .config.digest)
  img_metadata=$(curl -s -L -H "Authorization: Bearer $(echo $GH_TOKEN | base64)" "https://ghcr.io/v2/$ORG/$PACKAGE/blobs/$config_digest")
  echo $img_metadata
}
export -f inspect_remote_img
