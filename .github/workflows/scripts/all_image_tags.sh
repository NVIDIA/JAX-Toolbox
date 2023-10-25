#!/bin/bash

all_image_tags() {
  if [[ $# -ne 2 ]]; then
    echo 'all_image_tags $GH_TOKEN $IMAGE_REPO'
    echo Example: 'all_image_tags XXXXXXXXXXXX ghcr.io/nvidia/t5x'
    echo Returns all tags on ghcr.io for a given image repo
    return 1
  fi
  GH_TOKEN=$1
  # If you have image: ghcr.io/nvidia/t5x:nightly-YYYY-MM-DD then IMAGE_REPO=ghcr.io/nvidia/t5x
  IMAGE_REPO=$2

  PACKAGE=$(echo $IMAGE_REPO | rev | cut -d/ -f1 | rev)
  ORG=$(echo $IMAGE_REPO | rev | cut -d/ -f2 | rev)
  # Set n=123456789 to get all tags (impossibly large value)
  curl -s -H "Authorization: Bearer $(echo $GH_TOKEN | base64 )" "https://ghcr.io/v2/$ORG/$PACKAGE/tags/list?n=123456789" | jq -r '.tags[]'
}
