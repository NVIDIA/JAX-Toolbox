#!/bin/bash

source $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/inspect_remote_img.sh
get_build_date() {
  inspect_remote_img $@ | jq -r '.config.Labels["org.opencontainers.image.created"]'
}
