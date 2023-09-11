#!/bin/bash

source $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/inspect_remote_img.sh
get_remote_labels() {
  inspect_remote_img $@ | jq .config.Labels
}
