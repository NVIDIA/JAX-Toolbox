#!/bin/bash

source $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/inspect_remote_img.sh
get_remote_env() {
  inspect_remote_img $@ | jq .config.Env
}
