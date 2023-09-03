#!/bin/bash

# convert a list of variables to a json dictionary
function to_json() {
  jq -n \
    $(for var in "$@"; do echo --arg $var "${!var}"; done) \
    "{$(for var in "$@"; do echo "\"$var\": \$$var",; done)}"
}