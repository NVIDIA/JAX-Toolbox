#!/bin/bash

# convert a list of variables to a json dictionary
function to_json() {
  eval $(echo jq -n \
    $(for var in "$@"; do echo --arg _$var "'"${!var}"'"; done) \
    \'"{$(for var in "$@"; do echo -n "\"$var\": \$_$var, "; done)}"\'
  )
}
