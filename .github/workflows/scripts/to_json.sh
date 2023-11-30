#!/bin/bash

# convert a list of variables to a json dictionary
function to_json() {
  CMD="jq -n "
  CMD+=$(for var in "$@"; do
    echo "--arg _$var \"\$$var\" "
  done)

  JSON=$(for var in "$@"; do
    echo "$var: \$_$var, "
  done)
  CMD+=\'{$JSON}\'

  eval $CMD
}
