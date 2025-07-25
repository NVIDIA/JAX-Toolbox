#!/bin/bash

# convert a list of variables to a json dictionary
function to_json() {
  CMD="jq -n "
  CMD+=$(for var in "$@"; do
    echo "$([[ "${!var}" =~ ^[0-9]+$ ]] && echo --argjson || echo --arg) _$var \"\$$var\" "
  done)

  JSON=$(for var in "$@"; do
    echo "$var: \$_$var, "
  done)
  CMD+=\'{$JSON}\'

  eval $CMD
}

XPK_EXIT_CODE=1
summary="GKE-neor"
outcome=success
badge_color=brightgreen

if [ "${XPK_EXIT_CODE}" -gt 0 ]; then
    badge_color=red
    outcome=failed
    summary+=": fail"
else
    summary+=": pass"
fi
to_json summary \
        badge_color \
        outcome | \
tee sitrep.json
cat sitrep.json
