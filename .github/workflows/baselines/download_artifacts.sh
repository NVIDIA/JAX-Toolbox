#!/bin/bash

set -euo pipefail
usage() {
    echo -e "Usage: ${0} WORKFLOW_IDS..."
    exit 1
}

if [[ -z $GH_TOKEN ]]; then
  echo "GH_TOKEN env var must be set to download artifacts. Please export the GH_TOKEN var."
  echo "You can create a personal access token here: https://github.com/settings/tokens"
  echo "For more information, see GitHub official docs: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens"
  exit 1
fi

[ "$#" -ge "1" ] || usage


for WORKFLOW_RUN in $*; do
  mkdir -p $WORKFLOW_RUN
  pushd $WORKFLOW_RUN
  # cURL the list of artifacts
  ARTIFACTS=$(curl -L \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/NVIDIA/JAX-Toolbox/actions/runs/${WORKFLOW_RUN}/artifacts")

  COUNT=$(echo $ARTIFACTS | jq -r '.total_count')
  NAMES=$(echo $ARTIFACTS | jq -r '.artifacts[].name')
  URLS=$(echo $ARTIFACTS | jq -r '.artifacts[].archive_download_url')
  NAMES=($NAMES)
  URLS=($URLS)

  # Download artifacts
  for (( i=0; i<$COUNT; i++ )); do
    N=${NAMES[$i]}
    U=${URLS[$i]}

    curl -L \
      -H "Accept: application/vnd.github+json" \
      -H "Authorization: Bearer ${GH_TOKEN}" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      --output "${N}.zip" \
      "${U}"

    unzip ${N}.zip -d ${N}
    rm ${N}.zip
  done

  popd
done
