#!/bin/bash
set -u

_get_workflow_id() {
  WORKFLOW_RUN_ID=$1
  curl -s -L -H "Authorization: Bearer $GH_TOKEN" -H "X-GitHub-Api-Version: 2022-11-28" "https://api.github.com/repos/$REPOSITORY/actions/runs/$WORKFLOW_RUN_ID" | jq -r '.workflow_id'
}

_get_parent_run_id() {
  # This gets the parent run id in two ways:
  #  1. If the input run id is the current run id, inspect the parent run id from an environment variable named THIS_WORKFLOW_PARENT_RUN_ID. It should be set with ${{ github.event.workflow_run.id }}
  #  2. Otherwise, get parent id by downloading the artifact named "parent-run-id" which should be uploaded for all workflows
  WORKFLOW_RUN_ID=$1
  if [[ $WORKFLOW_RUN_ID -eq $THIS_WORKFLOW_RUN_ID ]]; then
    # Should be set outside of this script: export THIS_WORKFLOW_PARENT_RUN_ID=${{ github.event.workflow_run.id }}
    # This covers a special case where you cannot inspect the artifacts of the current ongoing run
    echo $THIS_WORKFLOW_PARENT_RUN_ID
    return
  fi
  
  download_url=$(curl -s -L -H "Authorization: Bearer $GH_TOKEN" -H "X-GitHub-Api-Version: 2022-11-28" "https://api.github.com/repos/$REPOSITORY/actions/runs/$WORKFLOW_RUN_ID/artifacts" | jq -r '.artifacts[] | select(.name == "parent-run-id") | .archive_download_url // ""')
  if [[ -z $download_url ]]; then
    echo ""
  else
    # Download the zipped artifact to a temporary file
    temp_zip=$(mktemp)
    curl -s -L -H "Authorization: Bearer $GH_TOKEN" -H "X-GitHub-Api-Version: 2022-11-28" $download_url -o "$temp_zip"
    # Unzip the content to another temporary file
    temp_unzipped=$(mktemp)
    unzip -p "$temp_zip" > "$temp_unzipped"
    # Read the content of the unzipped file
    cat "$temp_unzipped"
    rm "$temp_zip" "$temp_unzipped"
  fi
}

_get_root_run_id() {
  # Recursively search for the root run id. If the worfklow run name has the parent id,
  #  then no API request is needed. The workflow run name is used in cases where other
  #  instances of the workflow run have not completed, so their artifacts are not
  #  available yet
  WORKFLOW_RUN_ID=$1
  WORKFLOW_RUN_NAME=${2:-}
  ID_IN_NAME=$(echo "$WORKFLOW_RUN_NAME" | egrep -o 'parent_id=[0-9]+' | cut -d= -f2)
  if [[ -n $ID_IN_NAME ]]; then
    PARENT_ID=$ID_IN_NAME
  else
    PARENT_ID=$(_get_parent_run_id $WORKFLOW_RUN_ID)
  fi
  while [[ -n $PARENT_ID ]]; do
    WORKFLOW_RUN_ID=$PARENT_ID
    PARENT_ID=$(_get_parent_run_id $PARENT_ID)
  done
  echo $WORKFLOW_RUN_ID
}

_get_workflow_tree() {
  # This will return all workflows that 
  GH_TOKEN=$1
  REPOSITORY=$2
  GITHUB_SHA=$3
  THIS_WORKFLOW_RUN_ID=$4

  THIS_ROOT_ID=$(_get_root_run_id $THIS_WORKFLOW_RUN_ID)
  THIS_WORKFLOW_ID=$(_get_workflow_id $THIS_WORKFLOW_RUN_ID)

  # Print all workflows
  ALL_WORKFLOWS=$(curl -s -L -H "Authorization: Bearer $GH_TOKEN" -H "X-GitHub-Api-Version: 2022-11-28" "https://api.github.com/repos/$REPOSITORY/actions/runs?head_sha=$GITHUB_SHA&per_page=100" | jq -r '.workflow_runs[] | "\(.id)\t\(.workflow_id)\t\(.name)"' | sort -k1,1nr)
  echo "[DEBUG]: START all workflows:" >&2
  echo "$ALL_WORKFLOWS" >&2
  echo "[DEBUG]: END all workflows:" >&2
  # TODO: Max is 100, but may need to use jq to combine
  echo "$ALL_WORKFLOWS" | while IFS=$'\t' read -r run_id workflow_id workflow_name; do
    if [[ $workflow_id != $THIS_WORKFLOW_ID ]]; then
      continue
    fi

    root_id=$(_get_root_run_id "$run_id" "$workflow_name")
    if [[ $root_id != $THIS_ROOT_ID ]]; then
      continue
    fi

    printf "%s\t%s\t%s\t%s\n" "$run_id" "$root_id" "$workflow_id" "$workflow_name"
  done | sort -k1,1nr
}

block_and_check_if_largest() {
  if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo $#
    echo 'block_and_check_if_largest $GH_TOKEN $REPOSITORY $GITHUB_SHA $WORKFLOW_RUN_ID ${DELAY:-60}'
    echo 'Example: block_and_check_if_largest ${{ secrets.GITHUB_TOKEN }} ${{ github.repository }} ${{ github.sha }} ${{ github.run_id }} 60'
    echo 'Requires you to set the following outside this script: export THIS_WORKFLOW_PARENT_RUN_ID=${{ github.event.workflow_run.id }}'
    echo 'Exits 0 if it is the largest run_id'
    return 1
  fi
  GH_TOKEN=$1
  REPOSITORY=$2
  GITHUB_SHA=$3
  THIS_WORKFLOW_RUN_ID=$4
  DELAY=${5:-60}

  RAW_WORKFLOW_PATH=$(curl -s -L -H "Authorization: Bearer $GH_TOKEN" -H "X-GitHub-Api-Version: 2022-11-28" "https://api.github.com/repos/$REPOSITORY/actions/runs/$THIS_WORKFLOW_RUN_ID" | jq -r '. | "\(.head_sha)/\(.path)"')
  RAW_WORKFLOW_URL=https://raw.githubusercontent.com/$REPOSITORY/$RAW_WORKFLOW_PATH
  NUM_DEPENDENCIES=$(curl -s $RAW_WORKFLOW_URL | yq '.on.workflow_run.workflows | length')

  while true; do
    echo Sleeping for $DELAY sec
    sleep $DELAY
    workflows=$(_get_workflow_tree $GH_TOKEN $REPOSITORY $GITHUB_SHA $THIS_WORKFLOW_RUN_ID)
    num_workflows=$(echo "$workflows" | wc -l)
    largest_id=$(echo "$workflows" | cut -f1 | sort -nr | head -n1)
    echo "========== $(date)"
    echo "$workflows"
    if [[ $largest_id -eq $THIS_WORKFLOW_RUN_ID && $num_workflows -eq $NUM_DEPENDENCIES ]]; then
      echo "This workflow run id is the largest: $THIS_WORKFLOW_RUN_ID"
      break
    elif [[ $largest_id -ne $THIS_WORKFLOW_RUN_ID ]]; then
      echo "This workflow run id is NOT the largest: $THIS_WORKFLOW_RUN_ID"
      exit 1
    fi
  done
}

