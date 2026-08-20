#!/usr/bin/env bash

set -euo pipefail

pod_name="${1:?Usage: read-pod-exit-code.sh POD_NAME CONTAINER_NAME}"
container_name="${2:?Usage: read-pod-exit-code.sh POD_NAME CONTAINER_NAME}"

if ! pod_json="$(kubectl get pod "${pod_name}" -o json)"; then
  echo "Failed to read Kubernetes status for pod ${pod_name}" >&2
  exit 1
fi

exit_code="$(
  jq -r --arg container_name "${container_name}" '
    first(
      .status.containerStatuses[]?
      | select(.name == $container_name)
      | .state.terminated.exitCode
    ) // empty
  ' <<< "${pod_json}"
)"

if [[ ! "${exit_code}" =~ ^[0-9]+$ ]]; then
  pod_phase="$(jq -r '.status.phase // "Unknown"' <<< "${pod_json}")"
  echo \
    "Container ${container_name} in pod ${pod_name} has no terminated exit code (pod phase: ${pod_phase})" \
    >&2
  exit 1
fi

printf '%s\n' "${exit_code}"
