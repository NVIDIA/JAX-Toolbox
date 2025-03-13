#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 SLURM_LOGIN_USER SLURM_LOGIN_HOSTNAME SLURM_JOB_ID LOG_FILE" >&2
  exit 1
fi

LOGIN_USER="$1"
LOGIN_HOST="$2"
JOB_ID="$3"
LOG_FILE="$4"

SSH_CMD="ssh ${LOGIN_USER}@${LOGIN_HOST}"

JOB_INFO=$(${SSH_CMD} "sacct -j ${JOB_ID} --format=JobID,JobName,State,Exitcode --parsable2 --noheader | grep -E '^[0-9]+\|'" || true)
SLURM_STATE=$(echo "$JOB_INFO" | cut -f3 -d"|")
SLURM_EXITCODE=$(echo "$JOB_INFO" | cut -f4 -d"|")

echo "SLURM_STATE=${SLURM_STATE}" >> "$GITHUB_OUTPUT"
echo "SLURM_EXITCODE=${SLURM_EXITCODE}" >> "$GITHUB_OUTPUT"

echo "===== SLURM LOG TAIL ====="
${SSH_CMD} "tail -n 200 ${LOG_FILE}"
echo "===== END OF SLURM LOG ====="
