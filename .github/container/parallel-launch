#!/bin/bash
set -e
if (( $# < 3 )); then
  echo "Usage: $0 <var_name> <num_processes> <command ...>"
  echo "launches command 'num_processes' times in parallel with 'var_name' set to 0..num_processes-1"
  exit 1
fi
VAR_NAME=$1
shift
NPROCS=$1
shift
positive_integer='^[1-9][0-9]*$'
if ! [[ $NPROCS =~ $positive_integer ]]; then
  echo "Second argument must be a positive number of processes; got $NPROCS"
  exit 1
fi
pids=()
echo "Launching $@ $NPROCS times in parallel"
for (( i=0; i<$NPROCS; i++ )); do
  export $VAR_NAME=$i
  "$@" &
  pids+=($!)
done
for pid in ${pids[*]}; do
  wait $pid
done
jobs
