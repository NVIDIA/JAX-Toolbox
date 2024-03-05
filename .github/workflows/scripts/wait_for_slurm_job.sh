#!/bin/bash

# waits for a Slurm job to complete, fail, or be cancelled.
function wait_for_slurm_job() {

    opts=$(set +o)
    set +x

    host=$1
    job_id=$2
    check_every=${3:-15}

    while true; do
        status=$(ssh $host squeue --job $job_id --noheader --format=%T 2>/dev/null || echo "SSH error: $?")
        echo "[$(date)] job $job_id: $status"

        if [ -z "$status" ]; then
            break
        fi

        sleep $check_every
    done

    eval "$opts"
}
