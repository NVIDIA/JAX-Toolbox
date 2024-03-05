#!/bin/bash

# waits for a Slurm job to complete, fail, or be cancelled.
function wait_for_slurm_job() {
    host=$1
    job_id=$2
    check_every=${3:-15}

    while true; do
        status=$(ssh $host squeue --job $job_id --noheader --format=%T)

        echo "status = '$status'"

        if [ -z "$status" ]; then
            break
        else
            echo "[$(date)] job $job_id: $status"
        fi

        # Wait for a bit before checking again
        sleep $check_every
    done
}
