#!/bin/bash

# waits for a Slurm job to complete, fail, or be cancelled.
function wait_for_slurm_job() {
    job_id=$1
    check_every=${2:-15}

    while true; do
        status=$(sacct --job $job_id --noheader --format=State --parsable2)

        if [ -z "$status" ]; then
            break
        else
            echo "[$(date)] job $job_id: $status"
        fi

        # Wait for a bit before checking again
        sleep $check_every
    done
}
