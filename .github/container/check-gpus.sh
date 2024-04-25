#!/bin/bash

YELLOW='\033[0;33m'
NOCOLOR='\033[0m'


# Check that all devices are identical
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    # CUDA_VISIBLE_DEVICES is not set. Check what GPUs are available
    diff_gpu_count=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | sort | uniq | wc -l)
    if [[ $diff_gpu_count -ge 2 ]]; then
            echo -e "${YELLOW}

WARNING: There are $diff_gpu_count different GPUs in the system:

$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | sort | uniq)

Using all of them at once with JAX may cause the JAX to fail. To avoid that you can either specify CUDA_VISIBLE_DEVICES env variable:
   1. inside container:

        $ export CUDA_VISIBLE_DEVICES=<coma-separated GPU ids>

     or

        $ CUDA_VISIBLE_DEVICES=<coma-separated GPU ids> <your-script-to-run>

    2. at docker run:

        $ docker run ... --gpus=<coma-separated GPU ids> ...
    
       or

         $ docker run ... -e CUDA_VISIBLE_DEVICES=<coma-separated GPU ids> ...

where <coma-separated GPU ids> are IDs of the identical GPUs.

Example:

        docker run ... -e CUDA_VISIBLE_DEVICES=0,1,4,6 ...

${NOCOLOR}"
    fi
else
    # CUDA_VISIBLE_DEVICES env variable is set. Check only GPUs that are mentioned in it. 
    available_gpus=$(nvidia-smi --query-gpu=index,gpu_name --format=csv,noheader | sort)

    # Map available GPU idx and its name
    declare -A devices
    while IFS= read -r line; do
        n=$(echo $line | cut -d "," -f1)
        device_name=$(echo $line | cut -d "," -f2)
        devices[$n]="$device_name"
    done <<< "$available_gpus"

    # Loop thru GPUs in CUDA_VISIBLE_DEVICES and check if they are identical
    ID=""
    issue_warning=false
    devices_list=""
    while IFS=',' read -ra vis_dev; do
    for i in "${vis_dev[@]}"; do
        current_device=${devices["$i"]}
        devices_list="$devices_list$i: $current_device\n"
        if [[ -z $ID ]]; then 
            ID=$current_device
        else
            if [[ $ID != $current_device ]]; then
                issue_warning=true
            fi
        fi
    done
    done <<< "$CUDA_VISIBLE_DEVICES"

    if [[ "$issue_warning" == "true" ]]; then 
            echo -e "${YELLOW}

WARNING: There are different types of GPUs specified by CUDA_VISIBLE_DEVICES.
Using them all at once with JAX may cause JAX to fail. To avoid this, please restrict to only using identical GPUs.
List of devices in use:

${devices_list}
${NOCOLOR}"                
    fi
fi