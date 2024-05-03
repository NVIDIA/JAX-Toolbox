#!/bin/bash

YELLOW='\033[0;33m'
NOCOLOR='\033[0m'


VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq 0 $(($(nvidia-smi -L | wc -l)-1)))}
available_gpus=$(nvidia-smi --query-gpu=index,gpu_name --format=csv,noheader | sort)
# Map available GPU idx and its name
declare -A devices
while IFS= read -r line; do
    n=$(echo $line | cut -d "," -f1)
    device_name=$(echo $line | cut -d "," -f2)
    devices[$n]="$device_name"
done <<< "$available_gpus"
# Loop thru GPUs in VISIBLE_DEVICES and check if they are identical
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
done <<< "$VISIBLE_DEVICES"
if [[ "$issue_warning" == "true" ]]; then 
            echo -e "${YELLOW}

WARNING: There are different GPU models present.
Using them all at once with the JAX may cause JAX to fail. To avoid this, please restrict to only using identical GPUs:
   1. inside a container:

        $ export CUDA_VISIBLE_DEVICES=<coma-separated GPU ids>

     or

        $ CUDA_VISIBLE_DEVICES=<coma-separated GPU ids> <your-script-to-run>

    2. when starting a container:

        $ docker run ... --gpus=<coma-separated GPU ids> ...
    
       or

         $ docker run ... -e CUDA_VISIBLE_DEVICES=<coma-separated GPU ids> ...

where <coma-separated GPU ids> are IDs of the identical GPUs.

Example:

        docker run ... -e CUDA_VISIBLE_DEVICES=0,1,4,6 ...


List of devices in use:

${devices_list}
${NOCOLOR}"                
fi
