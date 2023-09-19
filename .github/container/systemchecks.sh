#!/bin/bash

# Grab the second line / second field of the output of `df`, 
# which is the size of shm in KBs. 
shm_size=$(df /dev/shm | awk 'NR==2 {print $2}')
minimum_shm_size=1048576 # ~1GB in KBs

# Yellow-ish
YELLOW='\033[0;33m'

# Stop using terminal colors.
STOPCOLOR='\033[0m'

SHM_WARNING="${YELLOW}
WARNING: Your shm is currenly less than 1GB. This may cause SIGBUS errors.
To avoid this problem, you can manually set the shm size in docker with:

$ docker run ... --shm-size=1g ...
${STOPCOLOR}"

if (( 0 < minimum_shm_size )); then
    echo -e "$SHM_WARNING"
fi
