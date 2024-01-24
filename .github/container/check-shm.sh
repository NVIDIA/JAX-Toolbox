#!/bin/bash

minimum_shm_size=1048576 # ~1GB in KBs

# Grab the second line / second field of the output of `df`, 
# which is the size of shm in KBs. 
actual_shm_size=$(df /dev/shm | awk 'NR==2 {print $2}')

if (( actual_shm_size < minimum_shm_size )); then
    YELLOW='\033[0;33m'
    NOCOLOR='\033[0m'

    echo -e "${YELLOW}
WARNING: Your shm is currenly less than 1GB. This may cause SIGBUS errors.
To avoid this problem, you can manually set the shm size in docker with:

    $ docker run ... --shm-size=1g ...
${NOCOLOR}"
fi
