#!/bin/bash

# Grab the CC (compute capability) of all available GPUs
cc_check=$(python -c "import jax; n=len(jax.devices()); jax.pmap(lambda x: x ** 2)(jax.numpy.arange(n))")

if [[ ! -z "$check_cc" ]]; then
    YELLOW='\033[0;33m'
    NOCOLOR='\033[0m'

    echo -e "${YELLOW}

WARNING: If you're going to use jax.pmap(), be aware that it requires all of the 
participating devices to be identical. You can filter in the identical GPUs by using
the following cmd:  

    $ docker run ... --gpus=0,1,2,...,N  ...

where 0,1,2,...,N are IDs of the identical GPUs.
${NOCOLOR}"
fi
