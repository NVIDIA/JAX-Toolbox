#!/bin/bash

if [ -z "${DISABLE_TCPX_CHECK}" ]; then

# Colors
GREEN='\033[0;32m'
NOCOLOR='\033[0m'


# Attempt to retrieve the instance ID from the GCP metadata server
INSTANCE_ID=$(curl  -m 1 -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/id" || true)

if [ -n "$INSTANCE_ID" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/tcpx/lib64"
    export NCCL_NET=GPUDirectTCPX_v7

    echo -e "${GREEN}

    It looks like you're running on GCP. In order to maximize your multi-GPU performance,
    you'll need to use Google's TCPx NCCL plugin. This should already be installed for you 
    and is located at /usr/local/tcpx in this container. 
    
    However, there are additional steps you will need to take. Mainly, you'll need to run a 
    separate receive-datapath-manager daemon on each of your nodes, and correctly configure your 
    networks and NICs.

    For more information, please see the guide at: 
    https://cloud.google.com/compute/docs/gpus/gpudirect#provide-access

    (To disable this message, set DISABLE_TCPX_CHECK=1)
    ${NOCOLOR}"
fi
fi