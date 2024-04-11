#!/bin/bash

if [ "${DISABLE_GCP_TCPX_SETUP}" == 1 ]; then 
  exit 0
fi

# Colors
GREEN='\033[0;32m'
NOCOLOR='\033[0m'

# If google.internal exists, we are most likely running on GCP.
if host "google.internal" > /dev/null; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TCPX_LIBRARY_PATH}"
    export NCCL_NET=GPUDirectTCPX_v7

    echo -e "${GREEN}
    ========================== JAX-ToolBox on GCP ==========================

    It looks like you're running on GCP. In order to maximize your multi-node performance,
    you'll need to use Google's TCPx NCCL plugin. This container ships the plugin at
    ${TCPX_LIBRARY_PATH}, which is already added to LD_LIBRARY_PATH by this script $1.

    However, there are additional steps you will need to take. Mainly, you'll need to run a 
    separate receive-datapath-manager daemon on each of your nodes, and correctly configure your 
    networks and NICs.

    For more information, please see the guide at: 
    https://cloud.google.com/compute/docs/gpus/gpudirect#provide-access

    If you believe this setup is interfering with your work, you can disable it
    by setting DISABLE_GCP_TCPX_SETUP=1
    ${NOCOLOR}"
fi
