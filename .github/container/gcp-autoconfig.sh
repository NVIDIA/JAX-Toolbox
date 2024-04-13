#!/bin/bash

if [ "${DISABLE_GCP_TCPX_SETUP}" == 1 ]; then 
  exit 0
fi

# Colors
GREEN='\033[0;32m'
NOCOLOR='\033[0m'

# If google.internal exists, we are most likely running on GCP.
if host metadata.google.internal &> /dev/null; then
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

    If you believe this setup is causing undesired effects, you can disable it
    by setting DISABLE_GCP_TCPX_SETUP=1
    ${NOCOLOR}"

    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TCPX_LIBRARY_PATH}"
    export NCCL_NET=GPUDirectTCPX_v7

    # find the NIC that is NUMA-aligned with I/O devices
    export NCCL_SOCKET_IFNAME=$(
      lstopo  --no-caches --no-smt --filter core:none --of xml |\
      yq -p=xml --xml-attribute-prefix='' --xml-skip-directives --xml-skip-proc-inst -o=json |\
      jq -c '.. | objects | select(.type == "Bridge" and (.object | type == "array"))' |\
      grep -i nvme |\
      jq -r '.. | objects | select(if .info | type == "array" then .info[1].value | contains("Ethernet") else false end ) | .object.name'
    )
    export NCCL_GPUDIRECTTCPX_CTRL_DEV=$NCCL_SOCKET_IFNAME
    # find NICs that are NUMA-aligned with the NVIDIA GPUs
    export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=$(
      lstopo  --no-caches --no-smt --filter core:none --of xml |\
      yq -p=xml --xml-attribute-prefix='' --xml-skip-directives --xml-skip-proc-inst -o=json |\
      jq -c '.. | objects | select(.type == "Bridge" and (.object | type == "array"))' |\
      grep -i nvidia |\
      jq -rs '[.. | objects | select(if .info | type == "array" then .info[1].value | contains("Ethernet") else false end) | .object.name] | join(",")'
    )
    export UDS_PATH="/run/tcpx-${SLURM_JOB_ID}"
    export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX=${UDS_PATH}
    export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=1000000
    export NCCL_GPUDIRECTTCPX_FORCE_ACK=0

    echo "[GCP autoconfig $(hostname)] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} NCCL_GPUDIRECTTCPX_CTRL_DEV=${NCCL_GPUDIRECTTCPX_CTRL_DEV} NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=${NCCL_GPUDIRECTTCPX_SOCKET_IFNAME}"
fi
