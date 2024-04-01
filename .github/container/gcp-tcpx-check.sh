#!/bin/bash

if [ -z "${DISABLE_TCPX_CHECK}" ]; then
# Colors
YELLOW='\033[0;33m'
NOCOLOR='\033[0m'

# Paths
FILE="libnccl.so"
DIR1="/var/lib/tcpx/lib64"
DIR2="/usr/local/tcpx/lib64"

# Attempt to retrieve the instance ID from the GCP metadata server
INSTANCE_ID=$(curl  -m 1 -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/id" || true)

if [ -n "$INSTANCE_ID" ]; then
    if [ -f "$DIR1/$FILE" ]; then
        :
    elif [ -f "$DIR2/$FILE" ]; then
        :
    else
            echo -e "${YELLOW}
WARNING: It looks like you are running on GCP, but we did not find libnccl.so
at either /var/lib/tcpx/lib64 or /usr/local/tcpx/lib64. You main end up defaulting
to a significantly worse network runtime if you are using multiple nodes.

To fix this, please follow the instructions at https://cloud.google.com/compute/docs/gpus/gpudirect

To disable this message, set DISABLE_TCPX_CHECK=1
${NOCOLOR}

"
    fi
fi

fi
