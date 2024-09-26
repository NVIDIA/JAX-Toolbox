#!/bin/bash

# 1. Check if AWS EFA installation script completed successfully
check=$(/opt/amazon/efa/bin/fi_info --version | grep "libfabric")
if [[ -z "$check" ]]; then
    echo "Fail to install AWS EFA"
    exit 1
fi

echo "AWS EFA installed successfully"

exit 0