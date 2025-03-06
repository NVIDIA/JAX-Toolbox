#!/bin/bash

GPUS_PER_NODE=$(nvidia-smi -L | grep -c '^GPU')
export GPUS_PER_NODE

IP=$(hostname -I | awk '{print $1}')
PORT=6379
REDIS_PORT=6380
IP_HEAD=$IP:$PORT

export NGPUS=$GPUS_PER_NODE # Since this is only for one node
export REDIS_ADDR=$IP:$REDIS_PORT

# Start the redis server
redis-server --bind "$IP" --port "$REDIS_PORT" --protected-mode no --daemonize yes

# Start the Ray head node
ray start --head --node-ip-address="$IP" --port="$PORT" --block &
# Sleep for a few seconds to let the head node start
sleep 5

# Start the ray worker node on the same physical node as the head node (only 1 since this is for a single physical node ray cluster)
# Typically we want to have the head node and the worker nodes be on separate physical nodes
# This is for simplicity
ray start --address "$IP_HEAD" --include-log-monitor=False --resources="{\"worker_units\": $GPUS_PER_NODE}" --min-worker-port=10002 --max-worker-port=11000 --block &
# Sleep for a second to let the node connect
sleep 5

python3 ray_example_driver.py