#!/bin/bash

set -ex

BIN_DIR=/usr/local/bin
NAME=nccl-sanity-check

# Build binary from source
nvcc -o "$BIN_DIR/$NAME" "/opt/$NAME.cu" -lcudart -lnccl

# Create the wrapper script that queries jax for the configuration
cat <<"EOF" > "$BIN_DIR/$NAME.sh"
#!/bin/bash
set -e
export NCCL_SANITY_CHECK_LATENCY_US=1000
NCCL_SANITY_CHECK_ARGS=$(python3 -c 'import jax; jax.distributed.initialize(); from jax._src.distributed import global_state as gs; print(gs.process_id, gs.num_processes, gs.coordinator_address)')
nccl-sanity-check $NCCL_SANITY_CHECK_ARGS
EOF
chmod +x "$BIN_DIR/$NAME.sh"

# REMOVE ME: To test with base container (rather than jax-from-source container)
pip3 install jax
