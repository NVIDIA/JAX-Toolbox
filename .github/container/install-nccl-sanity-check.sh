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
NCCL_SANITY_CHECK_ARGS=$(python3 -c 'import jax; import jax.distributed; jax.distributed.initialize(); lds = jax.local_devices(); assert(len(lds) == 1); from jax._src.distributed import global_state as gs; print(gs.num_processes, gs.process_id, lds[0].local_hardware_id, gs.coordinator_address)')
nccl-sanity-check $NCCL_SANITY_CHECK_ARGS
EOF
chmod +x "$BIN_DIR/$NAME.sh"
