set -x

export SCRIPT_DIR=/scripts

ulimit -n 1048576

NCCL_LIB_DIR=${NCCL_LIB_DIR} . /usr/local/nvidia/lib64/nccl-env-profile.sh

: "${NCCL_BENCHMARK:?Must set NCCL_BENCHMARK}"
NCCL_MINBYTES="${NCCL_MINBYTES:-8G}"
NCCL_MAXBYTES="${NCCL_MAXBYTES:-16G}"
NCCL_STEPFACTOR="${NCCL_STEPFACTOR:-2}"
NCCL_ITERS="${NCCL_ITERS:-100}"
NCCL_WARMUP_ITERS="${NCCL_WARMUP_ITERS:-0}"

run_nccl() {
  mpirun --mca btl tcp,self \
         --mca btl_tcp_if_include eth0 \
         --allow-run-as-root \
         -np $(( GPUS_PER_NODE * "${NHOSTS}" )) \
         --hostfile "${SCRIPT_DIR}/hostfiles${NHOSTS}/hostfile${GPUS_PER_NODE}"     \
         -x LD_LIBRARY_PATH \
         -x PATH     \
         -x NCCL_DEBUG=VERSION \
         -x NCCL_TESTS_SPLIT_MASK="${NCCL_TESTS_SPLIT_MASK:-0x0}"     \
         -x NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY="${NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY}"     \
         -x NCCL_LIB_DIR \
         -x NCCL_FASTRAK_IFNAME=${NCCL_FASTRAK_IFNAME} \
         -x NCCL_FASTRAK_CTRL_DEV="${NCCL_SOCKET_IFNAME}" \
         -x NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
         -x NCCL_CROSS_NIC=${NCCL_CROSS_NIC} \
         -x NCCL_ALGO=${NCCL_ALGO} \
         -x NCCL_PROTO=${NCCL_PROTO} \
         -x NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS} \
         -x NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE} \
         -x NCCL_P2P_PCI_CHUNKSIZE=${NCCL_P2P_PCI_CHUNKSIZE} \
         -x NCCL_P2P_NVL_CHUNKSIZE=${NCCL_P2P_NVL_CHUNKSIZE} \
         -x NCCL_FASTRAK_NUM_FLOWS=${NCCL_FASTRAK_NUM_FLOWS} \
         -x NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL=${NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL} \
         -x NCCL_BUFFSIZE=${NCCL_BUFFSIZE} \
         -x NCCL_FASTRAK_USE_SNAP=${NCCL_FASTRAK_USE_SNAP} \
         -x NCCL_FASTRAK_USE_LLCM=${NCCL_FASTRAK_USE_LLCM} \
         -x CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
         -x NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL} \
         -x NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=${NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING} \
         -x NCCL_TUNER_PLUGIN=${NCCL_TUNER_PLUGIN} \
         -x NCCL_TUNER_CONFIG_PATH=/usr/local/nvidia/lib64/a3plus_tuner_config.textproto \
         -x NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto \
         -x NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=${NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS} \
         -x NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE} \
         taskset \
         -c 32-63 \
         ${NCCL_BENCHMARK} --minbytes ${NCCL_MINBYTES} \
                           --maxbytes ${NCCL_MAXBYTES} \
                           --stepfactor ${NCCL_STEPFACTOR} \
                           --ngpus 1 \
                           --check 1 \
                           --warmup_iters ${NCCL_WARMUP_ITERS} \
                           --iters ${NCCL_ITERS} 2>&1 | \
         tee "/opt/output/${NCCL_BENCHMARK}_nh${NHOSTS}_ng${GPUS_PER_NODE}_i${NCCL_ITERS}.txt"
}

run_nccl "$@"
