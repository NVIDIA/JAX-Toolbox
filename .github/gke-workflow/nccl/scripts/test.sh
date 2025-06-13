set -x

: "${BENCHMARK:?Must set BENCHMARK}"
: "${NHOSTS:?Must set NHOSTS}"

ulimit -n 1048576

DATA_MIN="${DATA_MIN:-8G}"
DATA_MAX="${DATA_MAX:-8G}"
GPU_PER_NODE="${GPU_PER_NODE:-8}"
RUN_ITERS="${RUN_ITERS:-20}"
WARMUP_ITERS="${WARMUP_ITERS:-5}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

host_nic="eth0"
gpu_nics="eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8"
find_nics() {
    local nics_discovered=0
    local host_nic_discovered=""
    local gpu_nics_discovered=""
    for netdev in $(ls /sys/class/net | awk '{print $1}'); do
        pci_bdf=$(cat "/sys/class/net/${netdev}/device/uevent" 2>/dev/null | grep "PCI_SLOT_NAME" | awk -F '=' '{print $2}' || echo "")
        if [[ -z "${pci_bdf}" ]]; then
            continue
        fi
        if [[ "${pci_bdf}" =~ 00\:[0-9a-f]{2}\.0 ]]; then
            host_nic_discovered="${netdev}"
        elif [[ "" =~ [0|8][6-7d-e]\:00\.0 ]]; then
            if [[ -z "${gpu_nics_discovered}" ]]; then
                gpu_nics_discovered="${netdev}"
            else
                gpu_nics_discovered="${gpu_nics_discovered},${netdev}"
            fi
        fi
    done
    if [[ -n "" ]]; then
        host_nic="${host_nic_discovered}"
    fi
    if [[ -n "" ]]; then
        gpu_nics="${gpu_nics_discovered}"
    fi
}

find_nics

run_nccl() {
  mpirun --mca btl tcp,self \
         --mca btl_tcp_if_include eth0 \
         --allow-run-as-root \
         -np $(( GPU_PER_NODE * "${NHOSTS}" )) \
         --hostfile "${SCRIPT_DIR}/hostfiles${NHOSTS}/hostfile${GPU_PER_NODE}"     \
         -x NCCL_DEBUG_FILE="/tmp/${BENCHMARK}"-%h-%p.log     \
         -x NCCL_TOPO_DUMP_FILE="/tmp/${BENCHMARK}"_topo.txt     \
         -x NCCL_GRAPH_DUMP_FILE="/tmp/${BENCHMARK}"_graph.txt     \
         -x LD_LIBRARY_PATH \
         -x PATH     \
         -x NCCL_DEBUG=VERSION \
         -x NCCL_DEBUG_SUBSYS=INIT,NET,ENV,COLL,GRAPH \
         -x NCCL_TESTS_SPLIT_MASK="${NCCL_TESTS_SPLIT_MASK:-0x0}"     \
         -x NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY="${NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY}"     \
         -x NCCL_LIB_DIR \
         -x NCCL_FASTRAK_IFNAME="${gpu_nics}" \
         -x NCCL_FASTRAK_CTRL_DEV="${host_nic}" \
         -x NCCL_SOCKET_IFNAME="${host_nic}" \
         -x NCCL_CROSS_NIC=0 \
         -x NCCL_ALGO=Ring,Tree \
         -x NCCL_PROTO=Simple \
         -x NCCL_MIN_NCHANNELS=4 \
         -x NCCL_P2P_NET_CHUNKSIZE=524288 \
         -x NCCL_P2P_PCI_CHUNKSIZE=524288 \
         -x NCCL_P2P_NVL_CHUNKSIZE=1048576 \
         -x NCCL_FASTRAK_NUM_FLOWS=2 \
         -x NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL=0 \
         -x NCCL_BUFFSIZE=8388608 \
         -x NCCL_FASTRAK_USE_SNAP=1 \
         -x NCCL_FASTRAK_USE_LLCM=1 \
         -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
         -x NCCL_NET_GDR_LEVEL=PIX \
         -x NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=0 \
         -x NCCL_TUNER_PLUGIN=libnccl-tuner.so \
         -x NCCL_TUNER_CONFIG_PATH=/usr/local/nvidia/lib64/a3plus_tuner_config.textproto \
         -x NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto \
         -x NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000 \
         -x NCCL_NVLS_ENABLE=0 \
         taskset \
         -c 32-63 \
         "${BENCHMARK}" -b "${DATA_MIN}" -e "${DATA_MAX}" -f 2 -g 1 -w "${WARMUP_ITERS}" --iters "${RUN_ITERS}" 2>&1 | \
         tee "/tmp/${BENCHMARK}_nh${NHOSTS}_ng${GPU_PER_NODE}_i${RUN_ITERS}.txt"
}

run_nccl "$@"
