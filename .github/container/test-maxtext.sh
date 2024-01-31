#!/bin/bash

# # Parse command-line arguments

print_var() {
    echo "$1: ${!1}"
}

usage() {
    echo "Test MaxText throughput on sythetic data."
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "  OPTIONS                    DESCRIPTION"
    echo "  -a, --additional-args      Additional fiddle args to pass to MaxText/train.py"
    echo "  -b, --batch-per-gpu        Batch size per GPU, defaults to 2."
    echo "  --dtype                    Batch size, defaults to bfloat16."
    echo "  --enable-te                If set, will run with env var ENABLE_TE=1."
    echo "  --enable-fused-attn        If set, will run with env var NVTE_FUSED_ATTN=1." 
    echo "  -s, --steps                Number of steps to run, defaults to 500."
    echo "  --multiprocess             Enable the multiprocess GPU mode."
    echo "  -o, --output NAME          Name for the output folder, a temporary folder will be created if none specified."
    echo "  --data-parallel            Data parallelism to use. Defaults to 1."
    echo "  --fsdp                     Fully-sharded data parallelism to use. Defaults to 1."
    echo "  --tensor-parallel          Tensor parallelism to use. Defaults to 1."
    echo "  --pipeline-parallel        Pipeline parallelism to use. Defaults to 1 for no pipelining."
    echo "  -n, --nodes                Number of nodes."
    echo "  -h, --help                 Print usage."
    exit $1
}

args=$(getopt -o a:b:s:o:n:h --long additional-args:,batch-per-gpu:,dtype:,enable-te,enable-fused-attn,steps:,help,multiprocess,output:,data-parallel:,fsdp:,tensor-parallel:,pipeline-parallel:,nodes: -- "$@")
if [[ $? -ne 0 ]]; then
    exit $1
fi

# Default arguments
MULTIPROCESS=false
OUTPUT=$(mktemp -d)

BATCH_PER_GPU=2
DTYPE="bfloat16"
STEPS=10
DP=1
FSDP=1
TP=1
PP=1
NODES=1
ENABLE_TE=0
ENABLE_FUSED_ATTN=0
ADDITIONAL_ARGS=""

eval set -- "$args"
while [ : ]; do
    case "$1" in
        -a | --additional-args)
            ADDITIONAL_ARGS="$2"
            shift 2
            ;;
        -b | --batch-per-gpu)
            BATCH_PER_GPU="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --enable-te)
            ENABLE_TE=1
            shift 1
            ;;
        -s | --steps)
            STEPS="$2"
            shift 2
            ;;
        --multiprocess)
            MULTIPROCESS=true
            shift 1
            ;;
        -o | --output)
            OUTPUT=$2
            shift 2
            ;;
        --data-parallel)
            DP="$2"
            shift 2
            ;;
        --fsdp)
            FSDP="$2"
            shift 2
            ;;
        --tensor-parallel)
            TP="$2"
            shift 2
            ;;
        --pipeline-parallel)
            PP="$2"
            shift 2
            ;;
        -n | --nodes)
            NODES="$2"
            shift 2
            ;;
        -h | --help)
            usage 1
            ;;
        --)
            shift;
            break 
            ;;
        *)
            echo "UNKNOWN OPTION $1"
            usage 1
    esac
done

# # Set derived variables

GPUS_PER_NODE=$(nvidia-smi -L | grep -c '^GPU')
NGPUS=$(( GPUS_PER_NODE * NODES ))

print_var BATCH_PER_GPU
print_var DTYPE
print_var STEPS
print_var NGPUS
print_var MULTIPROCESS
print_var OUTPUT
print_var ENABLE_TE
print_var ENABLE_FUSED_ATTN
print_var DP
print_var FSDP
print_var TP
print_var PP

MAXTEXT_DIR="/opt/maxtext"
pushd ${MAXTEXT_DIR}

## Launch
set -ex

export NVTE_FUSED_ATTN=${ENABLE_FUSED_ATTN}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
export CUDA_DEVICE_MAX_CONNECTIONS=1

export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false 
                  --xla_gpu_simplify_all_fp_conversions --xla_gpu_enable_async_all_gather=true
                  --xla_gpu_enable_async_reduce_scatter=true  --xla_gpu_enable_highest_priority_async_stream=true
                  --xla_gpu_enable_triton_softmax_fusion=false  --xla_gpu_all_reduce_combine_threshold_bytes=51200 
                  --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
                  --xla_gpu_graph_level=0 --xla_gpu_enable_async_all_reduce=true --xla_gpu_enable_pipelined_all_gather=true 
                  --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true --xla_disable_hlo_passes=rematerialization"

RUN_NAME="500M_PP${PP}_DP${DP}_FSDP${FSDP}_TP${TP}"

RUN_SETTINGS="MaxText/train.py MaxText/configs/base.yml run_name=${RUN_NAME}\
    steps=$STEPS per_device_batch_size=2 base_emb_dim=2560 base_mlp_dim=8192 remat_policy=minimal\
    base_num_query_heads=8 base_num_kv_heads=8 base_num_decoder_layers=8 head_dim=128 enable_checkpointing=false\
    base_output_directory=$OUTPUT dataset_path=local dataset_type=synthetic multiprocess_gpu=${MULTIPROCESS}\
    dcn_fsdp_parallelism=1 ici_fsdp_parallelism=$FSDP\
    ici_data_parallelism=$DP dcn_data_parallelism=1\
    ici_tensor_parallelism=$TP dcn_tensor_parallelism=1"


echo "Command: python3 $RUN_SETTINGS"
python3 $RUN_SETTINGS

set +x
echo "Output at ${OUTPUT}"
