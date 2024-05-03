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
    echo "  --mem-fraction             Specify the percentage of memory to preallocate for XLA. Example: 0.90, 0.85, 0.65"
    echo "  --decoder-block            Specify decoder block to run. Example: llama2, default"
    echo "  --attn-type                Specify the attention type. Example: dot_product, cudnn_flash_te"
    echo "  --remat-policy             Specify remat policy. Example: minimal, minimal_flash, save_dot_except_mlp"
    echo "  -b, --batch-per-gpu        Batch size per GPU, defaults to 2."
    echo "  --dtype                    Batch size, defaults to bfloat16."
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

args=$(getopt -o a:b:s:o:n:h --long additional-args:,mem-fraction:,decoder-block:,attn-type:,remat-policy:,batch-per-gpu:,dtype:,steps:,help,multiprocess,output:,data-parallel:,fsdp:,tensor-parallel:,pipeline-parallel:,nodes: -- "$@")
if [[ $? -ne 0 ]]; then
    exit $1
fi

# Default arguments
HARDWARE='gpu'
OUTPUT=$(mktemp -d)
MEM_FRACTION=0.65

DECODER_BLOCK="default"
ATTN_TYPE="dot_product"
REMAT_POLICY="minimal"
BATCH_PER_GPU=2
DTYPE="bfloat16"
STEPS=10
DP=1
FSDP=1
TP=1
PP=1
NODES=1
ENABLE_FUSED_ATTN=0
ADDITIONAL_ARGS=""

eval set -- "$args"
while [ : ]; do
    case "$1" in
    -a | --additional-args)
        ADDITIONAL_ARGS="$2"
        shift 2
        ;;
    --mem-fraction)
        MEM_FRACTION="$2"
        shift 2
        ;;
    --decoder-block)
        DECODER_BLOCK="$2"
        shift 2
        ;;
    --attn-type)
        ATTN_TYPE="$2"
        shift 2
        ;;
    --remat-policy)
        REMAT_POLICY="$2"
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
        HARDWARE='gpu_multiprocess'
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
        shift
        break
        ;;
    *)
        echo "UNKNOWN OPTION $1"
        usage 1
        ;;
    esac
done

# # Set derived variables

GPUS_PER_NODE=$(nvidia-smi -L | grep -c '^GPU')
NGPUS=$((GPUS_PER_NODE * NODES))

# Heuristic to figure out ici and dcn of DP
# We only use DP across different nodes
if [ $NGPUS -gt 8 ]; then
    dcn_DP=$((NGPUS / 8))
    ici_DP=$((DP / dcn_DP))
else
    dcn_DP=1
    ici_DP=$DP
fi

if [ $ATTN_TYPE -eq 'cudnn_flash_te' ]; then
    ENABLE_FUSED_ATTN=1
    REMAT_POLICY="minimal_flash"
fi

print_var ADDITIONAL_ARGS
print_var MEM_FRACTION
print_var DECODER_BLOCK
print_var ATTN_TYPE
print_var REMAT_POLICY
print_var BATCH_PER_GPU
print_var DTYPE
print_var STEPS
print_var NGPUS
print_var HARDWARE
print_var OUTPUT
print_var ENABLE_FUSED_ATTN
print_var DP
print_var ici_DP
print_var dcn_DP
print_var FSDP
print_var TP
print_var PP

MAXTEXT_DIR="/opt/maxtext"
pushd ${MAXTEXT_DIR}

## Launch
set -ex

export NVTE_FUSED_ATTN=${ENABLE_FUSED_ATTN}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION}
export CUDA_DEVICE_MAX_CONNECTIONS=1

export BASE_XLA_FLAGS=${BASE_XLA_FLAGS:---xla_gpu_enable_latency_hiding_scheduler=true 
                --xla_gpu_enable_triton_gemm=false
                --xla_gpu_simplify_all_fp_conversions 
                --xla_gpu_graph_level=0 
                --xla_gpu_enable_highest_priority_async_stream=true
                --xla_gpu_all_reduce_combine_threshold_bytes=1073741824 
                --xla_gpu_all_gather_combine_threshold_bytes=1073741824 
                --xla_gpu_reduce_scatter_combine_threshold_bytes=1073741824
                --xla_gpu_enable_pipelined_all_gather=true 
                --xla_gpu_enable_pipelined_reduce_scatter=true 
                --xla_gpu_enable_pipelined_all_reduce=true 
                --xla_gpu_enable_while_loop_double_buffering=true
                --xla_gpu_enable_triton_softmax_fusion=false 
                --xla_gpu_enable_all_gather_combine_by_dim=false 
                --xla_gpu_enable_reduce_scatter_combine_by_dim=false 
                --xla_disable_hlo_passes=rematerialization}

export XLA_FLAGS="$BASE_XLA_FLAGS ${XLA_FLAGS:-}"

RUN_NAME="logdir" ## the RUN_NAME cannot be changed

RUN_SETTINGS="MaxText/train.py MaxText/configs/base.yml run_name=${RUN_NAME} logits_via_embedding=true decoder_block=${DECODER_BLOCK} \
    steps=$STEPS per_device_batch_size=${BATCH_PER_GPU} base_emb_dim=2560 base_mlp_dim=8192 remat_policy=${REMAT_POLICY}attention=${ATTN_TYPE}\
    base_num_query_heads=8 base_num_kv_heads=8 base_num_decoder_layers=8 head_dim=128 enable_checkpointing=false\
    base_output_directory=$OUTPUT dataset_path=local dataset_type=synthetic hardware=$HARDWARE\
    dcn_fsdp_parallelism=1 ici_fsdp_parallelism=$FSDP\
    ici_data_parallelism=$ici_DP dcn_data_parallelism=$dcn_DP\
    ici_tensor_parallelism=$TP dcn_tensor_parallelism=1 ${ADDITIONAL_ARGS}"

echo "Command: python3 $RUN_SETTINGS"
python3 $RUN_SETTINGS

set +x
echo "Output at ${OUTPUT}"
