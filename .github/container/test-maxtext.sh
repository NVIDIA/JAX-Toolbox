#!/bin/bash
set -exou pipefail

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
    echo "  -a, --additional-args      Additional args to pass to MaxText/train.py"
    echo "  --mem-fraction             Specify the percentage of memory to preallocate for XLA. Example: 0.90, 0.85, 0.65"
    echo "  --model-name               Specify the model names to run [Preffered]. If you specify model name then you do not need to specify decoder-block. Currently supported ootb models: 
                                       gemma-2b, gemma-7b, gpt3-175b, gpt3-22b, gpt3-52k, gpt3-6b, llama2-13b, llama2-70b, llama2-7b, llama3-70b, llama3-8b, mistral-7b, mixtral-8x7b" 
    echo "  --decoder-block            Specify decoder block to run. Example: llama2, default. Use this option only to define a custom model. This is not preferred, only used in CI"
    echo "  --attn-type                Specify the attention type. For gpt3-52k, we only use dot_product since the head_dim=8 is too small. Example: dot_product, cudnn_flash_te"
    echo "  --remat-policy             Specify remat policy. Example: minimal, minimal_flash, save_dot_except_mlp"
    echo "  -b, --batch-per-gpu        Batch size per GPU, defaults to 2."
    echo "  --dtype                    Data type, defaults to bfloat16. Example: bfloat16, fp8"
    echo "  -s, --steps                Number of steps to run, defaults to 500."
    echo "  --multiprocess             Enable the multiprocess GPU mode. Should be used when run on multinode"
    echo "  -o, --output NAME          Name for the output folder, a temporary folder will be created if none specified."
    echo "  --data-parallel            Data parallelism to use. Defaults to 1. If specified FSDP dims will be inferred."
    echo "  --fsdp                     Fully-sharded data parallelism to use. Defaults to 1. If none of the sharding specs are provided it will assume its FSDP across all available gpus."
    echo "  --tensor-parallel          Tensor parallelism to use. Defaults to 1. If specified, FSDP dims will be inferred."
    echo "  --pipeline-parallel        Pipeline parallelism to use. Defaults to 1 for no pipelining."
    echo "  -n, --nodes                Number of nodes."
    echo "  -h, --help                 Print usage. Some examples:  
                                       1. test-maxtext.sh -b 2 --model-name=gpt3-52k
                                       2. test-maxtext.sh -b 2 --model-name=gemma-2b --dtype=fp8
                                       3. test-maxtext.sh -n 1 -b 2 --model-name=llama2-7b --mem-fraction 0.90 --attn-type=cudnn_flash_te --remat-policy=minimal-flash --steps=10 --output train_output --multiprocess
                                       4. test-maxtext.sh -n 1 -b 2 --model-name=llama2-7b --mem-fraction 0.90 --attn-type=cudnn_flash_te --remat-policy=minimal-flash --dtype=fp8 --steps=10 --output train_output --multiprocess
                                       5. test-maxtext.sh -n 8 -b 2 --model-name=llama2-7b --mem-fraction 0.90 --attn-type=cudnn_flash_te --remat-policy=minimal-flash --steps=10 --output train_output --fsdp=8 --data-parallel=8 --multiprocess
                                       6. test-maxtext.sh -n 8 -b 2 --model-name=llama2-7b --mem-fraction 0.90 --attn-type=cudnn_flash_te --remat-policy=minimal-flash --steps=10 --output train_output --fsdp=4 --tensor-parallel=2 --data-parallel=8 --multiprocess
                                       7. test-maxtext.sh -n 16 -b 2 --model-name=llama2-70b --mem-fraction 0.90 --attn-type=cudnn_flash_te --remat-policy=save_dot_except_mlp --steps=10 --output train_output --fsdp=128 --multiprocess
                                       8. test-maxtext.sh -n 16 -b 2 --model-name=llama2-70b --mem-fraction 0.90 --attn-type=cudnn_flash_te --remat-policy=save_dot_except_mlp --steps=10 --output train_output --fsdp=64 --data-parallel=2 --multiprocess
                                       
                                       Note:
                                       a) FSDP and TP needs to defined for use; DP is not necessary to define, it will always be inferred from the other two.
                                       b) Multinode tests have to be launched with appropriate slurm commands i.e. sbatch and srun"
    exit $1
}

args=$(getopt -o a:b:s:o:n:h --long additional-args:,mem-fraction:,model-name:,decoder-block:,attn-type:,remat-policy:,batch-per-gpu:,dtype:,steps:,help,multiprocess,output:,data-parallel:,fsdp:,tensor-parallel:,pipeline-parallel:,nodes: -- "$@")
if [[ $? -ne 0 ]]; then
    exit $1
fi

# Default arguments
HARDWARE='gpu'
OUTPUT=$(mktemp -d)
MEM_FRACTION=0.65

MODEL="gpt3-52k"
DECODER_BLOCK=""
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
    --model-name)
        MODEL="$2"
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

# if not the gpt3-52k, we can use any any attention type such as cudnn_flash_te or dot_product
if [ $MODEL != "gpt3-52k" ]; then # gpt3-52k only works with dot_product
    ADDITIONAL_ARGS+=" attention=${ATTN_TYPE}"
fi

# for fp8 runs
if [ $DTYPE == "fp8" ]; then
    ADDITIONAL_ARGS+=" quantization=$DTYPE"
fi

GPUS_PER_NODE=$(nvidia-smi -L | grep -c '^GPU')
NGPUS=$((GPUS_PER_NODE * NODES))

# Heuristic to figure out ici and dcn of DP
# TP is always ici; after TP it will be FSDP and
# from TP and FSDP, we can find out ici and dcn DP
# in other words, DP dim across ici and dcn axis will always be inferred
ici_TP=${TP}
ici_DP=1
dcn_FSDP=1
if [ $((FSDP*TP)) -gt ${GPUS_PER_NODE} ]; then
    ici_FSDP=$((GPUS_PER_NODE/TP))
    dcn_FSDP=$((FSDP/ici_FSDP))
    dcn_DP=$((NGPUS/(ici_FSDP*ici_TP*ici_DP*dcn_FSDP)))
else
    ici_FSDP=$FSDP
    ici_DP=$((GPUS_PER_NODE/(FSDP*TP)))
    dcn_DP=$((NGPUS/(ici_FSDP*ici_TP*ici_DP*dcn_FSDP)))
fi

print_var ADDITIONAL_ARGS
print_var MEM_FRACTION
print_var MODEL
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
print_var ici_FSDP
print_var dcn_FSDP
print_var ici_TP
print_var PP

MAXTEXT_DIR="/opt/maxtext"
pushd ${MAXTEXT_DIR}

## Launch

export NVTE_FUSED_ATTN=${ENABLE_FUSED_ATTN}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION}
export CUDA_DEVICE_MAX_CONNECTIONS=1

export BASE_XLA_FLAGS=${BASE_XLA_FLAGS:---xla_gpu_enable_latency_hiding_scheduler=true 
                --xla_gpu_enable_triton_gemm=false
                --xla_gpu_graph_level=0 
                --xla_gpu_enable_highest_priority_async_stream=true
                --xla_gpu_all_reduce_combine_threshold_bytes=1073741824 
                --xla_gpu_all_gather_combine_threshold_bytes=1073741824 
                --xla_gpu_reduce_scatter_combine_threshold_bytes=134217728
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


if [ -z "$DECODER_BLOCK" ]; then

    # this part could be used to test different model ootb
    RUN_SETTINGS="MaxText/train.py MaxText/configs/base.yml run_name=${RUN_NAME} model_name=${MODEL}\
        steps=$STEPS per_device_batch_size=${BATCH_PER_GPU} remat_policy=${REMAT_POLICY} enable_checkpointing=false\
        base_output_directory=$OUTPUT dataset_path=local dataset_type=synthetic hardware=$HARDWARE\
        dcn_fsdp_parallelism=$dcn_FSDP ici_fsdp_parallelism=$ici_FSDP\
        ici_data_parallelism=$ici_DP dcn_data_parallelism=$dcn_DP\
        ici_tensor_parallelism=$ici_TP dcn_tensor_parallelism=1 ${ADDITIONAL_ARGS}"

else
    # this is essentially used for CI run
    RUN_SETTINGS="MaxText/train.py MaxText/configs/base.yml run_name=${RUN_NAME} logits_via_embedding=true decoder_block=${DECODER_BLOCK} \
        steps=$STEPS per_device_batch_size=${BATCH_PER_GPU} base_emb_dim=2560 base_mlp_dim=8192 remat_policy=${REMAT_POLICY} attention=${ATTN_TYPE}\
        base_num_query_heads=8 base_num_kv_heads=8 base_num_decoder_layers=8 head_dim=128 enable_checkpointing=false\
        base_output_directory=$OUTPUT dataset_path=local dataset_type=synthetic hardware=$HARDWARE\
        dcn_fsdp_parallelism=$dcn_FSDP ici_fsdp_parallelism=$ici_FSDP\
        ici_data_parallelism=$ici_DP dcn_data_parallelism=$dcn_DP\
        ici_tensor_parallelism=$ici_TP dcn_tensor_parallelism=1 ${ADDITIONAL_ARGS}"
fi

echo "Command: python3 $RUN_SETTINGS"
python3 $RUN_SETTINGS

echo "Output at ${OUTPUT}"