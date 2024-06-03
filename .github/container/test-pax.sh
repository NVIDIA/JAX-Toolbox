#!/bin/bash

# # Parse command-line arguments

print_var() {
    echo "$1: ${!1}"
}

usage() {
    echo "Test Pax throughput on a fake-data benchmark."
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "  OPTIONS                    DESCRIPTION"
    echo "  -a, --additional-args      Additional fiddle args to pass to paxml/main.py"
    echo "  -b, --batch-per-gpu        Batch size per GPU, defaults to 4."
    echo "  --dtype                    Batch size, defaults to bfloat16."
    echo "  --enable-te                If set, will run with env var ENABLE_TE=1." 
    echo "  --enable-dropout           If set, will set DROPOUT_PROB to 0.1."
    echo "  --enable-fused-attn        Whether to test fused attention through TE."
    echo "  --model-type               One of 126M, 5B, LLaMA70BProxy. Defaults to 126M"
    echo "  --evaluate                 Whether to test evaluation rather than training."
    echo "  -s, --steps                Number of steps to run, defaults to 500."
    echo "  --multiprocess             Enable the multiprocess GPU mode."
    echo "  --ici                      ICI mesh shape."
    echo "  --dcn                      DCN mesh shape."
    echo "  --enable-pipeline-parallel Whether to use pipeline parallelism."
    echo "  -o, --output NAME          Name for the output folder, a temporary folder will be created if none specified."
    echo "  -n, --nodes                Number of nodes."
    echo "  -h, --help                 Print usage."
    exit $1
}

args=$(getopt -o a:b:s:o:n:h --long additional-args:,batch-per-gpu:,dtype:,enable-te,enable-dropout,enable-fused-attn,enable-pipeline-parallel,model-type:,evaluate,steps:,help,multiprocess,output:,ici:,dcn:,nodes: -- "$@")
if [[ $? -ne 0 ]]; then
    exit $1
fi

# Default arguments
MULTIPROCESS=0
OUTPUT=$(mktemp -d)

BATCH_PER_GPU=4
DTYPE="bfloat16"
STEPS=500
ICI="[1,1,1]"
DCN="[1,1,1]"
NODES=1
ENABLE_TE=0
MODEL_TYPE=126M
NVTE_FUSED_ATTN=0
ENABLE_PP=0
DROPOUT=0
EVALUATE=0
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
        --enable-dropout)
            DROPOUT='0.1'
            shift 1
            ;;
        --enable-fused-attn)
            NVTE_FUSED_ATTN=1
            shift 1
            ;;
	--enable-pipeline-parallel)
	    ENABLE_PP=1
	    shift 1
	    ;;
        --model-type)
            MODEL_TYPE=$2
            shift 2
            ;;
        --evaluate)
            EVALUATE=1
            shift 1
            ;;
        -s | --steps)
            STEPS="$2"
            shift 2
            ;;
        --multiprocess)
            MULTIPROCESS=1
            shift 1
            ;;
        -o | --output)
            OUTPUT=$2
            shift 2
            ;;
        --ici)
            ICI=$2
            shift 2
            ;;
        --dcn)
            DCN=$2
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

print_var MODEL_TYPE
print_var BATCH_PER_GPU
print_var DTYPE
print_var STEPS
print_var NGPUS
print_var OUTPUT
print_var MULTIPROCESS
print_var ENABLE_TE
print_var NVTE_FUSED_ATTN
print_var EVALUATE
print_var DROPOUT
print_var ICI
print_var DCN

PAXML_DIR=$(dirname `python -c 'import paxml; print(*paxml.__path__)'`)
pushd ${PAXML_DIR}

## Create configs file
cat > ci_configs.py <<EOF
import math
import numpy as np
from paxml import tasks_lib, experiment_registry
from paxml.contrib.gpu.scripts_gpu.configs import (
    BaseLLaMA,
    Synthetic126M,
    configure_gpt3_task
)
from paxml.tasks.lm.params.c4 import TransformerLmSpmdPipelineAdam
from paxml.tasks.lm.params.lm_cloud import SyntheticDataset
from praxis import base_layer
from praxis import layers

num_gpus = ${NGPUS}
dropout = float(${DROPOUT})
dtype = "${DTYPE}"
pp = $ENABLE_PP

assert num_gpus == np.prod(${ICI}) * np.prod(${DCN}), f'product of parallel strategies should equal number of available gpus. Have {num_gpus} gpus, but product of parallel strategies is {np.prod(${ICI}) * np.prod(${DCN})}'

WeightInit = base_layer.WeightInit

class GPT126MPP(TransformerLmSpmdPipelineAdam):
  USE_REPEATED_LAYER = False
  ICI_MESH_SHAPE = [64,1,1]
  MAX_STEPS = 600000
  
  MAX_SEQ_LEN = 2048
  VOCAB_SIZE = 50304
  PACKED_INPUT = True
  PERCORE_BATCH_SIZE = 4
  
  NUM_LAYERS = 12
  NUM_HEADS = 12
  MODEL_DIMS = 768
  HIDDEN_DIMS = 3072
  DIMS_PER_HEAD = 64

  TRAINABLE_POSITION_EMB = True
  TRAINABLE_PE_MAX_SEQ_LEN = MAX_SEQ_LEN
  
  USE_BIAS = True
  LAYERNORM_EPSILON = 1e-5
  ATTEN_LOGIT_CAP = -1.0
  INIT_STD = 0.023
  SOFTMAX_INIT_STD = 0.023
  ACTIVATION_CLS = layers.GELU
    
  ## optimizer-related
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.95
  LEARNING_RATE = 6e-4
  ADAM_EPSILON_ROOT = 0.0
  ADAM_EPSILON = 1e-8
  WEIGHT_DECAY = 0.1
  ADAM_CLIP_THRESHOLD = -1.0
  CLIP_GRADIENT_NORM_TO_VALUE = 1.0

  ## lr schedule
  LR_SCHEDULE = 'linear_rampup_cosine_decay'
  LR_COS_WARMUP = 636
  LR_COS_DECAY_START = LR_COS_WARMUP+1
  LR_COS_DECAY_END = 500000
  R_COS_MIN_RATIO = 0.1
  LR_COS_MAX = 1.0

  ## dropout
  DROPOUT_PROB = dropout

  ## disable eval to avoid including eval
  ## in steps/sec calculation
  EVAL_INTERVAL_STEPS = 100000
  
  def task(self):
    task_p = super().task()
    task_p = configure_gpt3_task(self, task_p)

    task_p.train.num_train_steps = self.MAX_STEPS

    model_p = task_p.model
    
    ### compute layernorm reductions in fp32. Needed for stable training on GPUs
    stacked_p = model_p.lm_tpl.stacked_transformer_tpl
    if stacked_p.cls == layers.PipelinedTransformer:
      stacked_p = stacked_p.pipeline_stage
    if issubclass(stacked_p.cls, layers.StackedTransformerRepeated):
      stacked_p = stacked_p.block
    transformer_layer_p = stacked_p.transformer_layer_params_tpl
    transformer_layer_p.ln_tpl.reductions_in_fp32 = True
    transformer_layer_p.tr_fflayer_tpl.ln_tpl.reductions_in_fp32 = True
    task_p.model.lm_tpl.final_ln_tpl.reductions_in_fp32 = True
    
    model_p.params_init = WeightInit.Gaussian(self.INIT_STD)
    softmax_init = WeightInit.Gaussian(self.SOFTMAX_INIT_STD)
    model_p.lm_tpl.softmax_tpl.params_init = softmax_init
    
    model_p.apply_eval_sample_weights = True
    
    ## set input, residual, attention dropout to DROPOUT_PROB, remaining dropout to 0
    stacked_p.dropout_prob = 0.0
    stacked_p.input_dropout_prob = self.DROPOUT_PROB
    stacked_p.residual_dropout_prob = self.DROPOUT_PROB
    stacked_p.atten_dropout_prob = self.DROPOUT_PROB

    return task_p


### 1/20 of 70B model
class LLaMA70BSyntheticSmall(BaseLLaMA, SyntheticDataset):
    NUM_LAYERS = 4
    VOCAB_SIZE = 32000
    DIMS_PER_HEAD = 128
    NUM_HEADS = 64
    MODEL_DIMS = 8192
    HIDDEN_DIMS = 28672
    USE_MQA = True
    NUM_KV_HEADS = 8

    def task(self):
      task_p = super().task()
      task_p.train.always_use_train_for_model_init=False
      task_p.model.report_strict_acc=True
      return task_p


if pp == 1:
  @experiment_registry.register
  class Synthetic126MCI(GPT126MPP, SyntheticDataset):
    
    MICROBATCH_SIZE = 2
    NUM_STAGES = ${ICI}[0]
    FRPOP_DTYPE = dtype
    
    def task(self):
      task_p = super().task()
      task_p.train.always_use_train_for_model_init=False
      task_p.model.report_strict_acc=True
      return task_p

else:
  @experiment_registry.register
  class Synthetic126MCI(Synthetic126M):
    
    FRPOP_DTYPE = dtype

    DROPOUT_PROB = dropout

    ## disable eval
    EVAL_INTERVAL_STEPS = 100000
    
    def task(self):
      task_p = super().task()

      model_p = task_p.model
      stacked_p = model_p.lm_tpl.stacked_transformer_tpl
      if issubclass(stacked_p.cls, layers.StackedTransformerRepeated):
        stacked_p = stacked_p.block


      ## set input, residual, attention dropout to DROPOUT_PROB, remaining dropout to 0
      stacked_p.dropout_prob = 0.0
      stacked_p.input_dropout_prob = self.DROPOUT_PROB
      stacked_p.residual_dropout_prob = self.DROPOUT_PROB
      stacked_p.atten_dropout_prob = self.DROPOUT_PROB

      task_p.train.always_use_train_for_model_init=False
      task_p.model.report_strict_acc=True

      return task_p

EOF

## Launch
set -ex

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.65}
export ENABLE_TE=$ENABLE_TE
export NVTE_FUSED_ATTN=$NVTE_FUSED_ATTN
export VOCAB_PATH=${VOCAB_PATH:-gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model}

if [[ ${MODEL_TYPE} == "126M" ]]; then
  CONFIG=ci_configs.Synthetic126MCI
elif [[ ${MODEL_TYPE} == "5B" ]]; then
  CONFIG=paxml.contrib.gpu.scripts_gpu.configs.Synthetic5B
elif [[ ${MODEL_TYPE} == "LLaMA70BProxy" ]]; then
  CONFIG=ci_configs.LLaMA70BSyntheticSmall
## hard-code ICI mesh shape for Grok
elif [[ ${MODEL_TYPE} == "GrokProxy" ]]; then
  CONFIG=paxml.tasks.lm.params.nvidia.Grok_Proxy
  ADDITIONAL_ARGS+="--fdl.NUM_LAYERS=2"
else
  echo "Unsupported model ${MODEL_TYPE}"
  exit 1
fi

echo "HERE"

CMD_LINE_FLAGS="--fdl_config=${CONFIG} \
    --job_log_dir=${OUTPUT} \
    --alsologtostderr \
    --fdl.PERCORE_BATCH_SIZE=${BATCH_PER_GPU} \
    --fdl.ICI_MESH_SHAPE=${ICI} \
    --fdl.DCN_MESH_SHAPE=${DCN} \
    $ADDITIONAL_ARGS"
if [[ $MULTIPROCESS != 0 ]]; then
  CMD_LINE_FLAGS+=" --multiprocess_gpu"
fi

if [[ ${EVALUATE} -ne 0 ]]; then

  trap "rm -rf ${OUTPUT}/checkpoints" ERR INT HUP TERM EXIT

  ## train for 0 steps to generate an initial checkpoint
  python -m paxml.main ${CMD_LINE_FLAGS} --fdl.MAX_STEPS=0

  ## restore from initial checkpoint for eval
  python -m paxml.main ${CMD_LINE_FLAGS} --enable_checkpoint_saving=False --fdl.MAX_STEPS=0 --mode='eval'

else
  python -m paxml.main ${CMD_LINE_FLAGS} --enable_checkpoint_saving=False --fdl.MAX_STEPS=${STEPS}
fi

set +x
echo "Output at ${OUTPUT}"
