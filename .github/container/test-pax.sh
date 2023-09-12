#!/bin/bash

# # Parse command-line arguments

print_var() {
    echo "$1: ${!1}"
}

usage() {
    echo "Test T5X throughput on a fake-data Wikipedia benchmark."
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "  OPTIONS                    DESCRIPTION"
    echo "  -a, --additional-args      Additional fiddle args to pass to paxml/main.py"
    echo "  -b, --batch-per-gpu        Batch size per GPU, defaults to 4."
    echo "  --dtype                    Batch size, defaults to bfloat16."
    echo "  --enable-te                If set, will run with env var ENABLE_TE=1." 
    echo "  -s, --steps                Number of steps to run, defaults to 500."
    echo "  --multiprocess             Enable the multiprocess GPU mode."
    echo "  -o, --output NAME          Name for the output folder, a temporary folder will be created if none specified."
    echo "  --data-parallel            Data parallelism to use. Defaults to 1."
    echo "  --tensor-parallel          Tensor parallelism to use. Defaults to 1."
    echo "  --pipeline-parallel        Pipeline parallelism to use. Defaults to 1 for no pipelining." 
    echo "  -n, --nodes                Number of nodes."
    echo "  -h, --help                 Print usage."
    exit $1
}

args=$(getopt -o a:b:s:o:n:h --long additional-args:,batch-per-gpu:,dtype:,enable-te,steps:,help,multiprocess,output:,data-parallel:,tensor-parallel:,pipeline-parallel:,nodes: -- "$@")
if [[ $? -ne 0 ]]; then
    exit $1
fi

# Default arguments
MULTIPROCESS=0
OUTPUT=$(mktemp -d)

BATCH_PER_GPU=4
DTYPE="bfloat16"
STEPS=500
DP=1
TP=1
PP=1
NODES=1
ENABLE_TE=0
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
            MULTIPROCESS=1
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
print_var OUTPUT
print_var MULTIPROCESS
print_var DP
print_var TP
print_var PP

## Enter Paxml source folder
### TODO: check this.. not quite sure what's happening here
## getting paxml path? I guess we are just using python import to find where paxml is located

## TODO: figure out how to disable checkpoint saving altogether!! 
## do not want to save checkpoint 0!
PAXML_DIR=$(dirname `python -c 'import paxml; print(*paxml.__path__)'`)
pushd ${PAXML_DIR}

## Create configs file
cat > ci_configs.py <<EOF
import math
from paxml import tasks_lib, experiment_registry
from paxml.contrib.gpu.scripts_gpu.configs import GPT126M, configure_gpt3_task
from paxml.tasks.lm.params.c4 import TransformerLmSpmdPipelineAdam
from paxml.tasks.lm.params.lm_cloud import SyntheticDataset
from praxis import base_layer
from praxis import layers

dp = ${DP}
tp = ${TP}
pp = ${PP}
num_gpus = ${NGPUS}
percore_batch_size = ${BATCH_PER_GPU}
steps = ${STEPS}
dtype = "${DTYPE}"

assert num_gpus == dp*tp*pp, f'product of parallel strategies should equal number of available gpus. Have {num_gpus} gpus, but product of parallel strategies is {dp*tp*pp}'

## heuristics to get ici and dcn mesh shapes.
## these heuristics only support one parallel strategy across nodes
## but should be sufficient for now
dcn_factor = math.ceil(num_gpus / 8)
dcn_dp = 1
dcn_pp = 1
if dcn_factor > 1:
  if dp % dcn_factor == 0:
    dcn_dp = dcn_factor
    dp = int(dp / dcn_factor)
  elif pp % dcn_factor == 0: 
    dcn_pp = dcn_factor
    pp = int(pp / dcn_factor)

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
    
    return task_p


if pp > 1:
  @experiment_registry.register
  class Synthetic126M(GPT126MPP, SyntheticDataset):
    
    ICI_MESH_SHAPE = [pp, dp, 1, tp]
    DCN_MESH_SHAPE = [dcn_pp, dcn_dp, 1, 1]
    MICROBATCH_SIZE = 2
    NUM_STAGES = pp
    PERCORE_BATCH_SIZE = percore_batch_size
    FRPOP_DTYPE = dtype
    MAX_STEPS = steps
    
    def task(self):
      task_p = super().task()
      return task_p

else:
  @experiment_registry.register
  class Synthetic126M(GPT126M, SyntheticDataset):
    
    ICI_MESH_SHAPE = [dp, 1, tp]
    DCN_MESH_SHAPE = [dcn_dp, 1, 1]
    PERCORE_BATCH_SIZE = percore_batch_size
    FRPOP_DTYPE = dtype
    MAX_STEPS = steps
    
    def task(self):
      task_p = super().task()
      return task_p

EOF

## Launch
set -ex
ENABLE_TE=$ENABLE_TE python -m paxml.main \
    --fdl_config=ci_configs.Synthetic126M \
    --job_log_dir=${OUTPUT} \
    --alsologtostderr \
    $ADDITIONAL_ARGS \
    $([[ $MULTIPROCESS != 0 ]] && echo --multiprocess_gpu)

set +x
echo "Output at ${OUTPUT}"
