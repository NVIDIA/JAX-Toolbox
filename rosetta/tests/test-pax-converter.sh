#!/bin/bash 

print_var() {
    echo "$1: ${!1}"
}

usage() {
    echo "Test Pax checkpoint converter on a dummy checkpoint."
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "  OPTIONS                DESCRIPTION"
    echo "  --use-gqa              whether to test with GQA"
    echo "  -h, --help             Print usage."
    exit $1
}

args=$(getopt -o h --long use-gqa -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi

# Default arguments

USE_GQA=0

eval set -- "$args"
while [ : ]; do
    case "$1" in
        --use-gqa)
            USE_GQA=1
            shift 1
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

## Create configs file
cat > ci_configs.py <<EOF
import math
from paxml import tasks_lib, experiment_registry
from paxml.contrib.gpu.scripts_gpu.configs import (
    Synthetic126M,
)
from praxis import pax_fiddle
from praxis.layers import multi_query_attention

@experiment_registry.register
class GPT126MConvert(Synthetic126M):
  
  USE_MQA=False
  ICI_MESH_SHAPE=[8,1,1]
  NUM_KV_HEADS=4
  
  def task(self):
    task_p = super().task()
    if self.USE_MQA:
      transformer_layer_p = cast(
          pax_fiddle.Config[layers.Transformer],
          stacked_transformer_tpl.transformer_layer_params_tpl,
      )
      transformer_layer_p.tr_atten_tpl = pax_fiddle.Config(
          multi_query_attention.MultiQueryDotProductAttention,
          num_kv_heads=self.NUM_KV_HEADS,
      )
      transformer_layer_p.tr_atten_tpl.combine_qkv = False

EOF


## train for 0 steps to generate an initial checkpoint
ENABLE_TE=0 python -m paxml.main \
    --fdl_config=GPT126MConvert \
    --fdl.MAX_STEPS=0 \
    --fdl.USE_MQA=${USE_GQA} \
    --job_log_dir=dummy_pax_ckpt \
    --alsologtostderr
## run eval on dummy checkpoint
ENABLE_TE=0 python -m paxml.main \
    --fdl_config=GPT126MConvert \
    --job_log_dir=dummy_pax_ckpt \
    --mode='eval' \
    --fdl.MAX_STEPS=0 \
    --alsologtostderr \
    --fdl.USE_MQA=${USE_GQA}
    --enable_checkpoint_saving=False \
    
## convert the checkpoint
python -m rosetta.utils.te_pax_t5x_ckpt_converter.main.py \
    --input-path=dummy_pax_ckpt \
    --output-path=dummy_te_ckpt \
    --fw=pax \
    --direction=pax2te \
    --num-of-layer=12 \
    --num-of-head=12 \
    --head-dim=128 \
    --mlp-intermediate-dim=3072 \
    --$([[ $USE_GQA != 0 ]] && echo --pax-split-qkv --te-qkv-layout kv_packed --num-gqa-groups 4)

## run eval on converted checkpoint
ENABLE_TE=1 python -m paxml.main \
    --fdl_config=GPT126MConvert \
    --job_log_dir=dummy_te_ckpt \
    --mode='eval' \
    --fdl.MAX_STEPS=0 \
    --alsologtostderr \
    --fdl.USE_MQA=${USE_GQA}
    --enable_checkpoint_saving=False \