#!/bin/bash

## Parse command-line arguments

print_var() {
    echo "$1: ${!1}"
}

usage() {
    echo "Test T5X throughput on a fake-data Wikipedia benchmark."
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "  OPTIONS                   DESCRIPTION"
    echo "  -a, --additional-args     Additional gin args to pass to t5x/train.py"
    echo "  -b, --batch-size          Global batch size (REQUIRED)"
    echo "  -c, --use-contrib-configs If provided uses contrib/gpu configs instead of top-level configs. Notably, gpu configs use adamw instead of adafactor"
    echo "  -d, --dtype               Data type, defaults to bfloat16."
    echo "  --enable-te {0,1}         1 to enable, 0 to disable; defaults to ENABLE_TE in env or 0 if unset"
    echo "  -e, --epochs              Number of epochs to run, defaults to 7."
    echo "  --multiprocess            Enable the multiprocess GPU mode."
    echo "  -o, --output NAME         Name for the output folder, a temporary folder will be created if none specified."
    echo "  --save-hlo {0, 1}         1 to save the dumped hlo, 0 to remove the hlo dumped folder"
    echo "  --seed INT                Random seed for deterministim. Defaults to 42."
    echo "  -s, --steps-per-epoch INT Steps per epoch. Detauls to 100"
    echo "  --enable-fmha {0, 1}      1 to enable fmha testing, 0 to run test without fmha; default is 0"
    echo "  -h, --help                Print usage."
    exit $1
}

args=$(getopt -o a:b:cd:e:ho:s: --long additional-args:,batch-size:,use-contrib-configs,dtype:,enable-te:,enable-fmha:,epochs:,help,multiprocess,output:,seed:,save-hlo:,steps-per-epoch: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi

# Default arguments

ADDITIONAL_ARGS=""
BATCH_SIZE=0
USE_CONTRIB_CONFIGS=0
DTYPE=bfloat16
EPOCHS=7
MULTIPROCESS=0
OUTPUT=$(mktemp -d)
SEED=42
STEPS_PER_EPOCH=100
ENABLE_TE=${ENABLE_TE:-0}
ENABLE_FMHA=${ENABLE_FMHA:-0}
SAVE_HLO=${SAVE_HLO:-1}

eval set -- "$args"
while [ : ]; do
    case "$1" in
        -a | --additional-args)
            ADDITIONAL_ARGS="$2"
            shift 2
            ;;
        -b | --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -c | --use-contrib-configs)
            USE_CONTRIB_CONFIGS=1
            shift 1
            ;;
        -d | --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --enable-te)
            ENABLE_TE="$2"
            shift 2
            ;;
        --enable-fmha)
            ENABLE_FMHA="$2"
            shift 2
            ;;
        -e | --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -h | --help)
            usage 1
            ;;
        --multiprocess)
            MULTIPROCESS=1
            shift 1
            ;;
        -o | --output)
            OUTPUT="$2"
            shift 2
            ;;
        --save-hlo)
            SAVE_HLO="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -s | --steps-per-epoch)
            STEPS_PER_EPOCH="$2"
            shift 2
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

if [[ $BATCH_SIZE == 0 ]]; then
    echo "ERROR: Batch size must be specified."
    usage 1
fi

# Set hlo dump folder after output folder is set.
HLO_DIR=${OUTPUT}/hlo
export BASE_XLA_FLAGS="${BASE_XLA_FLAGS:---xla_dump_hlo_as_text --xla_dump_to=${HLO_DIR}}"
export XLA_FLAGS="${BASE_XLA_FLAGS} ${XLA_FLAGS:-}"
echo "HLO will be dumped in ${HLO_DIR} dir."

## Setting the env variables for FMHA
if [[ "$ENABLE_FMHA" -eq "1" ]]; then  
    echo "Setting XLA FMHA Flags";
    export BASE_XLA_FLAGS_FMHA="${BASE_XLA_FLAGS_FMHA:---xla_gpu_fused_attention_use_cudnn_rng=true --xla_gpu_enable_cudnn_fmha=true}"
    export XLA_FLAGS="${BASE_XLA_FLAGS_FMHA} ${XLA_FLAGS:-}"
fi

echo "XLA FLAGS: $XLA_FLAGS"
## Set derived variables

TRAIN_STEPS=$(($EPOCHS * $STEPS_PER_EPOCH))

print_var ADDITIONAL_ARGS
print_var BATCH_SIZE
print_var USE_CONTRIB_CONFIGS
print_var DTYPE
print_var ENABLE_TE
print_var ENABLE_FMHA
print_var EPOCHS
print_var OUTPUT
print_var MULTIPROCESS
print_var STEPS_PER_EPOCH
print_var TRAIN_STEPS
print_var SAVE_HLO

## Enter T5X source folder
T5X_DIR=$(dirname `python -c 'import t5x; print(*t5x.__path__)'`)
pushd ${T5X_DIR}

## Create Python module to define seqio data source
cat > dummy_wikipedia.py <<EOF
import functools
import seqio
import t5.data

seqio.TaskRegistry.add(
    "dummy_wikipedia",
    source=seqio.TfdsDataSource(tfds_name="wikipedia/20190301.als:1.0.0"),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5.data.preprocessors.unsupervised,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=dict(
        inputs=seqio.Feature(
            vocabulary=seqio.SentencePieceVocabulary(sentencepiece_model_file="gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model"), add_eos=True, required=False
        ),
        targets=seqio.Feature(
            vocabulary=seqio.SentencePieceVocabulary(sentencepiece_model_file="gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model"), add_eos=True
        )
    ),
    metric_fns=[]
)
EOF

## Create GIN file
cat > benchmark.gin <<EOF
from __gin__ import dynamic_registration
from t5x import partitioning
$(
  if [[ "$USE_CONTRIB_CONFIGS" -eq 0 ]]; then
    echo "from t5x.examples.t5 import network"
    echo "include 't5x/examples/t5/t5_1_1/small.gin'"
    echo "include 't5x/configs/runs/pretrain.gin'"
  else
    echo "from t5x.contrib.gpu.t5 import network"
    echo "include 't5x/contrib/gpu/t5/t5_1_1/small.gin'"
    echo "include 't5x/contrib/gpu/t5/configs/runs/pretrain.gin'"
    echo "include 't5x/contrib/gpu/t5/t5_1_1/adamw_opt.gin'"
  fi
)

# Register Dummy Wikipedia Seqio Task for benchmarking
import dummy_wikipedia

MIXTURE_OR_TASK_NAME = "dummy_wikipedia"
TASK_FEATURE_LENGTHS = {"inputs": 512, "targets": 128}
DROPOUT_RATE = 0.0
USE_CACHED_TASKS = False
TRAIN_STEPS = %gin.REQUIRED
BATCH_SIZE = %gin.REQUIRED

partitioning.PjitPartitioner:
    num_partitions = 1
EOF

## Launch
set -exou pipefail

ENABLE_TE=$ENABLE_TE python -m t5x.train \
    --gin_file benchmark.gin \
    --gin.MODEL_DIR=\"${OUTPUT}\" \
    --gin.network.T5Config.dtype=\"${DTYPE}\" \
    --gin.TRAIN_STEPS=${TRAIN_STEPS} \
    --gin.BATCH_SIZE=${BATCH_SIZE} \
    --gin.train.eval_steps=0 \
    --gin.train.eval_period=${STEPS_PER_EPOCH} \
    --gin.CheckpointConfig.save=None \
    --gin.train/utils.DatasetConfig.seed=${SEED} \
    --gin.train_eval/utils.DatasetConfig.seed=${SEED} \
    --gin.train.random_seed=${SEED} \
    $ADDITIONAL_ARGS \
    $([[ $MULTIPROCESS != 0 ]] && echo --multiprocess_gpu)
echo "Output at ${OUTPUT}"

if [[ "$ENABLE_FMHA" -eq "1" ]]; then 
    ## Check if fmha instructions are present in the HLO dumped file or not.
    fmha_regex="fmha[-bmm]?[-scale]?[-bias]?[-mask]?[-softmax]?[-dropout]?[-bmm]?[-backward]?*"
    result=$(grep -irlnE "$fmha_regex" "${HLO_DIR}/"*.txt)

    if [[ $SAVE_HLO -eq 0 ]]; then
        rm -rf $HLO_DIR
        echo "Removed dumped HLO directory!"
    fi

    if [ -z "$result" ]; then
        echo "E: No FMHA instructions were found in the hlo files!"
	exit 1
    else
        echo -e "Found FMHA instructions in the following HLO files: \n $result"
    fi
else
    if [[ $SAVE_HLO -eq 0 ]]; then
        rm -rf $HLO_DIR
 	echo "Removed dumped HLO directory!"
    fi
fi
