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
    echo "  OPTIONS            DESCRIPTION"
    echo "  -b, --batch-per-gpu    Batch size per GPU, defaults to 32."
    echo "  -d, --dtype            Batch size, defaults to bfloat16."
    echo "  -e, --epochs           Number of epochs to run, defaults to 7."
    echo "  -g, --gpus             IDs of GPUs to use, e.g. '0,1,2,3'."
    echo "  -s, --steps-per-epoch  Number of steps per epoch, defaults to 100."
    echo "  -h, --help         Print usage."
    exit $1
}

args=$(getopt -o b:d:e:g:s:h --long batch-per-gpu:,dtype:,epochs:,gpus:,steps-per-epoch: -- "$@")
if [[ $? -ne 0 ]]; then
    exit $1
fi

# Default arguments

BATCH_PER_GPU=32
DTYPE=bfloat16
EPOCHS=7
GPUS="all"
STEPS_PER_EPOCH=100

eval set -- "$args"
while [ : ]; do
    case "$1" in
        -b | --batch-per-gpu)
            BATCH_PER_GPU="$2"
            shift 2
            ;;
        -d | --dtype)
            DTYPE="$2"
            shift 2
            ;;
        -e | --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -g | --gpus)
            GPUS="$2"
            shift 2
            ;;
        -s | --steps-per-epoch)
            STEPS_PER_EPOCH="$2"
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

## Set derived variables

if [[ $GPUS == "all" ]]; then
    NGPUS=$(nvidia-smi -L | grep -c '^GPU')
    GPUS=$(seq -s , 0 $((NGPUS - 1)))
else
    NGPUS=$(echo $GPUS | tr ',' '\n' | wc -l)
fi

BATCH_SIZE=$(($BATCH_PER_GPU * $NGPUS))
TRAIN_STEPS=$(($EPOCHS * $STEPS_PER_EPOCH))

print_var BATCH_PER_GPU
print_var BATCH_SIZE
print_var DTYPE
print_var EPOCHS
print_var GPUS
print_var NGPUS
print_var STEPS_PER_EPOCH
print_var TRAIN_STEPS

## Install T5 without the torch dependency
pip config set global.root-user-action ignore
pip config set global.disable-pip-version-check true
pip install babel sacrebleu scikit-learn rouge_score
pip install --no-deps git+https://github.com/google-research/text-to-text-transfer-transformer.git

## Enter T5X source folder
pushd $(dirname `python -c 'import t5x; print(*t5x.__path__)'`)

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
            vocabulary=t5.data.get_default_vocabulary(), add_eos=True, required=False
        ),
        targets=seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=True
        )
    ),
    metric_fns=[]
)
EOF

## Create GIN file
cat > benchmark.gin <<EOF
from __gin__ import dynamic_registration
from t5x import partitioning
from t5x.examples.t5 import network

include "t5x/examples/t5/t5_1_1/small.gin"
include 't5x/configs/runs/pretrain.gin'

# Register Dummy Wikipedia Seqio Task for benchmarking
import dummy_wikipedia

MIXTURE_OR_TASK_NAME = "dummy_wikipedia"
TASK_FEATURE_LENGTHS = {"inputs": 512, "targets": 114}
DROPOUT_RATE = 0.0
USE_CACHED_TASKS = False
TRAIN_STEPS = %gin.REQUIRED
BATCH_SIZE = %gin.REQUIRED

partitioning.PjitPartitioner:
    num_partitions = 1
EOF


## Launch
set -x
export CUDA_VISIBLE_DEVICES=${GPUS}
python t5x/train.py \
    --gin_file benchmark.gin \
    --gin.MODEL_DIR=\"./model\" \
    --gin.network.T5Config.dtype=\"${DTYPE}\" \
    --gin.TRAIN_STEPS=${TRAIN_STEPS} \
    --gin.BATCH_SIZE=${BATCH_SIZE} \
    --gin.train.eval_steps=0 \
    --gin.train.eval_period=${STEPS_PER_EPOCH} \
    --gin.CheckpointConfig.save=None
