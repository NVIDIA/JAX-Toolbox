#!/bin/bash 

print_var() {
    echo "$1: ${!1}"
}

usage() {
    echo "Test ViT throughput on dummy data."
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "  OPTIONS                DESCRIPTION"
    echo "  -b, --batch-size       Global batch size (REQUIRED)"
    echo "  -d, --dtype            Data type, defaults to bfloat16."
    echo "  -t, --train-steps           Number of train steps to run, defaults to 500."
    echo "  --multiprocess         Enable the multiprocess GPU mode."
    echo "  -o, --output NAME      Name for the output folder, a temporary folder will be created if none specified."
    echo "  -h, --help             Print usage."
    exit $1
}

args=$(getopt -o b:d:t:o:h --long batch-size:,dtype:,train-steps:,help,multiprocess,output: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi

# Default arguments

BATCH_SIZE=0
DTYPE=bfloat16
TRAIN_STEPS=500
MULTIPROCESS=0
OUTPUT=$(mktemp -d)

eval set -- "$args"
while [ : ]; do
    case "$1" in
        -b | --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -d | --dtype)
            DTYPE="$2"
            shift 2
            ;;
        -t | --train-steps)
            TRAIN_STEPS="$2"
            shift 2
            ;;
        --multiprocess)
            MULTIPROCESS=1
            shift 1
            ;;
        -o | --output)
            OUTPUT="$2"
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

if [[ $BATCH_SIZE == 0 ]]; then
    echo "ERROR: Batch size must be specified."
    usage 1
fi

## Set derived variables

print_var BATCH_SIZE
print_var DTYPE
print_var TRAIN_STEPS
print_var NGPUS
print_var OUTPUT
print_var MULTIPROCESS
print_var TRAIN_STEPS

## install dependencies
pip install Pillow

## Create Python module to write dummy wds dataset
cat > generate_dummy_wds.py <<EOF
import argparse
from dataclasses import dataclass
import numpy as np
import os
import webdataset as wds

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='A script to generate the webdataset index files to be used during data loading.',
)
parser.add_argument('--output_tar_path', help='path to .tar files.')


@dataclass
class WebdatasetMetadata:
    num_examples: int = 20 * 4
    batch_size: int = 4
    image_size: int = 224
    channels: int = 3
    seq_len: int = 77
    image_key: str = 'jpg'
    text_key: str = 'txt'
    class_key: str = 'cls'
    num_classes: int = 10
    path: str | None = None


args = parser.parse_args()
tarball_path = args.output_tar_path

# HACK(terry): There is a bug in webdataset/writer.py that imports PIL, but not the module under it so we are doing it here as a WAR
#   https://github.com/webdataset/webdataset/issues/198
import PIL.Image  # noqa: F401
metadata = WebdatasetMetadata()
os.makedirs(tarball_path, exist_ok=True)
out_tar_path = os.path.join(tarball_path, 'dataset.tar')
print(f'output written to {out_tar_path}')
with wds.TarWriter(out_tar_path) as dst:
    for index in range(metadata.num_examples):
        dst.write({
            "__key__": f"sample{index:06d}",
            metadata.image_key: np.full((metadata.image_size, metadata.image_size, metadata.channels), fill_value=1.0/index if index else 0.0, dtype=np.float32),
            metadata.class_key: index % metadata.num_classes,
            metadata.text_key: f'A random image #{index}',
        })

EOF

set -exou pipefail

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}

DATA_PATH="/tmp/dummy_vit_data"
python -m generate_dummy_wds --output_tar_path=${DATA_PATH}

python -m t5x.train \
    --gin_file="/opt/rosetta/rosetta/projects/vit/configs/tests/small_pretrain_dummy.gin" \
    --gin.TRAIN_STEPS=${TRAIN_STEPS} \
    --gin.MIXTURE_OR_TASK_NAME=\"${DATA_PATH}/dataset.tar\" \
    --gin.MODEL_DIR=\"${OUTPUT}\" \
    --gin.DTYPE=\"${DTYPE}\" \
    --gin.BATCH_SIZE=${BATCH_SIZE} \
    --gin.train.stats_period=100 \
    --gin_search_paths=/opt/rosetta \
    --gin.CheckpointConfig.save=None \
    $([[ $MULTIPROCESS != 0 ]] && echo --multiprocess_gpu)
echo "Output at ${OUTPUT}"
