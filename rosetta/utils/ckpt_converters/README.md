# Bidirectional CKPT Converter for TE <-> T5X/Pax

### Arguments
```bash
-h, --help            show this help message and exit
--input-path INPUT_PATH
                        the path to load a source checkponint for this conversion. (Required)
--output-path OUTPUT_PATH
                    the path to store the converted checkponint. (Required)
--fw {pax,t5x}        the framework that stored the given source checkpoint. (Required)
--direction {fw2te,te2fw}
                    the direction of this conversion. (Required)
--num-of-layer NUM_OF_LAYER
                    the number of Transformer layer of the given source checkpoint. (Required)
--num-of-head NUM_OF_HEAD
                    the number of head of multi-head attention of the given source checkpoint. (Required)
--head-dim HEAD_DIM   the head dimension of multi-head attention of the given source checkpoint. (Required)
--mlp-intermediate-dim MLP_INTERMEDIATE_DIM
                    the intermediate dimension of MLP block (FFN) of the given source checkpoint. (Required)
--embed-dim EMBED_DIM
                    the embeded dimension of the given source checkpoint, must give if --fw=t5x. (default: None)
--kernel-chunk-size KERNEL_CHUNK_SIZE
                    the size to chucnk kernel (weighs) then store, only support with --fw=pax. Setting None means no chunking. (default: None)
--weight-only         indicate if the source checkpoint only includes weights. (default: False)
--skip-ln             indicate if skip the conversion for LayerNorm. (default: False)
--pax-repeat          indicate if the source Pax checkpoint enables Repeat. (default: False)
--t5x-fuse-qkv        indicate if the source T5X checkpoint enables fused_qkv_params of TE. (default: False)
```

### Usage Examples
#### Pax

1. TE -> Pax (Repeat):
```bash
python  converter/main.py \
    --input-path=/your_path_to_src_ckpt \
    --output-path=/your_path_to_output_ckpt \
    --fw=pax \
    --direction=te2fw \
    --pax-repeat \
    --num-of-layer=8 \
    --num-of-head=6 \
    --head-dim=64 \
    --mlp-intermediate-dim=1024
```

2. TE -> Pax (Not Repeat):
```bash
python  converter/main.py \
    --input-path=/your_path_to_src_ckpt \
    --output-path=/your_path_to_output_ckpt \
    --fw=pax \
    --direction=te2fw \
    --num-of-layer=8 \
    --num-of-head=6 \
    --head-dim=64 \
    --mlp-intermediate-dim=1024
```

3. Pax -> TE (Repeat):
```bash
python  converter/main.py \
    --input-path=/your_path_to_src_ckpt \
    --output-path=/your_path_to_output_ckpt \
    --fw=pax \
    --direction=fw2tw \
    --pax-repeat \
    --num-of-layer=8 \
    --num-of-head=6 \
    --head-dim=64 \
    --mlp-intermediate-dim=1024
```

2. Pax -> TE (Not Repeat):
```bash
python  converter/main.py \
    --input-path=/your_path_to_src_ckpt \
    --output-path=/your_path_to_output_ckpt \
    --fw=pax \
    --direction=fw2te \
    --num-of-layer=8 \
    --num-of-head=6 \
    --head-dim=64 \
    --mlp-intermediate-dim=1024
```

#### T5X
1. TE/FusedQKV -> T5X:
```bash
python  converter/main.py \
    --input-path=/your_path_to_src_ckpt \
    --output-path=/your_path_to_output_ckpt \
    --fw=t5x \
    --direction=te2fw \
    --t5x-fuse-qkv \
    --embed-dim=512 \
    --num-of-layer=8 \
    --num-of-head=6 \
    --head-dim=64 \
    --mlp-intermediate-dim=1024
```

2. TE/NotFusedQKV -> T5X:
```bash
python  converter/main.py \
    --input-path=/your_path_to_src_ckpt \
    --output-path=/your_path_to_output_ckpt \
    --fw=t5x \
    --direction=te2fw \
    --embed-dim=512 \
    --num-of-layer=8 \
    --num-of-head=6 \
    --head-dim=64 \
    --mlp-intermediate-dim=1024
```

3. T5X -> TE/FusedQKV:
```bash
python  converter/main.py \
    --input-path=/your_path_to_src_ckpt \
    --output-path=/your_path_to_output_ckpt \
    --fw=t5x \
    --direction=fw2te \
    --t5x-fuse-qkv \
    --embed-dim=512 \
    --num-of-layer=8 \
    --num-of-head=6 \
    --head-dim=64 \
    --mlp-intermediate-dim=1024
```

4. T5X -> TE/NotFusedQKV:
```bash
python  converter/main.py \
    --input-path=/your_path_to_src_ckpt \
    --output-path=/your_path_to_output_ckpt \
    --fw=t5x \
    --direction=fw2te \
    --embed-dim=512 \
    --num-of-layer=8 \
    --num-of-head=6 \
    --head-dim=64 \
    --mlp-intermediate-dim=1024
```

### Notes
#### Running converted CKPTs with Transformer Engine (TE) + FP8
If you run the converted TE checkpoints ,from frameworks Pax or T5X, with FP8 enabled, you might enounter
an error said that there is not FP8 meta found in the given checkpoint at restoring phase. That is because the
original checkpoints to convert do not contains information about FP8 meta. To address this issue, please run
a training process with the same model config on the target framework, plus TE and FP8, then store a checkpoint
at step 0. Next, use the converted checkpoint to replace weights of the checkpoint from famework + TE + FP8, and
restoring it to keep training.

#### The folder structure of CKPT by Pax and T5X
If you would like to run the converted CKPTs with frameworks, you may expect the converted CKPTs have the same folder
structure with CKPTs stored by frameworks. In this case, you could set `--output-path` to be the same stucture as the 
CKPTs from frameworks, and no need to pre-generate folders, since it would be generated when needed.
For Pax, you could set `--output-path` be like ` /${your_path_to_output}/checkpoints/checkpoint_${step}`.
For T5X, you could set `--output-path` be like `/${your_path_to_output}/checkpoint_${step}`.
