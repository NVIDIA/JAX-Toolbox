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
