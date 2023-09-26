# Vision Transformer

This directory provides an implementation of the [Vision Transformer (ViT)](https://arxiv.org/pdf/2010.11929.pdf) model. This implementation is a direct adaptation of Google's [original ViT implementation](https://github.com/google-research/vision_transformer/tree/main). We have extended the original ViT implementation to include model parallel support. Model configurations are also based on the the original ViT implementation. Presently, convergence has been verified on ViT-B/16. Support for a wider range of models will be added in the future.

## Hardware Specifications
Convergence and performance has been validated on NVIDIA DGX A100 (8x A100 80G) nodes. Pretraining and fine-tuning of ViT/B-16 can be performed on a single DGX A100 80G node. We provide both singlenode and multinode support for pre-training and fine-tuning. If running on a machine with less than 80G memory, some of the default configurations may run out of memory; if you run out of memory and have more GPUs available, increase your GPU count and decrease your batch size per GPU. You may also use gradient accumulation to reduce the microbatch size while keeping the global batch size and GPU count constant, but note that gradient accumulation with ViT works only with scale-invariant optimizers such as Adam and Adafactor. See the [known issues](#Known-issues) section below for more information.

## Building a Container
We provide and fully built and ready-to-use container here: `ghcr.io/nvidia/rosetta-t5x:vit-2023-07-21`

If you do not plan on making changes to the Rosetta source code and would simply like to run experiments on top of Rosetta, we strongly recommend using the pre-built container. Run the following command to launch a container interactively: 
```
export CONTAINER=ghcr.io/nvidia/rosetta-t5x:vit-2023-07-21
docker run -ti --gpus=all --net=host --ipc=host -v <IMAGENET_PATH>:/opt/rosetta/datasets/imagenet -v <WORKSPACE_PATH>:/opt/rosetta/workspace -v <TRAIN_INDEX_PATH>:/opt/rosetta/train_idxs -v <EVAL_INDEX_PATH>:/opt/rosetta/eval_idxs --privileged $CONTAINER /bin/bash
```
where  `<IMAGENET_PATH>` is the path to the ImageNet-1k dataset (see the [downloading the dataset](#Downloading-the-dataset) section below for details) and ``<TRAIN_INDEX_PATH>`` and ``<EVAL_INDEX_PATH>`` refer to the paths to the train and eval indices for the ImageNet tar files (see the [before launching a run](#before-launching-a-run) section below for more information about these paths). ``<WORKSPACE_PATH>`` refers to the directory where you would like to store any persistent files. Any custom configurations or run scripts needed for your experiments should reside here.

If you plan on developing within the Rosetta repository, please see [DEVELOPMENT.md](../../../docs/DEVELOPMENT.md) for details on how to build the distribution.

## Downloading the Dataset
ViT is pre-trained and fine-tuned on the ImageNet-1k dataset. See below for details on how to download and preprocess this dataset.

*Note: It is the responsibility of each user to check the content
of the dataset, review the applicable licenses, and determine if it is suitable for their intended use.
Users should review any applicable links associated with the dataset before placing the data on their machine.*

Please note that according to the ImageNet terms and conditions, automated scripts for downloading the dataset are not
provided. Instead, kindly follow the steps outlined below to download and extract the data.

1. Create an account on [ImageNet](https://image-net.org/download-images) and navigate to ILSVRC 2012.
2. Download "Training images (Task 1 & 2)" and "Validation images (all tasks)" to `raw_data/imagenet_1k`.
Extract the training data:
```
cd raw_data/imagenet_1k
mkdir -p train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
```
3. Extract the validation data and move the images to subfolders:
```
mkdir -p val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
### exit the `raw_data` directory
cd ../../..
```
4. Download "Development kit (Tasks 1 & 2)" from [here](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and place it in the root data directory (`raw_data/imagenet_1k`).
5. Rosetta expects the dataset to be in WebDataset format. To convert the data to the appropriate format, first enter into the container. Be sure to mount the raw data to the container. Also, mount the location where you would like to store the dataset (referred to here as `<IMAGENET_PATH>`) to the container:
```
docker run -ti --gpus=all --net=host --ipc=host -v ${PWD}/raw_data/imagenet_1k:/opt/rosetta/raw_data/imagenet_1k -v <IMAGENET_PATH>:/opt/rosetta/datasets/imagenet --privileged $CONTAINER /bin/bash
```

6. Next, we will run [makeshards.py](https://github.com/webdataset/webdataset-lightning/blob/main/makeshards.py) to convert the data to WebDataset format. `makeshards.py` requires torchvision, which can be installed in the container using the following command:
```
pip install torchvision
```
7. Download `makeshards.py` and write the WebDataset shards to `datasets/imagenet`:
```
wget https://raw.githubusercontent.com/webdataset/webdataset-lightning/main/makeshards.py
python3 makeshards.py --shards datasets/imagenet --data raw_data/imagenet_1k
```

## Before Launching a Run
ViT uses [DALI](https://github.com/NVIDIA/DALI/tree/c4f105e1119ef887f037830a5551c04f9062bb74) on CPU for performant dataloading. Loading WebDataset tar files is done using DALI's [webdataset reader](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.readers.webdataset.html#nvidia.dali.fn.readers.webdataset). The reader expects each tar file to have a corresponding index file. These files can be generated using `rosetta/data/generate_wds_indices.py`. To generate the indices for the training data, first enter into the container, being sure to mount the locations where you would like to store the index files:
```
docker run -ti --gpus=all --net=host --ipc=host -v <IMAGENET_PATH>:/opt/rosetta/datasets/imagenet -v <TRAIN_INDEX_PATH>:/opt/rosetta/train_idxs -v <EVAL_INDEX_PATH>:/opt/rosetta/eval_idxs --privileged $CONTAINER /bin/bash
```
The train and eval indices will be saved to `<TRAIN_INDEX_PATH>` and `<EVAL_INDEX_PATH>`, respectively. Once inside of the container, run the following command. Note that braceexpand notation is used to specify the range of tar files to generate index files for.
```
python3 -m rosetta.data.generate_wds_indices --archive "/opt/rosetta/datasets/imagenet/imagenet-train-{000000..000146}.tar" --index_dir "/opt/rosetta/train_idxs"
```
Similarly, to generate indices for the validation dataset,
```
python3 -m rosetta.data.generate_wds_indices --archive "/opt/rosetta/datasets/imagenet/imagenet-val-{000000..000006}.tar" --index_dir "/opt/rosetta/eval_idxs"
```
When launching subsequent runs in the container, mount `<TRAIN_INDEX_PATH>` and `<EVAL_INDEX_PATH>` to the container to avoid having to regenerate the indices.

_Note_: This step is optional. If no indices are provided to the webdataset reader, they will be inferred automatically, but it typically takes around 10 minutes to infer the index files for the train dataset.

## Training Runs

### Pre-training
#### Single-process
Use the following command to launch a single-process pre-training run interactively from the top-level directory of the repository:
```
bash rosetta/projects/vit/scripts/singleprocess_pretrain.sh base bfloat16 <NUM GPUS> <BATCH SIZE PER GPU> <LOG DIR> <MODEL DIR LOCAL> <TRAIN INDEX DIR> <EVAL INDEX DIR> <GRADIENT ACCUMULATION>
```
`<MODEL DIR LOCAL>` refers to the _relative_ path to save checkpoints, summaries and configuration details to.

The following command can be used to launch a pre-training convergence run interactively on 8 GPUs:
```
bash rosetta/projects/vit/scripts/singleprocess_pretrain.sh base bfloat16 8 512 log_dir base_pretrain_dir /opt/rosetta/train_idxs /opt/rosetta/eval_idxs 1
```

#### Multi-process
See  `rosetta/projects/vit/scripts/example_slurm_pretrain.sub` for an example submit file that can be used to launch a multiprocess pre-training run with a SLURM + pyxis cluster. The following command can be used to launch a pre-training convergence run on a single node:
```
BASE_WORKSPACE_DIR=<PATH TO WORKSPACE> BASE_WDS_DATA_DIR=<PATH TO DATASET> BASE_TRAIN_IDX_DIR=<PATH TO TRAIN INDICES> BASE_EVAL_IDX_DIR=<PATH TO EVAL INDICES> VIT_SIZE=base PREC=bfloat16 GPUS_PER_NODE=8 BSIZE_PER_GPU=512 GRAD_ACCUM=1 MODEL_DIR_LOCAL=base_pretrain_dir sbatch -N 1 -A <ACCOUNT> -p <PARTITION> -J <JOBNAME> example_slurm_pretrain.sub
```
Here, `MODEL_DIR_LOCAL` is the directory to save checkpoints, summaries and configuration details to, relative to `BASE_WORKSPACE_DIR`.

### Pre-training to Fine-tuning
For improved fine-tuning accuracy, ViT pre-trains using a resolution of 224 and finetunes using a resolution of 384. Additionally, the classification heads used during pre-training and fine-tuning differ: the classification head consists of a two-layer MLP during pre-training and a single linear layer during fine-tuning. The script `rosetta/projects/vit/scripts/convert_t5x_pre-train_to_finetune_ckpt.py` converts the pre-trained checkpoint to a checkpoint that is compatible with the desired fine-tuning configuration. Run the following command to generate the checkpoint to be used during fine-tuning:
```
python3 -m rosetta.projects.vit.scripts.convert_t5x_pretrain_to_finetune_ckpt --gin_file rosetta/projects/vit/configs/base_convert_pt_to_ft.gin --gin.PT_CKPT_DIR='"<MODEL DIR FROM PRETRAINING RUN>"' --gin.FT_CKPT_DIR='"<WHERE TO SAVE CONVERTED CHECKPOINT>"'
```


### Fine-tuning
#### Single-process
Use the following command to launch a single-process fine-tuning run:
```
bash rosetta/projects/vit/scripts/singleprocess_finetune.sh base bfloat16 <NUM GPUS> <BATCH SIZE PER GPU> <LOG DIR> <MODEL DIR LOCAL> <TRAIN INDEX DIR> <EVAL INDEX DIR> <GRADIENT ACCUMULATION>
```
where `<MODEL DIR LOCAL>` corresponds to the directory containing the converted pre-training checkpoint.

#### Multi-process
See  `rosetta/projects/vit/scripts/example_slurm_finetune.sub` for an example submit file for launching a fine-tuning run with a SLURM + pyxis cluster. The following command can be used to launch a fine-tuning convergence run:
```
BASE_WORKSPACE_DIR=<PATH TO WORKSPACE> BASE_WDS_DATA_DIR=<PATH TO DATASET> BASE_TRAIN_IDX_DIR=<PATH TO TRAIN INDICES> BASE_EVAL_IDX_DIR=<PATH TO EVAL INDICES> VIT_SIZE=base PREC=bfloat16 GPUS_PER_NODE=8 BSIZE_PER_GPU=128 GRAD_ACCUM=1 MODEL_DIR_LOCAL=base_finetune_dir sbatch -N 1 -A <ACCOUNT> -p <PARTITION> -J <JOBNAME> example_slurm_finetune.sub
```

## Convergence Results
Pre-training was performed on 1 node with a global batch size of 4096. Models were trained on NVIDIA DGX A100 (8x A100 80G) nodes. Fine-tuning was performed on a single node with a global batch size of 512. Both pre-training and fine-tuning were performed using pure data parallel. Note that estimated walltime is calculated assuming full throughput (seq/sec) continuously. In practice, actual walltime might be greater due to potential compilation and checkpointing overheads.

| Size     | pre-training/fine-tuning | #GPUs | BS / GPU | Samples/sec | Estimated Walltime (hours) |  Imagenet Accuracy | Convergence Log |
| -------- | ----------------------- | ----- | -------- |  ----------------- | --------------- | ----------------- | --------------- |
| ViT/B-16 | pre-training           | 8     | 512 |  ~7129.27 | 15.01 |          N/A      |      [log](https://tensorboard.dev/experiment/wcDaxdo5T76I9YLapEbaCQ/)       |
| ViT/B-16 | fine-tuning           | 8     | 64     | ~1841.43 | 1.54 |         0.794      |     [log](https://tensorboard.dev/experiment/Bx1BVfeVTLmmNtxBNfxIWQ/)      |

## Pre-training Performance Results

### VIT-B/16
| Size     |Number of Parameters | #GPUs | BS / GPU | GBS |  Data Parallel | Model Parallel | Gradient Accumulation | Samples/Sec (A100) | Samples/Sec (H100) |  TFLOPs/sec (A100) | TFLOPs/sec (H100) |
| -------- | ------------------- | ----- | -------- | --- | -------------- |--------------- | --------------------- | ----------- | -------------------- | ----------- | ----------- |
|B/16 |    87M      | 8    |   512       |    4096       |    32       |  1       |    1       |           7200.38            |          11669.52           |        731.41           |       1266.95        |


### ViT-G/14

### Weak Scaling
| Size     |Number of Parameters | #GPUs | BS / GPU | GBS |  Data Parallel | Model Parallel | Samples/Sec/Core | TFLOPs/sec | Scaling Efficiency |
| -------- | ------------------- | ----- | -------- | --- | -------------- |--------------- | ----------- | ------- | ------ |
|G/14 |    1.85B      | 8    |     16       |    128       |    8       |   1       |          34.63           |   718.44  | 1 |
|G/14 |    1.85B      | 32    |     16       |    512       |    32       |   1       |            30.36          |   2519.30  | 0.88 |
|G/14 |    1.85B      | 64    |     16       |    1026       |    64       |   1       |          27.92           |   4634.10  | 0.81 |

### Model Parallel Scaling
| Size     |Number of Parameters | #GPUs | BS / GPU | GBS |  Data Parallel | Model Parallel | Gradient Accumulation | Samples/Sec | TFLOPs/sec |
| -------- | ------------------- | ----- | -------- | --- | -------------- |--------------- | --------------------- | ----------- | ------- |
|G/14 |    1.85B      | 32    |     16       |    4096       |    32       |   1       |    8       |          ~1281            |   3322.29  |
|G/14 |    1.85B      | 32    |     64       |    4096       |    8       |   4       |    8       |          ~795            | 2062.40  |
|G/14 |    1.85B      | 32    |    128       |    4096       |    4       |   8       |    8       |          ~553            |   1433.38  |

## Future Improvements
1. ViT currently does not support [Transformer Engine](https://github.com/NVIDIA/TransformerEngine). We plan to add Transformer Engine support to further accelerate pre-training and fine-tuning in the near future.

## Known Issues
1. By default, gradient accumulation (GA) sums loss across the microbatches. As a result, loss is scaled up when using gradient accumulation, and training with GA only works when using a scale-invariant optimizer such as Adam or Adafactor. ViT fine-tuning is performed using SGD; thus, GA should not be used when fine-tuning.
2. The nightlies disable `NCCL_NVLS_ENABLE=0` ([doc](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nvls-enable)). Future releases will re-enable this feature.

