# Imagen
[Imagen](https://arxiv.org/abs/2205.11487) is a text-to-image generative diffusion model that operates in pixel-space. This repository contains the necessary tools and scripts for performantly training Imagen from base model to its superresolution models in JAX on GPUs.

![A racoon wearing a hat and leather jacket in front of a backyard window. There are raindrops on the window.](assets/A%20raccoon%20wearing%20a%20hat%20and%20black%20leather%20jacket%20is%20behind%20the%20backyard%20window.%20Rain%20droplets%20on%20the%20window_16.png)
![A blue colored pizza](assets/A%20blue%20coloured%20pizza_14.png)
![mystical portal man](assets/a%20highly%20detailed%20digital%20painting%20of%20a%20portal%20in%20a%20mystic%20forest%20with%20many%20beautiful%20trees.%20A%20person%20is%20standing%20in%20front%20of%20the%20portal_20.png)

Prompts:
- A racoon wearning a hat and leather jacketin front of a backyard window. There are raindrops on the window
- A blue colored pizza
- a highly detailed digital painting of a portal in a mystic forest with many beautiful trees. A person is standing in front of the portal.

## Architecture
For maximum flexibility and low disk requirements, this repo supports a **distributed architecture** for text embedding in diffusion model training. Upon launching training, it will spawn LLM inference servers that will performantly calculate text embeddings online (with no latency hit). It does this by creating several inference **clients** in the diffusion model trainer's dataloaders, which send embedding requests to the inference servers. These servers are based on [NVIDIA PyTriton](https://github.com/triton-inference-server/pytriton), so execute all requests batched. Currently, this inference server supports T5x LLMs, but can be changed to be based on anything (doesn't even have to be JAX!) since the diffusion model trainer's client is simply making PyTriton (http) calls.

## GPU Scripts and Usage
We provide [scripts](scripts) to run [interactively](scripts/singlenode_inf_train.sh) or on [SLURM](scripts/example_slurm_inf_train.sub).

### Container
We provide a fully built and ready-to-use container here: `ghcr.io/nvidia/t5x:imagen-2023-10-02.v2`.

We do not currently have custom-built container workflows, but are actively working on supporting this, stay tuned for updates!
Imagen will also be available in our T5x container in future releases.

### Dataset
This model accepts webdataset-format datasets for training. For reference, we have an imagenet webdataset example [here](https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/vit#downloading-the-dataset).(NOTE: imagen is not directly compatible with imagenet). For imagen training with a compatible dataset, you can find or create your own webdataset (with image and text modalities).

Once you have your webdataset, update the dataset configs {[base](configs/img-txt-ds-base.gin), [sr1](configs/img-txt-ds-sr1.gin), [sr2](configs/img-txt-ds-sr2.gin)} with the paths to your dataset(s) under ```MIXTURE_OR_TASK_NAME```.


The 'img-txt-ds' configs assume a webdataset with a text and image modality. The images are in jpg format and the text is raw text in a ```'.txt'``` file. Currently, the configs are set up to do resolution-based filtering, scale-preserved square random cropping, and low-resolution image generation for SR model training. This can be changed (i.e. if you want your text in ```.json``` format and want to do additional processing) in the dataset configuration files {[base](configs/img-txt-ds-base.gin), [sr1](configs/img-txt-ds-sr1.gin), [sr2](configs/img-txt-ds-sr2.gin)}. 

### Downloading the LLM checkpoint
You will need to acquire the LLM checkpoint for T5 (for multimodal training) from T5x [here](https://t5x.readthedocs.io/en/latest/models.html#t5-1-1-checkpoints). All models use T51.1 format T5-xxl by default. Once you have the checkpoint, place it at ```rosetta/projects/inference_serving/checkpoints/checkpoint_1000000_t5_1_1_xxl``` (appending the ```_{size}``` to the checkpoint folder). **NOTE**: We're working on adding TransformerEngine support to the inference server, but for now, please run with the ```DISABLE_TE=True``` environment variable (example scripts include this).

### Running interactively
**Note**: this should only be done with singlenode jobs

```bash
CONTAINER=ghcr.io/nvidia/t5x:imagen-2023-10-02
docker run --rm --gpus=all -it --net=host --ipc=host -v ${PWD}:/opt/rosetta -v ${DATASET_PATH}:/mnt/datasets --privileged $CONTAINER bash
```

### Single Node runs
Pretraining can be done on multiple gpus within 1 host with `scripts/singlenode_inf_train.sh`. This will build an Imagen model with the Adam optimizer and relevant parameters. It will also launch the relevant LLM inference servers.

```bash
#### Pretraining (interactive: already inside container) with example args
bash rosetta/projects/imagen/scripts/singlenode_inf_train.sh {DATASET NAME} {MODEL NAME} {PRECISION} {NUM GPUS} {BSIZE/GPU} {LOGDIR} {MODEL DIR} {NUM LLM INFERENCE GPUS} {INFERENCE SERVER LLM SIZE}

#### Pretraining (non-interactive)
docker run --rm --gpus=all --net=host --ipc=host -v ${DATASET_PATH}:/mnt/datasets $CONTAINER bash rosetta/projects/imagen/scripts/singlenode_inf_train.sh {args from above}
```

### Multi Node runs
For a SLURM+pyxis cluster, the `scripts/example_slurm_inf_train.sub` file provides an example slurm submit file (edit with your details), which calls `scripts/multinode_train.sh` and `scripts/specialized_run.py` to execute training.

### Pretraining run commands
All commands below assume you are in `$ROSETTA_DIR=/opt/rosetta` and have the scripts and slurm scripts locally.

### Multinode
Arguments are set as such:
```sh
sbatch -N {NODE_CT} rosetta/projects/imagen/scripts/example_slurm_inf_train.sub \
{DATASET NAME} {MODEL NAME} {PRECISION} {NUM GPUS / NODE} {BSIZE/GPU} {MODEL DIR} {NUM LLM INFERENCE GPUS} {INFERENCE SERVER LLM SIZE}
```

All parameters can be found in the relevant script.

### Example training Commands
Assumes 8GPU 80GB A100/H100 Nodes.

#### Imagen-base small (500M):
```sh
sbatch -N 14 rosetta/projects/imagen/scripts/example_slurm_inf_train.sub \
{DATASET} imagen_base_500M bfloat16 8 32 runs/imagen-base 48 xxl
```

#### Imagen-base large (2B):
```sh
sbatch -N 20 rosetta/projects/imagen/scripts/example_slurm_inf_train.sub \
{DATASET} imagen_base_2B bfloat16 8 16 runs/imagen-base 32 xxl
```

#### Imagen-sr1 (efficient unet) (600M):
```sh
sbatch -N 14 rosetta/projects/imagen/scripts/example_slurm_inf_train.sub \
{DATASET} imagen_sr1_efficientunet_600M bfloat16 8 32 runs/imagen-sr1 48 xxl
```

#### Imagen-sr2 (efficient unet) (600M):
```sh
sbatch -N 14 rosetta/projects/imagen/scripts/example_slurm_inf_train.sub \
{DATASET} imagen_sr2_efficientunet_600M bfloat16 8 32 runs/imagen-sr2 48 xxl
```


### Sampling
You can find example sampling scripts that use the 500M base model and EfficientUnet SR models in [scripts](scripts). Prompts should be specified as in [example](../diffusion/tests/custom_eval_prompts/custom_eval_prompts.txt)

#### Sampling 256x256 images
Defaults to [imagen_256_sample.gin](configs/imagen_256_sample.gin) config (can be adjusted in script)
```
CUDA_VISIBLE_DEVICES=<DEVICES> CFG=5.0 GLOBAL_BATCH_SIZE=<GBS> GEN_PER_PROMPT=1 BASE_PATH=<BASE_CKPT> SR1_PATH=<SR1_CKPT> PROMPT_TEXT_FILES=<FILE> ./rosetta/projects/imagen/scripts/sample_imagen_256.sh 
```

Here is an example:
```
# Note:
#  - the quoting of double quotes wrapping single quotes is necessary.
#  - BASE_PATH/SR1_PATH are checkpoint dirs, and are expected to contain a `checkpoint` file, e.g., the file $BASE_PATH/checkpoint should exist
#  - GLOBAL_BATCH_SIZE should be set with number of GPUs in mind. For instance GLOBAL_BATCH_SIZE >= num gpus, 
#    to ensure at least one example is sent to each GPU.
#  - Currently there is a limitation where the number of lines in PROMPT_TEXT_FILES should be divisible by the number of GPUs.
#    The easiest way to ensure that is just to pad the files with dummy prompts until it is divisible
CUDA_VISIBLE_DEVICES=0,1 CFG=5.0 GLOBAL_BATCH_SIZE=4 GEN_PER_PROMPT=1 BASE_PATH='"/mnt/imagen_ckpt/checkpoint_585000"' SR1_PATH='"/mnt/sr1_ckpt/checkpoint_5000"' PROMPT_TEXT_FILES='"./rosetta/projects/diffusion/tests/custom_eval_prompts/custom_eval_prompts.txt"' ./rosetta/projects/imagen/scripts/sample_imagen_256.sh
```

#### Sampling 1024x1024 images
Defaults to [imagen_1024_sample.gin](configs/imagen_1024_sample.gin) config (can be adjusted in script).
```
CUDA_VISIBLE_DEVICES=<DEVICES> CFG=5.0 GLOBAL_BATCH_SIZE=<GBS> GEN_PER_PROMPT=1 BASE_PATH=<BASE_CKPT> SR1_PATH=<SR1_CKPT> SR2_PATH=<SR2_CKPT> PROMPT_TEXT_FILES=<FILE> ./rosetta/projects/imagen/scripts/sample_imagen_1024.sh 
```


## Convergence and Performance
Global Batch size = 2048. We assume 2.5B Training examples in these calculations. LLM Inference server nodes are not included in these numbers.
| size                    | GPU          | Precision | #GPUs | BS / GPU | Images/Sec | Im/Sec/GPU | Est. Walltime (hr) | GPU-days | Config                              | 
| ----------------------- | ------------ | --------- | ----- | -------- | ---------- | ---------- | ------------------ | -------- | ----------------------------------- |
| Imagen-base-500M        | A100-80G-SXM | BF16      | 8     | 64       | 858        | 107.0      | 809                | 269      | [cfg](configs/imagen_base_500M.gin) |
| Imagen-base-500M        | A100-80G-SXM | BF16      | 32    | 64       | 3056       | 95.5       | 227                | 303      | [cfg](configs/imagen_base_500M.gin) |
| Imagen-base-2B          | A100-80G-SXM | BF16      | 8     | 16       | 219        | 27.4       | 3170               | 1057     | [cfg](configs/imagen_base_2B.gin)   |
| Imagen-base-2B          | A100-80G-SXM | BF16      | 32    | 16       | 795        | 24.8       | 873                | 1164     | [cfg](configs/imagen_base_2B.gin)   |
| Imagen-base-2B          | A100-80G-SXM | BF16      | 128   | 16       | 2934       | 22.9       | 236                | 1258     | [cfg](configs/imagen_base_2B.gin)   |
| Imagen-SR1-600M-EffUNet | A100-80G-SXM | BF16      | 8     | 64       | 674        | 84.3       | 1030               | 343      | [cfg](configs/imagen_sr1_efficientunet_600M.gin) |   
| Imagen-SR1-600M-EffUNet | A100-80G-SXM | BF16      | 32    | 64       | 2529       | 79.1       | 274                | 365      | [cfg](configs/imagen_sr1_efficientunet_600M.gin) |
| Imagen-SR2-600M-EffUNet | A100-80G-SXM | BF16      | 8     | 64       | 678        | 84.8       | 1024               | 341      | [cfg](configs/imagen_sr2_efficientunet_600M.gin) |
| Imagen-SR2-600M-EffUNet | A100-80G-SXM | BF16      | 32    | 64       | 2601       | 81.3       | 267                | 356      | [cfg](configs/imagen_sr2_efficientunet_600M.gin) |
| Imagen-SR1-430M-UNet    | A100-80G-SXM | BF16      | 8     | 16       | 194        | 24.3       | 3580               | 1193     | [cfg](configs/imagen_sr1_unet_430M.gin) |

`Imagen-SR1-430M-UNet` is not currently supported. You can use the sr1-efficient-unet instead. Coming Soon!


Imagen base 500M + Efficient SR1 (600M):
|cfg|FID-30K (256x256)|
| - |-----------------------------------------------|
| 2 | 11.30 |
| 3 | 10.23 |
| 4 | 11.33 |
| 6 | 12.34 |

## Known Issues
* Currently, the nightly images will not be able to run Imagen since they lack a patch that needs refactoring. This will be released soon!
