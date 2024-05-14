# PaliGemma

PaliGemma is a vision language model (VLM). These models are well-suited for a variety of tasks that require visually-situated text understanding and object localization. On a high level, the architecture follows a vision encoder (ViT) that encodes an image into tokens which together with text input (question, prompt, instruction), are passed to a decoder-only transformer (Gemma) that generates a text output.

For more details, refer to the [PaliGemma model card](https://ai.google.dev/gemma/docs/model_card) released by Google.

## Customizing PaliGemma with JAX

PaliGemma models are originally developed in JAX and released in Google's [Big Vision](https://github.com/google-research/big_vision) repository. In this repository, we provide an example for fine-tuning PaliGemma on NVIDIA GPUs. 

### Fine-Tuning

Full Fine-Tuning is the process of fine-tuning all of a model’s parameters on supervised data of inputs and outputs. It teaches the model how to follow user specified instructions and is typically done after model pre-training. 

Full fine-tuning is resource intensive so for this example to make it easy to run on a T4 colab runtime with 16GB HBM and 12GB RAM, we opt to only finetune the attention layers of the language model and freeze the other parameters.

This example will describe the steps involved in fine-tuning PaliGemma for generating image captions based on a training dataset of 90 pairs of images and long captions. 

[Get Started Here](./)

## Download the model and data

PaliGemma models are available on [Kaggle](https://www.kaggle.com/models/google/paligemma/) and in this notebook you can provide a Kaggle username and a Kaggle API key to download the model.

```python
kagglehub.model_download('google/paligemma/jax/pt_224', 'pt_224.f16.npz')
```

The tokenizer is available in a Google Cloud Storage bucket. You can install the Google Cloud CLI tool (gsutil) via pip.

```bash
gsutil cp gs://big_vision/paligemma_tokenizer.model ./paligemma_tokenizer.model
```

The dataset is also available in a Google Cloud Storage bucket.

```bash
gsutil -m -q cp -n -r gs://longcap100/ .
```

## Getting JAX Toolbox containers

JAX Toolbox containers include NVIDIA's latest performance optimizations in JAX and XLA. The containers are tested and validated against a nightly CI for performance regressions.

You can pull a container that includes JAX and all dependencies needed for this notebook with the following:

```bash
docker pull nvcr.io/nvidia/jax:gemma
```

The best way to run this notebook is from within the container. You can do that by launching the container with the following command

```bash
docker run --gpus all -it --rm -p 8888:8888 nvcr.io/nvidia/jax:gemma bash -c 'source /usr/local/nvm/nvm.sh && jupyter lab'
```