# Start a container:

```
srun  \
  --container-image=gitlab-master.nvidia.com:5005/dl/jax/jax-vllm-coupling:2025-08-27 \
  -p b100x4_preprod \
  --container-mounts=$HOME/workspace/:/opt/host --pty -t 8:00:00 /bin/bash
```

# How to run examples

> Please export HF_TOKEN and (optionally) KAGGLE_USERNAME and KAGGLE_KEY upon prompted.

From `/opt/jax-vllm-coupling/examples`:

## Llama 3 8B (default model), JAX:vLLM 2:4

```bash
bash coupling.sh
```

## Llama 3 70B, JAX:vLLM 4:4

```bash
N_GPUS_VLLM=4 N_GPUS_JAX=4 MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct bash coupling.sh
```

## Gemma 3 1B, JAX:vLLM 2:1

```bash
N_GPUS_VLLM=1 N_GPUS_JAX=2 MODEL_NAME=google/gemma-3-1b-it JAX_MODEL_NAME=google/gemma-3/flax JAX_MODEL_FLAVOR=gemma3-1b-it bash coupling.sh
```

## Running JAX and vLLM on different hosts

```bash
JAX_HOST=<host1> VLLM_HOST=<host2> N_GPUS_VLLM=<m> N_GPUS_JAX=<n> bash coupling_multihost.sh
```