# AXLearn
[AXLearn](https://github.com/apple/axlearn) is a deep learning design framework, built on top of JAX and XLA, to support the development of large-scale models. 


## Hardware and Software Specifications

The functionality have been validated on AWS p5.48xlarge EKS cluster (8x H100 80G). 


## Containers
We provide a multi-architecture container that is regularly updated. Use these containers to avoid dependency and environment issues. 
- Latest container: ghcr.io/nvidia/jax:axlearn
- Nightly dated container: ghcr.io/nvidia/jax:axlearn-YYYY-MM-DD

When you start an interactive session:

- Navigate to `/opt/axlearn` inside the container.
- Place your persistent files in a mounted directory (e.g. `/opt/axlearn/workspace`).

## Launching a container
Use the following command to launch a container:
```bash
docker run -ti --gpus=all --net=host --ipc=host -v <WORKSPACE_PATH>:/opt/axlearn/workspace -w /opt/axlearn <CONTAINER> /bin/bash
```
where `WORKSPACE_PATH` is the path to the directory where you would like to store any persistent files and `container` is the name of the maxtext container. You can additionally add dataset and vocab paths with the `-v` flag.

## Example: training `fuji-3B-v3-flash-single-host` on EKS
[Here is the YAML file](../../../.github/eks-workflow-files/axlearn/axlearn-fuji-model.yml) we're using for deploying the training of Fuji-3B model, that uses flash attention, and runs on a single host. The core part of the deployment is: 
```bash 
python3 -m axlearn.common.launch_trainer_main \
        --module=text.gpt.c4_trainer \
        --config=${CONFIG} \
        --trainer_dir=${TRAINER_DIR} \
        --data_dir=gs://axlearn-public/tensorflow_datasets \
        --jax_backend=gpu             
```
Where `CONFIG="fuji-3B-v3-flash-single-host`. The input dataset is the public tensorflow [C4 dataset](https://www.tensorflow.org/datasets/catalog/c4). 

## Testing
[Here is the YAML file](../../../.github/eks-workflow-files/axlearn/axlearn-job.yml) used for testing AXLearn funcitonalities. In particular, this test makes uses of [`test_axlearn.sh` script](../../../.github/container/test-axlearn.sh). The test runs `pytest` against all the tests contains in `/opt/axlearn/axlearn/common` folder.
