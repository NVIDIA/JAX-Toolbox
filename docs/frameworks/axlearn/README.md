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
where `WORKSPACE_PATH` is the path to the directory where you would like to store any persistent files and `container` is the name of the AXLearn container. You can additionally add dataset and vocab paths with the `-v` flag.

## Example 1: standard training for a Fuji model on EKS
AXLearn based models are called `fuji`. The `fuji` models come with several number of parameters: `1B`, `3B`, `7B` and `70B`. For each model there's a `V1`, `V2` and `V3` version where:
- `V1` exists for `fuji-7B` and `fuji-70B`. It specifies a vocab size of `32 * 1024`, a max sequence length of `2048`, and a total of 1T tokens for the `7B` version and  1.4T for the `70B` one;
- `V2` exists for `fuji-7B` and `fuji-70B`. It specifies a vocab size of `32 * 1024`, a max sequence length of `4096`, and 2T tokens for both the `7B` and `70B` model;
- `V3` is is used for `1B`, `3B`, `7B` and `70B`. The vocab size is `128*1024`, the max sequence length `8192` and it provides 15T tokens for all the four models.
You can check the above on the  [AXLearn code](https://github.com/apple/axlearn/blob/main/axlearn/experiments/text/gpt/fuji.py).
Each model can then work in a different mode:
- `-flash`: uses flash attention;
- `-flash-single-host`: uses flash attention and it's tuned to work on a single host.
To run one of these models, on an EKS instance, you can [follow this deployment file](https://github.com/NVIDIA/JAX-Toolbox/blob/626d1a76da5ca1decfd9822f512849a2b5164cef/.github/eks-workflow-files/axlearn/axlearn-fuji-model.yml), with the running command depicted in the [c4_trainer description](https://github.com/apple/axlearn/blob/main/axlearn/experiments/text/gpt/c4_trainer.py), whose skeleton looks like:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
    name: axlearn-fuji
spec:
    completions: 1
    parallelism: 1
    template:
        spec:
            restartPolicy: Never
            containers:
                - name: axlearn-fuji-model
                  image: ghcr.io/nvidia/jax:axlearn
                  command:
                    - bash
                    - -xo
                    - pipefail
                    - -c
                    - |
                      BASEDIR="/opt/axlearn"
                      CONFIG="fuji-3B-v3-flash-single-host"

                      LOG_DIR=${BASEDIR}/logs
                      TRAINER_DIR=${LOG_DIR}/${CONFIG}-eks/trainer-dir
                      mkdir -p ${TRAINER_DIR}

                      python3 -m axlearn.common.launch_trainer_main \
                          --module=text.gpt.c4_trainer \
                          --config=${CONFIG} \
                          --trainer_dir=${TRAINER_DIR} \
                          --data_dir=gs://axlearn-public/tensorflow_datasets \
                          --jax_backend=gpu
```
This will run the `fuji-3B-v3-flash-single-host` model, and all the input configurations (e.g. max number of steps, sequence length, parallelism) can be found [here](https://github.com/apple/axlearn/blob/main/axlearn/experiments/testdata/axlearn.experiments.text.gpt.c4_trainer/fuji-3B-v3-flash-single-host.txt). The input dataset is the public tensorflow [C4 dataset](https://www.tensorflow.org/datasets/catalog/c4).

## Example 2: custom configuration training for a Fuji model on EKS
For specifying a custom configuration definition, we are using a [Python script](../../../.github/container/fuji-train-perf.py). The script is made based the following [AXLearn c4 trainer script](https://github.com/apple/axlearn/blob/main/axlearn/experiments/text/gpt/c4_trainer.py). The core configuration part is the following:
```python

# Build the model config
config_fn = c4_trainer.named_trainer_configs()[config_name]
trainer_config: SpmdTrainer.Config = config_for_function(config_fn).fn()
# Intra-node parallelism
ici_mesh_shape = mesh_shape_from_axes(
        pipeline=ici_pp_size, data=ici_dp_size, fsdp=ici_fsdp_size, seq=ici_sqp_size
)
# Inter-node parallelism
dcn_mesh_shape = mesh_shape_from_axes(
        pipeline=dcn_pp_size, data=dcn_dp_size, fsdp=dcn_fsdp_size, seq=dcn_sqp_size
)
# Create a mesh
mesh_shape = HybridMeshShape(
        ici_mesh_shape=ici_mesh_shape, dcn_mesh_shape=dcn_mesh_shape
)
# GA & FSDP setup
mesh_rule = (
"custom",
ChainConfigModifier.default_config().set(
        config_modifiers=[
        GradientAccumulationModifier.default_config().set(
                grad_acc_steps=ga_size
        ),
        MeshShapeModifier.default_config().set(mesh_shape=mesh_shape),
        ]
),
)
trainer_config.mesh_rules = mesh_rule
trainer_config.mesh_shape = mesh_shape
# Max step
trainer_config.max_step = max_step
# Checkpoint directory
trainer_config.dir = trainer_dir
trainer_config.input.input_dispatcher.global_logical_batch_size = gbs_size
trainer_config.input.source.max_sequence_length = seq_len
trainer_config.checkpointer.save_policy.n = save_checkpoint_steps
trainer_config.checkpointer.keep_every_n_steps = save_checkpoint_steps
trainer_config.summary_writer.write_every_n_steps = write_summary_steps
```
After parsing the input parameters, `config_fn = c4_trainer.named_trainer_configs()[config_name]` retrieves the standard configuration for the model specified in `config_name`. The parallelism is define intra-node and inter-node. [This function in AXLearn](https://github.com/apple/axlearn/blob/main/axlearn/common/utils.py#L1636) defines the construction of the mesh. Remember that for the intra-node mesh (`ici_mesh`), the parallelism nubmers product muber be as the same as the number of devices on a single node, while, for the inter-node mesh (`dcn_mesh`), the product must be equal to the number of nodes.  The rest of the code specifies the gradient accumulation size, the global batch size, the max sequence length, and when to save checkpoints and summary files.

Then, we're ready to launch the job with the following lines:
```python
launch.setup()
trainer_config.set(
        recorder=config_for_function(lambda: measurement.global_recorder)
)
# Launch training
launch_trainer.run_trainer(
trainer_config=trainer_config,
)
```
In particular, `launch.setup()` refers to [this code in AXLearn](https://github.com/apple/axlearn/blob/main/axlearn/common/launch.py), wher ether'es the main call to `jax` and its [distributed initialization](https://docs.jax.dev/en/latest/_autosummary/jax.distributed.initialize.html).
[Here](../../../.github/eks-workflow-files/axlearn/axlearn-fuji-model.yml) you can find an example of deployment to EKS, with the above script, that uses `fuji-3B-v3-flash` model.


## Testing
[Here is the YAML file](../../../.github/eks-workflow-files/axlearn/axlearn-job.yml) used for testing AXLearn funcitonalities. In particular, this test makes uses of [`test_axlearn.sh` script](../../../.github/container/test-axlearn.sh). The test runs `pytest` against all the tests contains in `/opt/axlearn/axlearn/common` folder.
