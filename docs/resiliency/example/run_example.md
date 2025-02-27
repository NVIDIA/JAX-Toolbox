## Illustrative example

The example is a single node toy example that demonstrates how to run a simple training loop with a toy model, checkpoint along the way and recover upon encountering simulated failures and hangs. It is meant to be easy to run without needing access to a large compute cluster. 

## Building the image

Start by building the docker image using the Dockerfile in this directory.

```shell
nvidia-docker build -t ray_resiliency_example -f Dockerfile .
```

## Running the example

 After sshing into a node with at least 2 GPUs run the following command to get into the previously built container.

```shell
nvidia-docker run --name resilient_jax --network=host --security-opt seccomp=unconfined --cap-add SYS_PTRACE -it --shm-size=50g --ulimit memlock=-1 resilient_jax
```

Once inside the container run the `launch_ray_job.sh` script to run the example:

```shell
./launch_ray_job.sh
```
