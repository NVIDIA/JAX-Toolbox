# Resilient Training with Ray and Jax

This guide demonstrates how resilient training of models written in libraries built on Jax can be achieved using Ray. More specifically, we will delve into the following:
- What a Ray cluster is and how to bring a Ray cluster up on underlying hardware infrastructure
- How to leverage the brought up Ray cluster to implement *fault tolerant* finetuning of a pretrained LLM; this includes:
    -  automatic recovery from failures and hangs by leveraging Ray's machinery for failure detection and recovery from a model checkpoint
    -  fast recovery from failures and hangs by leveraging model compilation caching

It assumes familiarity with Ray and Ray Clusters and requires a couple of extra packages these are Ray itself and redis-py: 

```console
python3 -m pip install "ray[default]" redis
```

The need for redis will be explained a little later in this guide.
### Why Ray?

Ray's runtime is based on a "head"/"worker" architecture where a "head" process called the coordinator, spawns "worker" processes called actors. This architecture
allows computation to be run across the workers and the coordinator to act as a watchdog and manage the control plane. As a result, the coordinator can detect any crash or hang experienced
by an actor and start the recovery process. This can be viewed as a single-controller runtime where the coordinator is the single-controller. We are going to build our resilient training system
using Ray's single-controller runtime, with the actors being responsible for training the model and the coordinator being responsible for failure detection and recovery. 

The workflow for implementing resilient training with Ray and JAX can be broadly split up into the following steps:
- Launching a [Ray cluster](https://docs.ray.io/en/latest/cluster/key-concepts.html#cluster-key-concepts)
- Implementing the coordinator and actors
- Launching a job with the coordinator and actors on the Ray cluster

We will delve into each of these steps in detail in that order in the rest of this guide.

## Starting the Ray Cluster

The first step to launching any workload using Ray is to start a Ray Cluster. The Ray cluster consists of a single "head node" and a number
of "worker nodes". There is a slight abuse of terminology that can be a little confusing here with the usage of the word "node" but both Ray
head nodes and Ray worker nodes can in general be thought of as sub-allocations within a physical node. This means that on a physical
node with 8 GPUs and 128 CPUs, a Ray worker node could pertain to a sub-allocation of up to 8 GPUs and 128 CPUs. And similarly for the Ray head node. The coordinator process
always runs on the Ray head node and actors always run on Ray worker nodes. In this guide we will assume that each actor gets 1 GPU, 16 CPUs and that we operate in a 1 process per GPU setting. 

### Starting a Ray Cluster manually

We will begin with a simple example of how to manually start a Ray cluster on 2 physical nodes. This will involve a single Ray head node and 2 Ray worker nodes, where each Ray worker node is allocated all GPUs of the node it runs on. We will assume the IP addresses of the physical nodes are `IP_ADDR_1` and `IP_ADDR_2` and that the head node will be allocated on the physical node with `IP_ADDR_1`. 

First, run the following script on one physical node:
```console
#!/bin/bash

# Number of GPUs per node
gpus_per_node=8
min_worker_port=10001
max_worker_port=10257

# On the physical node with `IP_ADDR_1`:
CUDA_VISIBLE_DEVICES="" ray start --node-ip-address=IP_ADDR_1 --port=<HEAD_NODE_PORT>
ray start --address "IP_ADDR_1:<HEAD_NODE_PORT>" \
                    --resources="{\"worker_units\": $gpus_per_node}" \
                    --min-worker-port=$min_worker_port \
                    --max-worker-port=$max_worker_port
```
The script above will allocate the Ray head node on the physical node with `IP_ADDR_1` and also start 1 ray worker node on the same physical node. We note the following about the script above:
- The address argument to `ray start` refers to the address of the Ray head node.
- We specify `CUDA_VISIBLE_DEVICES=""` when allocating the Ray head node. This is because the coordinator will not be performing any GPU computation and thus the Ray head node required no GPUs.
- Resources like GPUs, CPUs and memory are physically afforded to Ray worker nodes by the underlying physical node. But Ray allows users to define custom virtual resources to give users more finegrained control of actor scheduling. In this case we define a physical resource called a `worker_unit`. Since we are going to be assigning one process per GPU we give each ray worker node 8  units of the custom `worker_unit` resource. This allows us to "tag" the ray worker nodes that will be used as the primary workers for our finetuning job. To ensure that actors only ever gets scheduled on these ray worker nodes we will request a single worker_unit for each actor during scheduling.
 
At this point the Ray cluster has 1 Ray head node and 1 Ray worker node. We will allocate the second Ray worker node on the physical node with ip address `IP_ADDR_2` by running the following script:

```console
#!/bin/bash

# Number of GPUs per node
gpus_per_node=8
min_worker_port=10001
max_worker_port=10257

ray start --address "IP_ADDR_1:<HEAD_NODE_PORT>" \
                    --resources="{\"worker_units\": $gpus_per_node}" \
                    --min-worker-port=$min_worker_port \
                    --max-worker-port=$max_worker_port
```
Now the Ray cluster consists of 1 Ray head node and 2 Ray worker nodes, all allocated across the 2 physical nodes with IP addresses `IP_ADDR_1` and `IP_ADDR_2`.

### Starting a Ray Cluster with a SLURM allocation

While the example above is simple and illustrates how to quickly bringup a Ray cluster on two physical nodes, it is inefficient to manually run the scripts when scaling to many more nodes. Additionally, it is likely that physical resources are allocated to an application by a scheduler like SLURM or Kubernetes. In this guide we will assume that SLURM is the physical resource allocator. Below is a template of a SLURM batch script that sets up a ray cluster with the specification as above (1 Ray worker node per physical node / 8 GPUs per Ray worker node), but over `<NUM_NODES>` physical nodes.

```console
#!/bin/bash
#SBATCH --nodes=<NUM_NODES>+1
#SBATCH --exclusive
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<SLURM_PARTITION>
#SBATCH --time=<SLURM_ALLOCATION_TIME>

# Number of GPUs per node
gpus_per_node=8

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# Getting the node names and IP addresses in the SLURM allocation
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
ip_addresses_array=()

for node in $nodes; do
    ip_address=$(host $node | awk '/has address/ { print $4 }')
    # Add the IP address to the array
    ip_addresses_array+=("$ip_address")
done

head_node=${nodes_array[0]}
head_node_ip=${ip_addresses_array[0]}

port=<PORT>
ip_head=$head_node_ip:$port

# First we start the head of the ray cluster on one of the physical nodes
# In this case we are giving an entire physical node to the ray head node
# The ray head node is marked by including --head to the ray start command
srun --nodes=1 --ntasks=1 -w "$head_node" ray start --head \
                                                    --node-ip-address="$head_node_ip" \
                                                    --port=$port \
                                                    --block" &

# We now need to start the Ray worker nodes
# We define the following variables to help
num_ray_worker_nodes=$((SLURM_JOB_NUM_NODES - 1)) # One physical node is for the ray head node, rest are for ray worker nodes
export NUM_ACTORS=$((gpus_per_node * num_ray_worker_nodes)) # 8 actors per ray worker node (one for each GPU)

# Start Ray worker nodes
# We want 1 Ray worker node per physical node
# Worker nodes are started with ray start but without the --head flag
min_worker_port=10001
max_worker_port=10257
for ((i = 1; i <= num_ray_worker_nodes; i++)); do
    node_i=${nodes_array[$i]}
    
    srun --exact --nodes=1 --ntasks=1 --cpus-per-task=$((16 * gpus_per_node)) -w "$node_i" \
    ray start --address "$ip_head" \
              --resources="{\"worker_units\": gpus_per_node}" \
              --min-worker-port=$min_worker_port \
              --max-worker-port=$max_worker_port --block &
    sleep 3
  done
done

# At this stage the Ray cluster bringup has started on the physical nodes in the allocation
# Before we launch a job on this cluster we need to make sure that the bringup is complete
# We do so by querying the number of worker_units in the ray cluster and asserting = NUM_ACTORS
extract_worker_units() {
  status_output=$(srun --overlap --nodes=1 --ntasks=1 -w "$head_node" ray status)
  if echo "$status_output" | grep -q "worker_units"; then
    worker_units=$(echo "$status_output" | grep "worker_units" | awk -F'[/. ]' '{print $4}')
    echo $worker_units
  else
    echo 0
  fi
}

# Poll to make sure that all Ray worker nodes have connected to the head.
# All workers have connected when number of GPUs in ray cluster
# is equal to NUM_ACTORS. We use the utility function above
# to check how many GPUs have come online in the ray cluster
while true; do
  worker_units=$(extract_worker_units)
  if [ "$worker_units" -eq "$NUM_ACTORS" ]; then
    break
  fi
  sleep 2
done

echo "All workers connected!"

# We can now launch a job on this cluster
# We do so by launching a driver process on the physical node that the head node is on
# This driver process is responsible for launching a job on the Ray cluster
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" python3 launch_ray_cluster_job.py
```

## Implementing the coordinator and actors

Now that we have the ray cluster brought up, we can launch a job on this cluster. But before we do so we need to define the actions of the coordinator and the actors.

### Designing the coordinator

Let us start defining a class called `RayClusterCoordinator`, that encapsulates the functionality of the coordinator. We will start simple and add functionality as we go:

```python
# coordinator.py

import ray
import random
import asyncio
import redis
        
class RayClusterCoordinator:
  def __init__(self, worker_cls, num_workers) -> None:
    self.worker_cls = worker_cls
    self.num_workers = num_workers
  
    self.workers = [worker_cls.options(num_gpus=1, 
                                       num_cpus=16, 
                                       resources={"worker_units": 1}).remote() for _ in range(self.num_workers)]
   
    def initialize_workers(self, **kwargs):
        self.worker_init_kwargs = kwargs
        coordinator_ip = ray.get(self.workers[0].get_host_ip.remote())
        coordinator_port = random.randint(1, 100000)  % 2**12 + (65535 - 2**12 + 1)
        self.jax_coordinator_addr = f"{coordinator_ip}:{coordinator_port}"
        ray.get([w.initialize.remote(i, self.jax_coordinator_addr, self.num_workers, **kwargs) for i, w in enumerate(self.workers)])
        self.workers_initialized = True
    
    async def run(self, **kwargs):
        if not self.workers_initialized:
            raise Exception("""Cannot run workers without initializing them first. 
                               Please call the initialize_workers method of your cluster coordinator first.""")
        # Launch computation on actors...
    
```
The first job of the coordinator is to spawn actors and have them be scheduled on Ray worker nodes. This happens in the constructor of the `RayClusterCoordinator`. `worker_cls` is the class that encapsulates the functionality of each actor. Notice how we request 1 GPU, 16 CPUs and 1 worker_unit per actor as mentioned earlier. There are two additional mandatory methods that RayClusterCoordinator implements:
- `initialize_workers`: the purpose of this method is to trigger the initialize method of the `worker_cls` representing an actor
- `run`: the purpose of this method is to launch computation on all the scheduled actors. We haven't included the details of the run method quite yet, nor have we discussed why it is an async method. We will get to both after defining some of the functionality of the class representing an actor.

### Designing the actor

Our design of the actor involves defining two classes: `JaxWorker` and `ModelTrainer`. `JaxWorker` will define a high level interface for the role an actor plays in this system and `ModelTrainer` is a subclass of `JaxWorker` that will implement specific functionality to train a model in a failure resilient manner, including model checkpointing and compilation caching. Let's start with `JaxWorker`.

```python
# jaxworker.py

import jax
import socket
import os
import redis

class JaxWorker:
    def __init__(self):
        self.host_ip = socket.gethostbyname(socket.gethostname())
        self.logical_gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES'))
        
    def get_process_id(self):
        return self.process_id
    
    def get_host_ip(self):
        return self.host_ip
     
    def get_logical_gpu_id(self):
        return self.logical_gpu_id
        
    def initialize(self, process_id, coordinator_addr, num_processes):
        self.process_id = process_id
        jax.distributed.initialize(coordinator_address=coordinator_addr, num_processes=num_processes, process_id=self.process_id, local_device_ids=0)
    
    def run(self, *args, **kwargs):
        raise NotImplementedError
```

The JaxWorker interface defines a number of methods, the most important of which are the `initialize` and `run` methods:
- `initialize`: is responsible for initializing `jax.distributed` across all the spawned actors. Every class that inherits from JaxWorker can initialize any task specific state in it's initialize method, but must call the base class' initialize method to setup the jax.distributed runtime.
-  `run`: every class that inherits from JaxWorker must implement its own run method. This is the entry point to the computation that the actors will perform.

To understand how this class and inheritance structure works, let's start implementing the `ModelTrainer` class.

```python
# modeltrainer.py

import ray
import jax
import orbax.checkpoint as ocp
from optax._src.transform import ScaleByAdamState
from optax._src.base import EmptyState
from jaxworker import JaxWorker

@ray.remote
class ModelTrainer(JaxWorker):
    def __init__(self):
        super().__init__()
    
    def initialize(self, process_id, coordinator_addr, num_processes, **kwargs) -> None:
        super().initialize(process_id, coordinator_addr, num_processes)
        self.config = self._get_training_config(kwargs['training_config_dict'])
        self.ckpt_dir = kwargs['ckpt_dir']
        self.ckpt_freq = kwargs['ckpt_freq']
        ckpt_mgr_options = ocp.CheckpointManagerOptions(save_interval_steps=self.ckpt_freq, max_to_keep=5)
        self.ckpt_mgr = ocp.CheckpointManager(self.ckpt_dir, 
                                              item_names=('params', 'opt_state', 'stop_point'), 
                                              options=ckpt_mgr_options)
        jax.config.update("jax_compilation_cache_dir", kwargs['jax_compilation_cache'])
                                              
    def run(self, num_physical_nodes, restore=False) -> List[Any] | None:
        # Method implementing the training loop
```
In this guide, the obejctive of the actors is to finetune a pretrained LLM. Additionally, they have to do so in a *fault tolerant* manner. `ModelTrainer` is the class that encapsulates this behavior and inherits from `JaxWorker`. It's responsibilities are to finetune an LLM, initialize checkpointing and enable compilation caching to facilitate fast restarting. The class is decorated with `ray.remote` since instances of this class are to be scheduled as actors.

As a subclass of `JaxWorker`, `ModelTrainer` *must* implement its own `run` method. But before we look at the details of the `run` method 
let's look at the details of the overriden `initialize` method in `ModelTrainer`. It is *not* mandatory for subclasses of `JaxWorker` to override `initialize`, but it is often useful to initialize child class specific state by doing so. In addition to calling the base class' `initialize` method to initialize `jax.distributed`, `ModelTrainer` initializes the following in its `initialize` method:
1. **model checkpointing state**: initialization of an [Orbax](https://github.com/google/orbax) checkpoint manager. One of the important aspects of fault tolerant training is to be able to recover from the latest checkpoint. Hence our model is checkpointed at fixed intervals during the finetuning process
2. **jax compilation cache**: to avoid the penalty of recompilation when restoring the training run from the latest checkpoint after a crash or a hang

Fault tolerant training necessitates quick failure detection and quick recovery. Model checkpointing and compilation caching are particularly important steps towards that objective. We will now look at the details of the `run` method of `ModelTrainer` with the following caveat: we will merely gloss over parts of the `run` method that don't pertain directly to fault tolerant training, these include:
- Dataset creation and tokenizing
- Sharding model parameters
- Dataloading details
- Implementation of a single training step

The methods below provide signatures for some of these functions (that will be used in `run`):

```python
def _load_model_params(self):
    ### Load model parameters from a pretrained checkpoint ###
    ...
    ...
    return pretrained_checkpoint

def _shard_parameters(self, params, mesh):
    ### Shard parameters across mesh ###
    ...
    ...
    return sharded_parameters

def _shard_inputs(self, *inputs, mesh):
    ### Shard inputs across mesh ###
    ...
    ...
    return sharded_inputs

@staticmethod
def train_step(
    model, 
    params, 
    optimizer: optax.GradientTransformation, 
    opt_state: optax.OptState, 
    tokens: jax.Array, 
    input_mask: jax.Array,
) -> Tuple[jax.Array, ParameterPytree, optax.OptState]:
    ### Single Forward + Backward pass ###
```

Details on how to implement these aspects of model training can be found in many excellent guides online, one of which is [this](https://github.com/google-deepmind/gemma/blob/main/colabs/fine_tuning_tutorial.ipynb) tutorial on finetuning a [Gemma](https://github.com/google-deepmind/gemma/tree/main) model. We will instead focus on the aspects that demonstrate how Ray can be used to achieve fault tolerant training. With the helper methods defined as above, the `run` method looks as follows:

```python
def run(self, num_nodes, restore=False) -> List[Any] | None:
        # Init / load model parameters
        if not restore:
            # Load from pre-trained checkpoint
            model, params = self._load_model_params()
            start_epoch, start_iter, ckpt_step = 0, 0, 0
        else:
            # Load from checkpoint saved during training, includes saved optimizer state
            model, params, opt_state, stop_points = self.restore()
            start_epoch, start_iter, ckpt_step = stop_points["epoch"], stop_points["iter"], stop_points["ckpt_step"]

        # Shard model parameters
        dp_dim = num_nodes
        tp_dim = len(jax.devices()) // dp_dim
        devices = np.asarray(jax.devices()).reshape(dp_dim, tp_dim)
        mesh = Mesh(devices, ('DP', 'TP'))

        if not restore:
            params = self._shard_parameters(params, mesh)

        # Create data loader
        train_dataloader = DataLoader(...)

        # Compile the function that performs a single step of forward + backward
        jitted_train_step = jax.jit(ModelTrainer.train_step, static_argnames=['model','optimizer'])
        
        # Initialize optimizer
        optimizer = optax.adam(learning_rate=1e-3)
        if not restore:
            # First time initializing the optimizer state
            opt_state = optimizer.init(params)
        
        # Training loop
        for epoch in range(start_epoch, NUM_EPOCHS):
            for i, (tokens, mask) in enumerate(train_dataloader):
                # Skip computation till we reach the position
                # in the dataloader where the failure occurred
                if epoch == start_epoch and i <= start_iter:
                    continue
                

                global_tokens, global_mask = self._shard_inputs(tokens, mask, mesh)      
                train_loss, params, opt_state = jitted_train_step(model, 
                                                                  params, 
                                                                  optimizer, 
                                                                  opt_state, 
                                                                  global_tokens, 
                                                                  global_mask, 
                )
                self.checkpoint(params, opt_state, epoch, i, ckpt_step)             
```

Let's examine the details of the `run` method above. It takes two arguments `num_nodes` and `restore`. `num_nodes` is the number of physical nodes that all the actors are spawned across and is only used to create the mesh that computation is going to be spread out across. `restore` is a parameter that lets `run` know if `run` has been called during the recovery process after a failure or if it is at the start of a training job. If `restore` is false, the method looks very familiar to a standard training loop: a model is created, a data loader is created, parameters are sharded, the optimizer state is initialized and the model is optimized in the training loop with checkpointing being facilitated by the following method:

```python
def checkpoint(self, params, opt_state, epoch, iter, ckpt_step):
    self.ckpt_mgr.wait_until_finished()
    stop_point = {"epoch" : epoch, "iter" : iter, 'ckpt_step' : ckpt_step}
    opt_state_pytree = {"count" : opt_state[0].count, "mu" : opt_state[0].mu, "nu" : opt_state[0].nu}
    self.ckpt_mgr.save(ckpt_step, args=ocp.args.Composite(params=ocp.args.StandardSave(params), 
                                                  opt_state=ocp.args.StandardSave(opt_state_pytree), 
                                                  stop_point=ocp.args.JsonSave(stop_point))
    )

```
When `restore` is true, there are a number of differences. The first is that parameters are not loaded from the pretrained checkpoint, but are instead loaded from the latest saved checkpoint. The latest saved checkpoint also contains the latest optimizer state and some additional parameters (epoch, iteration anhd ckpt_step) that help pick up training from the point at which the failure took place. The `restore` method looks as follows:

```python
def restore(self):
        ckpt =  self.ckpt_mgr.restore(self.ckpt_mgr.latest_step(), 
                                      args=ocp.args.Composite(params=ocp.args.StandardRestore(), 
                                                              opt_state=ocp.args.StandardRestore(), 
                                                              stop_point=ocp.args.JsonRestore())
        )

        params, opt_state_pytree, stop_point = ckpt.params, ckpt.opt_state, ckpt.stop_point
        model = Model(...) # Flax implementation of the language model being finetuned
        opt_state = (ScaleByAdamState(count=opt_state_pytree["count"], mu=opt_state_pytree["mu"], nu=opt_state_pytree["nu"]), EmptyState())
        return model, params, opt_state, stop_point
```
So now that we have the `run` method for our actors in place, how exactly do we launch them? And more importantly, how do we detect and recover from failures? To do so, we have to go back to the coordinator and flesh out the `run` method of the `RayClusterCoordinator` class.

### Revisiting the coordinator

As mentioned earlier, the purpose of the `run` method of the `RayClusterCoordinator` is to launch computation on actors. Now that we have the computation that each actor is going to perform defined in the `run` method of `ModelTrainer`, we will have the `RayClusterCoordinator` call it to launch computation on the actors. Additionally, we will also include logic to detect failures and start recovery in the `run` method of the `RayClusterCoordinator`. The first version of this method looks as follows:

```python
# Run method of RayClusterCoordinator
async def run(self, **kwargs):
    if not self.workers_initialized:
        print(f"Cannot run workers without initializing them first. Please call the initialize_workers method of your cluster coordinator first.")
        return []

    task = asyncio.create_task(self._run_workers_async(**kwargs))
    while True:
        try:
            task_results = await asyncio.wait(task)
        except Exception as e:
            # Failure in one or more of the launched Ray actors, start recovery
            
            # Step 1: Kill the current task
            task.cancel()
            
            # Step 2: Kill all actors to free resources (all actors besides the failed actors will be in hanged state)
            for w in self.workers:
                ray.kill(w)
            self.workers_initialized = False

            # Step 3: Recreate actor handles and assign them to worker nodes
            self.workers = [self.worker_cls.options(num_gpus=1, num_cpus=16, resources={"worker_units": 1}).remote() 
                            for _ in range(self.num_workers)]
            
            # Step 4: Initialize the recreated actors and set the restore argument of the Actor run method to True
            self.initialize_workers(**self.worker_init_kwargs)
            kwargs["restore"] = True

            # Step 5: Restart task to launch actor computation
            task = asyncio.create_task(self._run_workers_async(**kwargs))
        else:
            return task_results[0]
```
Let's examine this method. The first detail to note (as mentioned when `RayClusterCoordinator` was introduced) is that the method is an async method. The reason for this is that we want the coordinator to be able to perform multiple non-blocking tasks. One of which is launching actors and being able detect when any of them fail. This is what the `_run_workers_async` method does:

```python
async def _run_workers_async(self, **kwargs):
    worker_run_futures = [w.run.remote(self.num_physical_nodes, **kwargs) for w in self.workers]
    while True:
        for i, wf in enumerate(worker_run_futures):
            try:
                ray.get(wf, timeout=0)
            except ray.exceptions.GetTimeoutError:
                pass

        # Give control back to the asyncio event loop
        await asyncio.sleep(30)
```
In this method each `w` is a reference to a `ModelTrainer` actor and `[w.run.remote(self.num_physical_nodes, **kwargs) for w in self.workers]` calls the `run` method of each actor to start the finetuning job. When a Ray actor fails, it sends a signal back to the coordinator that can be handled. This happens whether the failure is "graceful", in that the Actor is able to throw an exception; or if the failure is less "graceful" like a segmentation fault. Calling `ray.get(wf, timeout=0)` on any future from an actor that **has not** failed, will always result in a `ray.exceptions.GetTimeoutError`, if the actor's `run` method has not completed execution. However, if the actor has failed in any manner, graceful or not, `ray.get(wf, timeout=0)` will always throw one of `ray.exceptions.RayTaskError` or `ray.exception.ActorDiedError`, which propagates back to the coordinator. 

When the coordinator receives a failure signal from the `_run_workers_async` task, it can start the recovery process. This is what the code in the `except` block of the `run` method in `RayClusterCoordinator` does.  

### Adding hang detection capabilities

In its current state, this system can recover from crashes in any of the actors. But another important, and trickier class of failures to be able to detect and recover from is hangs. Hangs can occur for a variety of reasons during a run and so it's important for the system to be able to detect them in a general sense.

Being able to detect hangs requires more proactivity on the part of the coordinator, for the simple reason that if any actor is in a hanged state, it cannot send any signals to the coordinator to communicate that it is in such a state. So in addition to the crash detection logic outlined in `_run_workers_async` we will have the coordinator perform its second non-blocking task via another async function called `_detect_worker_hang_async`, into which we will add the logic to detect hangs. At a high level, the logic to detect hangs depends on the actors "checking in" periodically with the coordinator via a distributed key-value store. This is what we will use Redis for. To start a redis server on the same physical node as the Ray head node, we make the following change to the head node launch command in the launch script:

```console
# Script before looks the same...

srun --nodes=1 --ntasks=1 -w "$head_node" bash -c "redis-server --bind $head_node_ip --port 6380 --daemonize yes && \
                                                   ray start --head --node-ip-address="$head_node_ip" --port=$port --block" &

export REDIS_ADDR=$head_node_ip:6380

# Rest of the script looks the same...
```
With the redis server running we need to have both the coordinator and each actor connect to the server, which we can do by adding the following two lines to the constructor of `RayClusterCoordinator` and `JaxWorker`:

```python
self.redis_addr = os.environ.get('REDIS_ADDR').split(':')
self.redis = redis.Redis(host=self.redis_addr[0], port=int(self.redis_addr[1]), decode_responses=True, password=None)
```

As mentioned, the the hang detection logic we are yet to implement in `_detect_worker_hang_async` depends on actors "checking in" with the coordinator. This is facilitated through a heartbeat in the form of a timestamp from each actor being written to the redis KV store. They key is the process ID and the value is the timestamp of the last time the actor was alive. This is implemented through a context manager that we add to the `JaxWorker` class:

```python
@contextmanager
def EnableHeartbeat(self):
    try:
        yield
    finally:
        self.redis.set(self.process_id, time.time())
```
and then wrapping the computation in the the main training loop in the `run` method of `ModelTrainer` with this context manager:

```python
def run(self, num_nodes, restore=False) -> List[Any] | None:
    
    # All the code before the training loop is the same ...

    for epoch in range(start_epoch, NUM_EPOCHS):
        for i, (tokens, mask) in enumerate(train_dataloader):
            if epoch == start_epoch and i <= start_iter:
                continue

            with self.EnableHeartbeat():
                global_tokens, global_mask = self._shard_inputs(tokens, mask, mesh)      
                train_loss, params, opt_state = jitted_train_step(model, 
                                                                  params, 
                                                                  optimizer, 
                                                                  opt_state, 
                                                                  global_tokens, 
                                                                  global_mask, 
                )
                self.checkpoint(params, opt_state, epoch, i, ckpt_step)   
```
That's it for the actors. Now we can implement `_detect_worker_hang_async` to complete the hang detection feature. The logic here is straightforward, wait a certain amount of time `t` seconds and then check the timestamp of each actor's heartbeat in the redis KV store. If an actor hasn't checked in in `t` seconds, it's likely hanged. It looks as follows:

```python
async def _detect_worker_hang_async(self):       
    # Check if processes are hanging
    while True:
        # Wait for 5 minutes
        await asyncio.sleep(300)
        for pid in range(self.num_workers):
            elapsed = time.time() - float(self.redis.get(pid))
            if elapsed > 300:
                raise Exception(f"Worker {self._worker_process_ids[pid]} appears to have hanged")
```
A couple of points to note about the method above:
- The first is that the `asyncio.sleep` serves two purposes. One if to wait for `t` as part of the hang detection logic. The other is to give control back to the asyncio event loop so that it can switch contexts to the `_run_workers_async` method to check if any actors have failed.
- The chosen time of `t=300` does mean that it will take at least 5 minutes to detect a hang after its occurence. This is the best case and occurs if and only if an actor hangs at the exact moment the `asyncio.sleep(300)` command is issued. In the worst case, the time from occurrence to detection is 10 minutes and happen if the hang occurs at the exact moment the `asyncio.sleep(300)` command finishes. However, `t` is configurable and the chosen value here is just meant to illustrate functionality. 

With `_detect_worker_hang_async` implemented, we just need to change the `run` method of `RayClusterCoordinator` to launch the task. The final updated method looks as follows:

```python
async def run(self, **kwargs):
    if not self.workers_initialized:
        print(f"Cannot run workers without initializing them first. Please call the initialize_workers method of your cluster coordinator first.")
        return []

    tasks = [asyncio.create_task(self._run_workers_async(**kwargs)), 
             asyncio.create_task(self._detect_worker_hang_async())]
    while True:
        try:
            task_results = await asyncio.gather(*tasks)
        except Exception as e:
        
            for task in tasks:
                task.cancel()

            for w in self.workers:
                ray.kill(w)
            self.workers_initialized = False

            self.workers = [self.worker_cls.options(num_gpus=1, num_cpus=16, resources={"worker_units": 1}).remote(self.log_dir) 
                            for _ in range(self.num_workers)]
            self.initialize_workers(**self.worker_init_kwargs)
            kwargs["restore"] = True

            tasks = [asyncio.create_task(self._run_workers_async(**kwargs)), 
                     asyncio.create_task(self._detect_worker_hang_async())]
        else:
            return task_results[0]

```

When the coordinator determines an actor has hanged, it raises an exception that is handled exactly like a crash is handled, except during the recovery process it needs to kill two tasks instead of one.

## Launching a job Ray job

With that we've described every important aspect of the coordinator and the actors that help them work together to achieve failure resilient training. The final piece of the puzzle is to launch the training job that leverages all the logic implemented in `RayClusterCoordinator`, `JaxWorker` and `ModelTrainer`, on the Ray cluster. This is achieved through the following two scripts:

```python
# main.py

import os
import ray
from coordinator import RayClusterCoordinator
from modeltrainer import ModelTrainer

ray.init(address='auto')
num_workers = int(os.environ.get('NUM_ACTORS'))
# Create the Ray cluster coordinator object
cluster_coorindator = RayClusterCoordinator(ModelTrainer, num_workers)
# Get the job configuration set during launch.
# This is automatically set by Ray
job_runtime_env = json.loads(os.environ.get('RAY_JOB_CONFIG_JSON_ENV_VAR'))['runtime_env']

# Initialize workers
cluster_coordinator.initialize_workers(jax_compilation_cache=job_runtime_env['jax_compilation_cache'],
                                      ckpt_dir=job_runtime_env['ckpt_dir'],
                                      ckpt_freq=job_runtime_env['ckpt_freq'])

# Run the workers
run_results = asyncio.run(cluster_coordinator.run(restore=False))
```

```python
# launch_ray_cluster_job.py

from ray.job_submission import JobSubmissionClient, JobStatus

# Jobs are scheduled on a Ray cluster through a JobSubmissionClient
# The JobSubmissionClient connect to the Ray cluster via localhost
# Create the JobSubmissionClient
client = JobSubmissionClient("http://127.0.0.1:8265")
# Submit main.py to run on the Ray cluster whose head node's IP
# address is localhost
job_id = client.submit_job(
            entrypoint="python3 main.py",
            runtime_env={"working_dir" : ".",
                        "jax_compilation_cache" : "/path/to/compilation_cache",
                        "ckpt_dir" : "/path/to/checkpoint/directory",
                        "ckpt_freq" : 100,
            }
        )

while True:
    status = client.get_job_status(job_id)
    if status in {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED}:
        break
    time.sleep(90)

logs = client.get_job_logs(job_id)
print(logs)
```

Recall that the last line of the SLURM batch script executes the `launch_ray_cluster_job.py` script. This script is called the driver. The driver script is run on the same physical node as the Ray head node and is the reason the address to connect the JobSubmissionClient to the Ray cluster is `http://127.0.0.1` or
`localhost`. 

## Sharp bits and current limitations

This tutorial is meant to serve as a guide for how to use Ray to implement a fault tolerant training framework with JAX, but we'd like to note some limitations of the infrastructure explained in this tutorial:

1. While most aspects of the design are agnostic to the type of hardware being used, it has only been tested on GPUs. It is likely that the same principles can be applied to make the design applicable with minimal change to run on TPUs and other accelerators but that has not been tested yet. 

2. The implicit assumption in the design is that the same number of physical nodes is available before and after failure; meaning that it doesn't support recovery in the case of dead nodes unless hot spares are available.
