import asyncio.selector_events
import time
import ray
import traceback
import os
import jax
import random
import redis
import datetime
import asyncio
from contextlib import contextmanager
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy as NASS
from flax import linen as nn
import jax.numpy as jnp
import optax
from optax._src.transform import ScaleByAdamState
from optax._src.base import EmptyState
import orbax.checkpoint as ocp

# Define the coordinator and the ResilientWorker
class RayClusterCoordinator:
    def __init__(self, worker_cls, hang_time_threshold) -> None:
        self.worker_cls = worker_cls
        self.num_workers = int(os.environ.get('NGPUS'))
        self.num_workers_per_node = int(os.environ.get('GPUS_PER_NODE'))
        self.workers_initialized = False
        self.log = lambda user_str: print(user_str, flush=True)
        self.hang_time_threshold = hang_time_threshold if hang_time_threshold is not None else 300

        self.redis_addr = os.environ.get('REDIS_ADDR').split(':')

        worker_node_info, self.num_physical_nodes = self._get_schedulable_worker_info()
        self.workers = [worker_cls.options(num_gpus=1,
                                           num_cpus=16,
                                           resources={"worker_units": 1},
                                           scheduling_strategy=NASS(node_id=worker_node_info[i][0], soft=False)).remote(i, 
                                                                                                                        worker_node_info[i][1],
                                                                                                                        worker_node_info[i][2])
                                           for i in range(self.num_workers)]

        self.jax_coordinator_ip = worker_node_info[0][2]
        self.redis = redis.Redis(host=self.redis_addr[0], port=int(self.redis_addr[1]), decode_responses=True, password=None)
        self._init_sync_dict()
    
    def _get_schedulable_worker_info(self):
        worker_node_info = []
        worker_nodes = sorted([node for node in ray.nodes() if (node['Alive'] and 'worker_units' in node['Resources'])], 
                              key=lambda x: x['NodeID'])

        num_nodes_required = self.num_workers // self.num_workers_per_node
        num_nodes_available = len(worker_nodes)
        assert num_nodes_required <= num_nodes_available
        
        worker_nodes = worker_nodes[:num_nodes_required]
        for worker_node_id, worker_node in enumerate(worker_nodes):
            for _ in range(self.num_workers_per_node):
                worker_node_info.append((worker_node['NodeID'], worker_node_id, worker_node['NodeName']))

        return worker_node_info, num_nodes_required

    def _init_sync_dict(self):
        self.redis.flushdb()
        init_time = datetime.datetime.now().isoformat()
        for pid in range(self.num_workers):
            self.redis.set(pid, init_time)

    def initialize_workers(self, **kwargs):
        self.worker_init_kwargs = kwargs
        coordinator_port = random.randint(1, 100000)  % 2**12 + (65535 - 2**12 + 1)
        self.jax_coordinator_addr = f"{self.jax_coordinator_ip}:{coordinator_port}"
        ray.get([w.initialize.remote(self.jax_coordinator_addr, self.num_workers, **kwargs) for i, w in enumerate(self.workers)])
        self.workers_initialized = True
    
    async def _run_workers_async(self, *args, **kwargs):
        worker_run_futures = [w.run.remote(*args, **kwargs) for w in self.workers]
        while True:
            completed_worker_results = []
            for _, wf in enumerate(worker_run_futures):
                try:
                    worker_result = ray.get(wf, timeout=0)
                    completed_worker_results.append(worker_result)
                except ray.exceptions.GetTimeoutError:
                    continue
            
            if len(completed_worker_results) < len(self.workers):
                self.log(f"All workers seem to be alive, but only {len(completed_worker_results)} completed")
                await asyncio.sleep(30)
            else:
                self.log(f"All {len(completed_worker_results)} workers completed. Returning results.")
                return completed_worker_results
    
    async def _detect_worker_hang_async(self):
        # Check if processes are hanging
        while True:
            await asyncio.sleep(30)
            for pid in range(self.num_workers):
                current_time = datetime.datetime.now()
                last_hearbeat_time = datetime.datetime.fromisoformat(self.redis.get(pid))
                elapsed = (current_time - last_hearbeat_time).total_seconds()
                if elapsed > self.hang_time_threshold:
                    self.log(f"Worker {pid} has been hanged for {elapsed / 60} minutes")
                    raise Exception(f"Worker {pid} appears to have hanged")

            self.log("No hangs detected")

    async def run(self, *args, **kwargs):
        if not self.workers_initialized:
            raise ValueError("""Cannot run workers without initializing them first. 
                                Please call the initialize_workers method of your cluster coordinator first.""")

        runners = asyncio.create_task(self._run_workers_async(*args, **kwargs))
        hang_detector = asyncio.create_task(self._detect_worker_hang_async())
        while True:
            try:
                done, _ = await asyncio.wait({runners, hang_detector}, return_when=asyncio.FIRST_COMPLETED)
                for task in done:    
                    # If the runner finish with exception first this will raise an exception
                    # If the hang detector finishes with exception first this will raise an exception
                    # The only case in which task.result() does not raise an exception is when
                    # the runners finish first without raising an exception. In that case
                    # get the results from the runners and cancel the hang detector task 
                    # before returning
                    result = task.result()
                    hang_detector.cancel()
                    return result
            except Exception as e:
                self.log(f"Encountered exception {type(e).__name__}")
                self.log(traceback.format_exc())
                
                self.log("Cancelling all tasks in event loop...")
                runners.cancel()
                hang_detector.cancel()
                self.log("Done cancelling all tasks in event loop")

                self.log("Killing all ray actors...")
                for w in self.workers:
                    ray.kill(w)
                self.workers_initialized = False
                del self.workers
                self.log("Done killing all ray actors")

                # Restart workers and reinitialize tasks
                self.log("Restarting all actors")
                worker_node_info, self.num_physical_nodes = self._get_schedulable_worker_info()
                self.workers = [self.worker_cls.options(num_gpus=1, 
                                                        num_cpus=16,
                                                        resources={"worker_units": 1},
                                                        scheduling_strategy=NASS(node_id=worker_node_info[i][0], soft=False)).remote(i, 
                                                                                                                                     worker_node_info[i][1],
                                                                                                                                     worker_node_info[i][2])
                                           for i in range(self.num_workers)]
                self.jax_coordinator_ip = worker_node_info[0][2]
                self._init_sync_dict()
                self.initialize_workers(**self.worker_init_kwargs)

                self.log("Reinitializing tasks")
                kwargs["recovery_id"] += 1
                runners = asyncio.create_task(self._run_workers_async(*args, **kwargs))
                hang_detector = asyncio.create_task(self._detect_worker_hang_async())

class ResilientWorker:
    def __init__(self, process_id, physical_node_id, physical_node_ip):
        self.process_id = process_id
        self.physical_node_id = physical_node_id
        self.host_ip = physical_node_ip

        self.redis_addr = os.environ.get('REDIS_ADDR').split(':')
        self.logical_gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES'))
        self.redis = redis.Redis(host=self.redis_addr[0], port=int(self.redis_addr[1]), decode_responses=True, password=None)
    
    def get_process_id(self):
        return self.process_id
    
    def get_host_ip(self):
        return self.host_ip
     
    def get_logical_gpu_id(self):
        return self.logical_gpu_id

    def get_physical_node_id(self):
        return self.physical_node_id
    
    def initialize(self, coordinator_addr, num_processes):
        jax.distributed.initialize(coordinator_address=coordinator_addr, num_processes=num_processes, process_id=self.process_id, local_device_ids=0)

    def send_heartbeat(self):
        current_time = datetime.datetime.now().isoformat()
        self.redis.set(self.process_id, current_time)
    
    def run(self, *args, **kwargs):
        raise NotImplementedError

# Define the model 
# For this example we will just use a toy 2-layer MLP and feed it random normal data
class FFN(nn.Module):
    hidden_dim: int
    output_dim: int

    def setup(self):
        self.linear1 = nn.Dense(features=self.hidden_dim)
        self.linear2 = nn.Dense(features=self.output_dim)
    
    def __call__(self, input):
        return nn.relu(self.linear2(nn.relu(self.linear1(input))))

# Define the class that inherits from JaxWorker
# This is the class that will encapsulate all the computation to be performed
@ray.remote
class ModelTrainer(ResilientWorker):
    def __init__(self, process_id, physical_node_id, physical_node_ip):
        super().__init__(process_id, physical_node_id, physical_node_ip)
    
    def initialize(self, coordinator_addr, num_processes, **kwargs):
        return super().initialize(coordinator_addr, num_processes)

    @staticmethod
    def forward_and_loss(model, params, input):
        model_output = model.apply(params, input)
        return jnp.sum(model_output)
    
    @staticmethod
    def train_step(model, params, optimizer, optimizer_state, input):
        loss, grads = jax.value_and_grad(ModelTrainer.forward_and_loss, argnums=1)(model, params, input)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)
        return loss, params, optimizer_state
    
    @staticmethod
    def checkpoint(ckpt_mgr, params, optimizer_state, step):
        ckpt_mgr.wait_until_finished()
        optimizer_state_pytree = {"count" : optimizer_state[0].count, "mu" : optimizer_state[0].mu, "nu" : optimizer_state[0].nu}
        ckpt_mgr.save(step, 
                          args=ocp.args.Composite(params=ocp.args.StandardSave(params), 
                                                  optimizer_state=ocp.args.StandardSave(optimizer_state_pytree)))

    def run(self, input_shape, num_steps, recovery_id):
        rng = jax.random.key(42)

        # Define mesh to shard parameters, in this simple
        # example we will just use data parallelism
        mesh = jax.sharding.Mesh(jax.devices(), ('data_parallel',))
        parameter_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))

        # Define the model, optimizer and checkpoint manager
        model = FFN(hidden_dim=1024, output_dim=2048)
        optimizer = optax.adam(learning_rate=1e-3)
        ckpt_mgr = ocp.CheckpointManager("/tmp/ray_example/checkpoints", 
                                         item_names=('params', 'optimizer_state'),
                                         options=ocp.CheckpointManagerOptions(save_interval_steps=1, max_to_keep=3))
        
        # For the model parameters and optimizer state check if there is a checkpoint
        # If yes, load from checkpoint
        # Otherwise initialize the parameters
        if ckpt_mgr.latest_step():
            print(f"Checkpoint found! Restoring from step {ckpt_mgr.latest_step()}", flush=True)
            ckpt_mgr.wait_until_finished()
            ckpt = ckpt_mgr.restore(ckpt_mgr.latest_step(), 
                                    args=ocp.args.Composite(params=ocp.args.StandardRestore(), optimizer_state=ocp.args.StandardRestore()))
            params, optimizer_state_pytree = ckpt.params, ckpt.optimizer_state
            optimizer_state = (ScaleByAdamState(count=optimizer_state_pytree["count"], 
                                          mu=optimizer_state_pytree["mu"], 
                                          nu=optimizer_state_pytree["nu"]), 
                                          EmptyState())
        else:
            print("No checkpoint found, starting from scratch", flush=True)
            params = model.init(rng, jax.random.normal(rng, input_shape))
            params = jax.tree.map(lambda leaf: jax.make_array_from_process_local_data(parameter_sharding, leaf, leaf.shape), params)
            optimizer_state = optimizer.init(params)

        train_step = jax.jit(ModelTrainer.train_step, static_argnames=['model','optimizer'])
        input_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data_parallel', None))
        starting_step = (ckpt_mgr.latest_step() + 1) if ckpt_mgr.latest_step() is not None else 0

        # Train loop
        failure_timer_start = datetime.datetime.now()
        for step in range(starting_step, num_steps):
            input = jax.random.normal(rng, shape=input_shape)
            sharded_input = jax.make_array_from_process_local_data(input_sharding, input, input.shape)
            loss, params, optimizer_state = train_step(model, params, optimizer, optimizer_state, sharded_input)

            # Wait for step to finish before checkpointing then save checkpoint
            params, optimizer_state = jax.block_until_ready((params, optimizer_state))
            ModelTrainer.checkpoint(ckpt_mgr, params, optimizer_state, step)
            self.send_heartbeat()

            # In the first two instances of the program (before any failures and after 1 failure)
            # Only make the process with id = 0 (the JAX coordinator) fail with two different
            # failure modes: an "ungraceful" crash and a hang at steps 10 and 20 respectively
            # In other instances of of the program (beyond the second recovered run), let other processes
            # randomly fail every 10 iterations
            if recovery_id == 0 and step == 10:
                if self.process_id == 0:
                    eval((lambda:0).__code__.replace(co_consts=()))
            
            if recovery_id == 1 and step == 20:
                if self.process_id == 0:
                    time.sleep(300)
            
            if recovery_id == 2 and (step > 20 and step % 10 == 0):
                if random.random() < 0.5:
                    # Cause a seg fault, no graceful exception propagation
                    eval((lambda:0).__code__.replace(co_consts=()))
                else:
                    time.sleep(300)

            print(f"Process ID = {self.process_id}, Step = {step}, Loss = {loss}", flush=True)

        return
    
if __name__ == '__main__':
    ray.init(address='auto', logging_level=0)
    hang_time_threshold = 120
    coordinator = RayClusterCoordinator(ModelTrainer, hang_time_threshold)
    coordinator.initialize_workers()
    coordinator.log("Initialized workers")
    asyncio.run(coordinator.run(input_shape=(16, 512), num_steps=50, recovery_id=0))