from ctypes import cdll
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

num_devices = 2
assert jax.local_device_count() >= num_devices, (
    "This test needs a machine with [at least] 2 GPUs"
)
devices = jax.local_devices()[:num_devices]

libcudart = cdll.LoadLibrary("libcudart.so")

f32_per_device = 4096
mesh = Mesh(devices, axis_names=("i",))


def device_put_local(x: jax.Array):
    return [jax.device_put(x, d) for d in devices]


input = jax.make_array_from_single_device_arrays(
    (num_devices, f32_per_device),
    NamedSharding(mesh, P("i")),
    device_put_local(jnp.zeros((1, f32_per_device), dtype=jnp.float32)),
)


@partial(jax.jit, compiler_options={"xla_gpu_enable_latency_hiding_scheduler": True})
@partial(shard_map, mesh=mesh, in_specs=P("i"), out_specs=P("i"))
def where_the_magic_happens(x):
    # computation at the start of the module
    a = x.T @ x
    # collective, can happen in parallel with b2
    b1 = jax.lax.psum(a, "i")
    # computation, can happen in parallel with b1
    b2 = a @ a.T
    # computation depending on both b1 and b2
    return b1 @ b2


where_the_magic_happens(input)
where_the_magic_happens(input)
libcudart.cudaProfilerStart()
where_the_magic_happens(input)
where_the_magic_happens(input)
libcudart.cudaProfilerStop()
