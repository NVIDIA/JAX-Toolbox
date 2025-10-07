import argparse
import functools
import jax

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int)
parser.add_argument("--nranks", type=int)
args = parser.parse_args()

jax.distributed.initialize(
    coordinator_address="127.0.0.1:42424",
    num_processes=args.nranks,
    process_id=args.rank,
    local_device_ids=[args.rank],
)
assert jax.device_count() == 2
assert jax.local_device_count() == 1


@functools.partial(
    jax.jit, compiler_options={"xla_gpu_enable_latency_hiding_scheduler": True}
)
def distinctively_named_function_with_lhs(x):
    return x @ x * x


square = jax.random.normal(jax.random.key(1), (32, 32))
for _ in range(5):
    square = distinctively_named_function_with_lhs(square)


@functools.partial(
    jax.jit, compiler_options={"xla_gpu_enable_latency_hiding_scheduler": False}
)
def distinctively_named_function_without_lhs(x):
    return x @ x.T


for _ in range(5):
    square = distinctively_named_function_without_lhs(square)


@jax.jit
def only_in_process_zero(x):
    return x * 3


# Do something only in process 0
if jax.process_index() == 0:
    square = only_in_process_zero(square)


@jax.jit
def another_distinctively_named_function(x):
    return x * x


# Do more stuff in all processes
for _ in range(5):
    square = another_distinctively_named_function(square)


@jax.jit
def another_one_only_in_process_zero(x):
    return x * 6


# Do something else only in process 0
if jax.process_index() == 0:
    square = another_one_only_in_process_zero(square)


@jax.jit
def only_in_process_one(x):
    return x * 42


# Do something only in process 1
if jax.process_index() == 1:
    square = only_in_process_one(square)


@jax.jit
def different_stacktraces(x):
    return x * 934


# This means the stack traces are different for the two calls
def trampoline0(square):
    return different_stacktraces(square)


def trampoline1(square):
    return different_stacktraces(square)


if jax.process_index() == 0:
    square = trampoline0(square)
    square = trampoline1(square)
else:
    square = trampoline1(square)
    square = trampoline0(square)
