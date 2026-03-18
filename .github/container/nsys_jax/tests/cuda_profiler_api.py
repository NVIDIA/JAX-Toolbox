from ctypes import cdll
import jax
import sys
import time

# what should this test program do after cudaProfilerStop?
mode = sys.argv[1]
assert mode in {"sleep", "exit42"}
assert all(map(lambda d: d.platform == "gpu", jax.devices()))


@jax.jit
def distinctively_named_function(x):
    return x @ x.T


libcudart = cdll.LoadLibrary("libcudart.so")
square = jax.random.normal(jax.random.key(1), (32, 32))
square = distinctively_named_function(square)
libcudart.cudaProfilerStart()
square = distinctively_named_function(square)
square.block_until_ready()
libcudart.cudaProfilerStop()
if mode == "sleep":
    # Sleep long enough that nsys always wins the race to termination
    time.sleep(600)
    sys.exit(24)
elif mode == "exit42":
    print("Exiting with status code 42")
    sys.exit(42)
raise Exception(f"Unknown exit mode {mode}")
