import jax


@jax.jit
def distinctively_named_function(x):
    return x @ x.T


square = jax.random.normal(jax.random.key(1), (32, 32))
for _ in range(5):
    square = distinctively_named_function(square)
