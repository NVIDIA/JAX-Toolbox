# User Guide on Native XLA-FP8
JAX offers support for many variations of FP8, such as E4M3 (`jnp.float8_e4m3fn`) and E5M2 (`jnp.float8_e5m2`). E4M3 boasts twice the resolution of E5M2 but covers only half the range. Typically, E4M3 is the recommended data type for the forward pass, while E5M2 is preferred for gradients in the backward pass.

Due to the limited range of FP8 data types, higher-precision data must be scaled to fit within the FP8 representable range, a process known as quantization (Q). Conversely, de-quantization (DQ) rescales the FP8 data back to its original data type.

In cublasLt, the FP8 matrix multiplication (matmul) operation is facilitated through an API that accepts two FP8 tensors and two scaling factors in FP32 as scalars. XLA, our compiler, is capable of recognizing patterns like FP8->DQ->Matmul and subsequently invokes the FP8 cublasLt API with scaling factors obtained from the DQ operation.

## JAX

The JAX matmul operations already support the FP8 dtype inputs. To perform FP8 matmul, you can use the `jax.lax.dot` function with FP8 dtype inputs as shown below:

```python
A = jax.random.uniform(random_key, (16, 32)).astype(jnp.float8_e4m3fn)
B = jax.random.uniform(random_key, (32, 64)).astype(jnp.float8_e4m3fn)
C = jax.lax.dot(A, B) # Result in E4M3
```

However, there are two main issues with this approach. Firstly, it uses a fixed scaling factor of 1.0 for the inputs. Secondly, it does not support inputs of different FP8 dtypes. For example, if A is in e5m2 format and B is in e4m3 format, the matrix multiplication is performed on promoted dtype fp16. These limitations make this method less practical for real-world scenarios, where updating scaling factors is necessary at each training step, and mixed FP8 dtypes are required, especially for backpropagation.

To address these limitations and create a more versatile FP8 matmul, we recommend leveraging XLA-FP8. With XLA-FP8, users can specify different input dtype combinations and how their associated scaling factors are updated. XLA can recognize these patterns and optimize FP8-related operations, such as type casting and scaling factor bookkeeping, to minimize overhead.

Here's an example of how to exploit XLA-FP8 using JAX:
```python
A = jax.random.uniform(random_key, (16, 32), dtype=jnp.bfloat16)
B = jax.random.uniform(random_key, (32, 64), dtype=jnp.bfloat16)

# Use scaling factors from last step.
A_scale = 1.0 
B_scale = 1.0 

# Quantization: Convert to FP8.
A_fp8 = (A / A_scale).astype(jnp.float8_e4m3fn)
B_fp8 = (B / B_scale).astype(jnp.float8_e4m3fn)

# JIT your model, which takes-in both the FP8 data along with scaling factors.
@jax.jit
def matmul_fp8(a_fp8, a_scale, b_fp8, b_scale):
    # Dequantization: Up-cast from FP8 to a wider type.
    a = a_fp8.astype(jnp.bfloat16) * a_scale
    b = b_fp8.astype(jnp.bfloat16) * b_scale
    c = jax.lax.dot(a, b)
    return c

C = matmul_fp8(A_fp8, A_scale, B_fp8, B_scale) # Result in BF16.

# Delayed Scaling: Calculate the scaling factors, which are always stored in wider types.
E4M3_MAX = jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.float16)
A_scale = jnp.max(jnp.abs(A)) / E4M3_MAX
B_scale = jnp.max(jnp.abs(B)) / E4M3_MAX
```

This approach overcomes the limitations of the previous method. It allows you to define how scaling factors for `A` and `B` are assigned and experiment with different combinations of `A_fp8` and `B_fp8` FP8 dtypes. Additionally, the example demonstrates delayed scaling, where scaling factors computed from previous steps are used for the current matrix multiplication. After the multiplication, you can update the scaling factors for the next training iteration. For further details, refer to [the provided documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#Mixed-precision-training-with-FP8).


## FLAX

With the FLAX library, incorporating custom operations into existing FLAX layers is a seamless process. Users can effortlessly integrate the provided custom op into FLAX layers by employing a simple "code-injection" approach. For instance, the only thing users need to do is to specify the custom operation as follows: `Dense(..., dot_general_cls=Fp8DotGeneralOp)`. This custom op encapsulates all FP8-related operations, addressing key aspects such as the placement of QDQ ops, algorithms for updating scaling factors, and the selection of FP8 dtype combinations for forward and backward propagation.

Consider the following example, where the linear operation is performed using FP8:
```python
from flax.linen import Dense
from flax.linen.fp8_ops import Fp8DotGeneralOp

A = jax.random.uniform(random_key, (16, 32)).astype(jnp.bfloat16)
model = Dense(features=64, dtype=jnp.bfloat16, dot_general_cls=Fp8DotGeneralOp)
var_init = model.init(init_key, A)

@jax.jit
def fp8matmul(var, a): 
  c = model.apply(var, a) # Result in BF16
  return c

C = fp8matmul(var_init, A)
```

While this code achieves FP8 matrix multiplication, it does not cover the aspect of updating the scaling factors, as indicated in the last three lines of the JAX example. Internally, FLAX layers handle the allocation of scaling factors and other FP8-related variables, such as amax history, within a collection named `overwrite_with_gradient`. As the name suggests, variables in this collection are updated directly by replacing them with their gradients. Fortunately, users are relieved from the intricacies of this updating process, as FLAX's `flax.training.train_state.TrainState` conveniently supports this type of collection. In this regard, users are not required to modify their scripts.

For a comprehensive example, please refer to the provided unit tests linked [here](https://github.com/google/flax/blob/85245ada6a5c39ae13fda6de644dceb8801dc6b4/tests/linen/linen_test.py#L901).


## Praxis/PaxML
Performing matmul with FP8 precision in the praxis/paxml environment is straightforward and seamless. The custom operation, `fp8_ops.Fp8EinsumOp`, facilitates this process, and integrating it into existing praxis layers involves configuring it within the `einsum_tpl`. Here's how you can achieve this:

```python
from praxis import base_layer
from praxis import pax_fiddle
from praxis.layers import linears
from praxis.layers.injection import fp8_nvidia_gpu as fp8_ops

p = pax_fiddle.Config(linears.Linear, input_dims=32, output_dims=64, fprop_dtype=jnp.bfloat16,
                      einsum_tpl=pax_fiddle.Config(fp8_ops.Fp8EinsumOp))
model = base_layer.instantiate(p)
var_init = model.init(init_key, A)

@jax.jit
def fp8matmul(var, a): 
  c = model.apply(var, a) # Result in BF16
  return c

C = fp8matmul(var_init, A)
print("Result in", C.dtype)
```

In a similar vein, managing FP8 variables is handled within the `overwrite_with_gradient` collection. The pax trainer function can recognize these variables and update them directly with their gradients. For a comprehensive example, please refer to the provided unit tests linked [here](https://github.com/google/praxis/blob/main/praxis/layers/injection/fp8_nvidia_gpu_test.py).

### Transformer Layer
PAXML models are constructed using the transformer layer provided by praxis. In theory, users can locate the matmul layers and reconfigure them to incorporate custom FP8 operations. However, we recommend utilizing FP8 matmul specifically for the QKV projection, attention output projection and the linear transformations in the feed-forward networks.

Enabling this feature is effortless. Users only need to include the option `--fdl.USE_FP8=True` in their experiment configuration. This simple step activates the recommended layers, allowing the transformer layer to employ FP8 matmul. It's crucial to highlight that ENABLE_TE must be turned off for this functionality to work effectively.

In addition to the suggested XLA flags mentioned in [this section](https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/rosetta/projects/pax/README.md#xla-flags), we also recommend setting these following XLA flags. The execution script should look like:
```bash
export XLA_FLAGS=" \
    --xla_gpu_enable_reduction_epilogue_fusion=false \
    --xla_gpu_enable_triton_gemm=false \
    --xla_gpu_enable_cublaslt=true \
    --xla_gpu_enable_latency_hiding_scheduler=true \
    --xla_gpu_enable_async_collectives=true \
    --xla_gpu_enable_highest_priority_async_stream=true \
    --xla_gpu_all_reduce_combine_threshold_bytes=51200 "
export ENABLE_TE=0
python -m paxml.main \
    ...
    --fdl.USE_FP8=True \
    ...
```

### Transformer Engine vs Native FP8 Support
Native XLA-FP8 specifically targets matrix multiplication operations. In contrast, the Transformer Engine focuses on enhancing the overall performance of the entire transformer layer. This encompasses not only the FP8 matrix multiplication but also attention mechanisms, layer normalizations, and other components.

In practical terms, XLA-FP8 performs pattern matching and rewrites the matrix multiplication operations in the operation graph to utilize FP8 matrix multiplication. On the other hand, with TE, the [entire Praxis transformer](https://github.com/google/praxis/blob/main/praxis/layers/transformers.py) layer will be substituted with our [Transformer Engine
layer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/jax.html#transformer_engine.jax.flax.TransformerLayer), offering a comprehensive performance enhancement.

