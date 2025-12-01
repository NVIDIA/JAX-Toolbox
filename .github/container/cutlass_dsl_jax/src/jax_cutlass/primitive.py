# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Union, Sequence, Callable
from functools import partial
import logging

import cuda.bindings.driver as cuda

import jax, jax.numpy as jnp
import jax.extend
from jax.interpreters import mlir
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src import ffi
from jax.tree import flatten, unflatten

try:
    from jax.experimental.buffer_callback import buffer_callback
except ImportError as e:
    # buffer_callback is used until we implement C++/FFI interface
    raise ImportError(
        "A more recent version (>=0.6.2) of Jax is required for cutlass_call. Current version:"
        f" {jax.__version__}"
    ) from e

import cutlass

from .compile import get_or_compile_kernel, build_function_spec
from .types import row_major_layout, default_tensor_mode, TensorMode

logger = logging.getLogger(__name__)

cutlass_call_inner_p = jax.extend.core.Primitive("cutlass_call_inner")
cutlass_call_inner_p.multiple_results = True


def cutlass_call(
    fn: Callable[..., None],
    *,
    output_shape_dtype: Any,
    input_layout: Any = None,
    output_layout: Any = None,
    input_mode: Any = None,
    output_mode: Any = None,
    input_output_aliases={},
    convert_tensors=True,
    allow_cuda_graph=True,
    compile_options=None,
    use_static_tensors=False,
    **kwargs,
):
    """Creates a callable that invokes a @cute.jit function.

    Args:
        fn: A @cute.jit decorated function that launches a cutlass kernel.
        output_shape_dtype: A pytree representing the shape and dtype of the output buffers.
        input_output_aliases: A mapping of input to output aliases. Positions are specified assuming
            a flattened input and output pytree.
        input_layout: Specifies the Jax layout for input arrays. If None then the layout will
            assume row-major order.
        output_layout: Specifies the Jax layout for output arrays. If None then the layout will
            assume row-major order.
        input_mode: Specifies a cute.Tensor dimension order for input tensors. If None then the order
            will assume the corresponding layout order specific by input_layout.
        output_mode: Specifies a cute.Tensor dimension order for output tensors. If None then the order
            will assume the corresponding layout order specific by output_layout.
        convert_tensors: Jax array buffers will be converted to cute.Tensor with static shape and
            layout. If disabled the kernel is instead given a JaxArray pointer directly.
        allow_cuda_graph: If false will prevent XLA from building a cuda graph of for this call.
        compile_options: Optional compiler arguments to pass into cute.compile.
        use_static_tensors: If True, tensor shapes and strides are treated as constexpr values by
        default. This can improve performance through compiler specialization but may not work
        properly with all kernels. Specific tensors may be marked static or dynamic using the mode
        and override this flag.
        kwargs: Optional constexpr parameters to pass into the kernel fn.

    Note: This API is experimental and subject to change!
    """
    output_shape_dtype = jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), output_shape_dtype
    )
    return _cutlass_call_impl(
        fn,
        output_shape_dtype=output_shape_dtype,
        input_layout=input_layout,
        output_layout=output_layout,
        input_mode=input_mode,
        output_mode=output_mode,
        input_output_aliases=input_output_aliases,
        convert_tensors=convert_tensors,
        allow_cuda_graph=allow_cuda_graph,
        compile_options=compile_options,
        use_static_tensors=use_static_tensors,
        **kwargs,
    )


def _normalize_tensor_mode(value: Any):
    if value is None:
        return [None]
    elif isinstance(value, (tuple, list)):
        if isinstance(value[0], int):  # single tuple of modes
            return TensorMode(tuple(value))
        else:
            flat, _ = jax.tree.flatten(
                [_normalize_tensor_mode(x) for x in value],
                is_leaf=lambda x: x is None or isinstance(x, TensorMode),
            )
            return flat
    elif isinstance(value, TensorMode):
        return [value]
    else:
        raise TypeError(f"Unexpected value for TensorMode {value} {type(value)}")


def _cutlass_call_impl(
    fn,
    *,
    output_shape_dtype: Any,
    input_layout: Any,
    output_layout: Any,
    input_mode: Any,
    output_mode: Any,
    input_output_aliases,
    convert_tensors,
    allow_cuda_graph,
    compile_options,
    use_static_tensors,
    **kwargs,
):
    multiple_results = isinstance(output_shape_dtype, Sequence)
    if not multiple_results:
        output_shape_dtype = (output_shape_dtype,)
    output_shape_dtype_flat, output_tree = jax.tree.flatten(output_shape_dtype)

    @partial(jax.jit, inline=True)
    def call_wrapper(*args):
        args_flat, args_tree = jax.tree.flatten(args)

        if input_layout is None:
            input_layout_flat = [row_major_layout(x) for x in args_flat]
        else:
            input_layout_flat = list(input_layout)
            for idx, (layout, arg) in enumerate(zip(input_layout_flat, args_flat)):
                if layout is None:
                    input_layout_flat[idx] = row_major_layout(arg)
        input_layout_flat = tuple(input_layout_flat)

        if output_layout is None:
            output_layout_flat = [row_major_layout(x) for x in output_shape_dtype_flat]
        else:
            output_layout_flat = list(output_layout)
            for idx, (layout, arg) in enumerate(
                zip(output_layout_flat, output_shape_dtype_flat)
            ):
                if layout is None:
                    output_layout_flat[idx] = row_major_layout(arg)
        output_layout_flat = tuple(output_layout_flat)

        if len(input_layout_flat) != len(args_flat):
            raise ValueError("Must has same number of input layouts as input arrays.")

        if len(output_layout_flat) != len(output_shape_dtype_flat):
            raise ValueError("Must has same number of output layouts as output arrays.")

        if input_mode is None:
            input_mode_flat = tuple(default_tensor_mode(x) for x in args_flat)
        else:
            input_mode_flat = _normalize_tensor_mode(input_mode)
            for idx, (mode, arg) in enumerate(zip(input_mode_flat, args_flat)):
                if mode is None:
                    input_mode_flat[idx] = default_tensor_mode(arg)
            input_mode_flat = tuple(input_mode_flat)

        if output_mode is None:
            output_mode_flat = tuple(
                default_tensor_mode(x) for x in output_shape_dtype_flat
            )
        else:
            output_mode_flat = _normalize_tensor_mode(output_mode)
            for idx, (mode, arg) in enumerate(
                zip(output_mode_flat, output_shape_dtype_flat)
            ):
                if mode is None:
                    output_mode_flat[idx] = default_tensor_mode(arg)
            output_mode_flat = tuple(output_mode_flat)

        if len(input_mode_flat) != len(args_flat):
            raise ValueError(
                f"Must has same number of input modes ({len(input_mode_flat)}) as input arrays ({len(args_flat)})."
            )

        if len(output_mode_flat) != len(output_shape_dtype_flat):
            raise ValueError(
                f"Must has same number of output modes ({len(output_mode_flat)}) as output arrays ({len(output_shape_dtype_flat)})."
            )

        # Validate dynamic mode settings match whatever static shape
        # information we got as input.
        for idx, (arg, mode) in enumerate(zip(args_flat, input_mode_flat)):
            if mode.mode is not None and len(mode.mode) != len(arg.shape):
                raise ValueError(
                    f"Input #{idx} has invalid mode {mode.mode} for shape {arg.shape}."
                )
        for idx, (arg, mode) in enumerate(
            zip(output_shape_dtype_flat, output_mode_flat)
        ):
            if mode.mode is not None and len(mode.mode) != len(arg.shape):
                raise ValueError(f"Output #{idx} has invalid mode.")

        output_flat = cutlass_call_inner_p.bind(
            *args_flat,
            fn=fn,
            args_tree=args_tree,
            output_shape_dtype_flat=tuple(output_shape_dtype_flat),
            output_tree=output_tree,
            input_layout_flat=tuple(input_layout_flat),
            output_layout_flat=tuple(output_layout_flat),
            input_mode_flat=tuple(input_mode_flat),
            output_mode_flat=tuple(output_mode_flat),
            input_output_aliases=tuple(input_output_aliases.items()),
            convert_tensors=convert_tensors,
            allow_cuda_graph=allow_cuda_graph,
            compile_options=compile_options,
            use_static_tensors=use_static_tensors,
            **kwargs,
        )

        output = jax.tree.unflatten(output_tree, output_flat)
        return output if multiple_results else output[0]

    return call_wrapper


@cutlass_call_inner_p.def_abstract_eval
def cutlass_call_inner_p_abstract(*_, output_shape_dtype_flat, **__):
    return [jax.core.ShapedArray(x.shape, x.dtype) for x in output_shape_dtype_flat]


def cutlass_call_inner_p_impl(
    *args_flat,
    fn,
    args_tree: Any,
    output_shape_dtype_flat: Any,
    output_tree: Any,
    input_layout_flat: Any,
    output_layout_flat: Any,
    input_mode_flat: Any,
    output_mode_flat: Any,
    input_output_aliases,
    convert_tensors,
    allow_cuda_graph,
    compile_options,
    use_static_tensors,
    **kwargs,
):
    input_output_aliases = dict(input_output_aliases)

    # TODO: Need to support device-less compilation and defer module load
    # so we can compile at trace time. While we could compile here with one
    # device, the JitExecutor does not support more than one device per
    # instance. For now we explicitly compile and load under each context.
    spec = build_function_spec(
        args_flat,
        args_tree,
        output_shape_dtype_flat,
        output_tree,
        input_layout_flat,
        output_layout_flat,
        input_mode_flat,
        output_mode_flat,
        input_output_aliases,
        convert_tensors,
        compile_options,
        use_static_tensors,
        kwargs,
    )

    def wrap(fn, spec):
        def _inner(ctx, *args):
            kernel = get_or_compile_kernel(fn, spec, ctx.stream)
            kernel(ctx, *args)

        return _inner

    fun = buffer_callback(
        wrap(fn, spec),
        result_shape_dtypes=output_shape_dtype_flat,
        input_output_aliases=dict(spec.input_output_aliases),
        command_buffer_compatible=allow_cuda_graph,
    )
    return fun(*args_flat)


def _cutlass_call_jvp_rule(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError(
        "cutlass_call does not support VJP. Please use `jax.custom_jvp` for taking gradients."
    )


ad.primitive_jvps[cutlass_call_inner_p] = _cutlass_call_jvp_rule


def _cutlass_call_transpose_rule(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError(
        "cutlass_call does not support transpose. Please use `jax.custom_vjp` for taking gradients."
    )


ad.primitive_transposes[cutlass_call_inner_p] = _cutlass_call_transpose_rule


def _cutlass_call_vmap_rule(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError(
        "cutlass_call does not support batching with jax.vmap. Please "
        "use jax.custom_batching.custom_vmap for applying vmap. "
    )


batching.primitive_batchers[cutlass_call_inner_p] = _cutlass_call_vmap_rule

jax._src.dispatch.simple_impl(cutlass_call_inner_p)
jax._src.dispatch.prim_requires_devices_during_lowering.add(cutlass_call_inner_p)
lowering = mlir.lower_fun(cutlass_call_inner_p_impl, multiple_results=True)
mlir.register_lowering(cutlass_call_inner_p, lowering, platform="cuda")
