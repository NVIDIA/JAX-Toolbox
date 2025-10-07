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

import os
import gc
import ctypes
import inspect
from typing import Any, Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import time
import logging
import threading

import cuda.bindings.driver as cuda

import jax
import jax.numpy as jnp
from jax.experimental.buffer_callback import ExecutionContext
import jaxlib

from .types import (
    jax_to_cutlass_dtype,
    from_dlpack,
    JaxArray,
    JaxArrayList,
    TensorMode,
    make_placeholder_array,
    DEFAULT_CUTLASS_DEVICE_MEMSPACE,
    DEFAULT_CUTLASS_DEVICE_BUFFER_ALIGNMENT,
)

import cutlass
import cutlass.cute as cute
from cutlass.cute import AddressSpace
from cutlass.cutlass_dsl.cutlass import CuTeDSL
from cutlass.base_dsl.runtime.cuda import unload_cubin_module

logger = logging.getLogger(__name__)

_CUTLASS_COMPILE_CACHE = {}


@dataclass(frozen=True)
class Arg:
    idx: int  # position in pytree
    shape: tuple[int, ...]
    dtype: jnp.dtype
    layout: tuple[int, ...]
    mode: TensorMode


@dataclass(frozen=True)
class FunctionSpec:
    """Contains a specification of the inputs and outputs to the kernel."""

    in_args: tuple[Arg, ...]
    input_tree: Any
    out_args: tuple[Arg, ...]
    output_tree: Any
    input_output_aliases: tuple[tuple[int, int], ...]
    input_layout: tuple[tuple[int, ...]]
    input_mode: tuple[TensorMode, ...]
    output_layout: tuple[tuple[int, ...]]
    output_mode: tuple[TensorMode, ...]
    convert_tensors: bool
    kwargs: tuple[tuple[str, Any]]

    def get_compile_args(self):
        """Returns the arguments to provide to cute.compile."""
        compiler_ins = [
            make_placeholder_array(
                jax_to_cutlass_dtype(leaf.dtype),
                leaf.shape,
                leaf.layout,
                DEFAULT_CUTLASS_DEVICE_MEMSPACE,
                mode.ptr_assumed_align,
            )
            for leaf, mode in zip(self.in_args, self.input_mode)
        ]
        compiler_outs = [
            make_placeholder_array(
                jax_to_cutlass_dtype(leaf.dtype),
                leaf.shape,
                leaf.layout,
                DEFAULT_CUTLASS_DEVICE_MEMSPACE,
                mode.ptr_assumed_align,
            )
            for leaf, mode in zip(self.out_args, self.output_mode)
        ]
        return JaxArrayList(tuple(sum([compiler_ins, compiler_outs], [])))

    def get_runtime_args(self, out, *args):
        """Returns the arguments to provide to the compiled function at runtime."""
        ins = [from_dlpack(args[i]).iterator for i, spec in enumerate(self.in_args)]
        outs = [from_dlpack(out[i]).iterator for i, spec in enumerate(self.out_args)]
        return JaxArrayList(tuple(sum([ins, outs], [])))


@cute.jit
def jit_wrapper(
    stream: cuda.CUstream,
    args: JaxArrayList,
    *,
    wrapped_fn: cutlass.Constexpr,
    spec: cutlass.Constexpr,
):
    # split buffer argument into inputs and outputs and return to tree
    ins, outs = args[: len(spec.in_args)], args[(len(spec.in_args)) :]
    if cutlass.const_expr(spec.convert_tensors):
        ins = [x.get_tensor(a.mode) for x, a in zip(ins, spec.in_args)]
        outs = [x.get_tensor(a.mode) for x, a in zip(outs, spec.out_args)]
    ins = jax.tree.unflatten(spec.input_tree, ins)
    outs = jax.tree.unflatten(spec.output_tree, outs)
    wrapped_fn(stream, *ins, *outs, **dict(spec.kwargs))


@dataclass
class CompileResult:
    """Holds reference to the compiled kernel and arguments.

    compiled_fn: The compiled function (a JitExecutor).
                 This reference keeps CUDA modules alive.

    """

    compiled_fn: cutlass.base_dsl.jit_executor.JitExecutor
    spec: FunctionSpec

    def __call__(self, ctx: ExecutionContext, out, *args):
        self.compiled_fn(
            cuda.CUstream(ctx.stream), self.spec.get_runtime_args(out, *args)
        )


def _check_is_valid_type(x, is_input):
    if not is_input:
        if not isinstance(x, jax.ShapeDtypeStruct):
            raise TypeError("Invalid output value passed.", x)
    else:
        if not isinstance(x, jax.Array):
            raise TypeError("Invalid type passed.", x)


def _build_arg_tree(args, specs, is_input):
    args = []
    for idx, (arg, layout) in enumerate(zip(args_flat, specs)):
        _check_is_valid_type(arg, is_input)
        args.append(Arg(idx, arg.shape, arg.dtype, layout))
    args = jax.tree.unflatten(args_tree, args)

    return args, args_tree, is_single_leaf_node


def build_function_spec(
    ins,
    in_tree,
    outs,
    out_tree,
    input_layout,
    output_layout,
    input_mode,
    output_mode,
    input_output_aliases,
    convert_tensors,
    kwargs,
):
    # TODO: Improve type checking and validate pytree structures.
    # TODO: Improve Pytree support for more complex or user defined structures.

    in_args = []
    for idx, (arg, layout, mode) in enumerate(zip(ins, input_layout, input_mode)):
        _check_is_valid_type(arg, is_input=True)
        in_args.append(Arg(idx, arg.shape, arg.dtype, layout, mode))

    out_args = []
    for idx, (arg, layout, mode) in enumerate(zip(outs, output_layout, output_mode)):
        _check_is_valid_type(arg, is_input=False)
        out_args.append(Arg(idx, arg.shape, arg.dtype, layout, mode))

    # Return the argument specs to the original pytree structure
    # We need this structure to sanely match index positions of the
    # arguments to the kernel.
    ins_args_structured = jax.tree.unflatten(in_tree, in_args)
    out_args_structured = jax.tree.unflatten(out_tree, out_args)

    # Assign per-leaf aliases
    input_output_aliases_per_leaf = {}
    for input_arg_alias_idx in input_output_aliases:
        flat_in, _ = jax.tree.flatten(ins_args_structured[input_arg_alias_idx])
        flat_out, _ = jax.tree.flatten(
            out_args_structured[input_output_aliases[input_arg_alias_idx]]
        )
        for i, o in zip(flat_in, flat_out):
            input_output_aliases_per_leaf[i.idx] = o.idx

    # Remove aliased arguments from output set since they are also provided
    # as inputs. This is done at the very top level of the tree to simplify
    # how we handle aliasing. The assumption is that the entire pytree is
    # aliased.
    out_args_structured = list(out_args_structured)
    for out_idx in sorted(tuple(set(input_output_aliases.values())), reverse=True):
        try:
            out_args_structured.pop(out_idx)
        except:
            raise ValueError(f"Invalid output alias in input_output_aliases.")
    out_args_structured = tuple(out_args_structured)

    in_args_flat, _ = jax.tree.flatten(ins_args_structured)
    out_args_flat, out_tree = jax.tree.flatten(out_args_structured)

    spec = FunctionSpec(
        tuple(in_args_flat),
        in_tree,
        tuple(out_args_flat),
        out_tree,
        tuple(input_output_aliases_per_leaf.items()),
        tuple(input_layout),
        tuple(input_mode),
        tuple(output_layout),
        tuple(output_mode),
        convert_tensors,
        tuple((k, kwargs[k]) for k in kwargs),
    )

    return spec


_compile_lock = threading.Lock()


def get_or_compile_kernel(fn, spec, stream):
    """Gets or compiles fn and returns a CutlassCompileResult.

    The function and its specification is used as a key to determine if a new
    function must be compiled.
    """
    cache_key = (fn, spec, stream)
    if cache_key in _CUTLASS_COMPILE_CACHE:
        return _CUTLASS_COMPILE_CACHE[cache_key]

    # Don't allow more than 1 thead to compile at any time.
    # We assume that the cache key is per thread so we don't need to lock
    # the above check in compile cache,
    # TODO: ideally this lock would happen in cute.compile as needed.
    compiled_fn = None
    with _compile_lock:
        start = time.time()
        try:
            compiled_fn = cutlass.cute.compile(
                jit_wrapper,
                cuda.CUstream(stream),
                spec.get_compile_args(),
                wrapped_fn=fn,
                spec=spec,
            )
        except Exception as e:
            # Log here because Jax can obscure the exception details.
            logger.exception("Compilation failure for kernel.")
            raise e
        end = time.time()
    logger.debug(f"Took {end-start} to compile cute kernel.")

    result = CompileResult(compiled_fn=compiled_fn, spec=spec)
    _CUTLASS_COMPILE_CACHE[cache_key] = result
    return result


def release_compile_cache():
    """Releases entries from the compile cache.

    Note that is may prevent cute dsl from saving its persistent compilation cache entries.
    """
    _CUTLASS_COMPILE_CACHE.clear()
    dsl = CuTeDSL._get_dsl()
    dsl.jit_cache.clear()
    # TODO: This is needed to release frames being held in the DSL
    # We should avoid holding such references as they unexpectedly
    # extend object lifetime.
    dsl.frame = None
    gc.collect()


class _DummyInitKernel:
    @cute.kernel
    def kernel(self):
        pass

    @cute.jit
    def init(self):
        pass


_CUTLASS_DSL_INITIALIZED = False


def initialize_cutlass_dsl():
    """Initializes cutlass DSL."""
    global _CUTLASS_DSL_INITIALIZED
    if _CUTLASS_DSL_INITIALIZED:
        return

    # TODO(mgoldfarb-nvidia): There are several runtime libraries that export C++ symbols
    # which conflict with jax libraries. Initializing cutlass before jax will cause these
    # symbols to incorrectly interpose. Our WAR is to for loading of jaxlib and its
    # dependant libraries to ensure all symbols are loaded prior to compiling cutedsl programs.
    # This linking issue is planed to be resolved in cute DSL 4.3.
    jaxlib_common = Path(jaxlib.__file__).parent / "libjax_common.so"
    if jaxlib_common.exists():
        ctypes.CDLL(str(jaxlib_common), mode=ctypes.RTLD_GLOBAL)

    kernel = _DummyInitKernel()
    with _compile_lock:
        logger.debug("Initializing cutlass dsl...")
        _ = cutlass.cute.compile(kernel.init)

    _CUTLASS_DSL_INITIALIZED = True
