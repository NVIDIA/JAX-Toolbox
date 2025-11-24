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

from typing import Type, Optional, Sequence, Union, Callable, Any, TypeVar
import sys
import ctypes
import math
import inspect
from dataclasses import dataclass, field
from functools import partial, reduce
from operator import mul
from itertools import chain
from typing import Annotated

import cuda.bindings.driver as cuda

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack as _from_dlpack
from cutlass.cute import AddressSpace, Numeric, IntTuple
from cutlass._mlir import ir

JAX_DTYPE_TO_CUTLASS_DTYPE = {
    # TODO(mgoldfarb-nvidia): Check passing boolean arrays via __dlpack__
    jnp.bool.dtype: cutlass.Boolean,
    jnp.int8.dtype: cutlass.Int8,
    jnp.int16.dtype: cutlass.Int16,
    jnp.int32.dtype: cutlass.Int32,
    jnp.int64.dtype: cutlass.Int64,
    jnp.uint8.dtype: cutlass.Uint8,
    jnp.uint16.dtype: cutlass.Uint16,
    jnp.uint32.dtype: cutlass.Uint32,
    jnp.uint64.dtype: cutlass.Uint64,
    jnp.bfloat16.dtype: cutlass.BFloat16,
    jnp.float16.dtype: cutlass.Float16,
    jnp.float32.dtype: cutlass.Float32,
    jnp.float64.dtype: cutlass.Float64,
    jnp.float8_e8m0fnu.dtype: cutlass.Float8E8M0FNU,
    jnp.float8_e5m2.dtype: cutlass.Float8E5M2,
    jnp.float8_e4m3.dtype: cutlass.Float8E4M3,
    jnp.float8_e4m3fn.dtype: cutlass.Float8E4M3FN,
    jnp.float8_e4m3b11fnuz.dtype: cutlass.Float8E4M3B11FNUZ,
    jnp.float4_e2m1fn.dtype: cutlass.Float4E2M1FN,
}

DEFAULT_CUTLASS_DEVICE_MEMSPACE = AddressSpace.gmem
DEFAULT_CUTLASS_DEVICE_BUFFER_ALIGNMENT = 256


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TensorMode:
    """Provides a specification of cute.Tensor modes and additional metadata about
    dynamic/static modes.

    Arguments:
        mode : Specifies the position of each mode in the tensor (M0, M1, ... MN)
    """

    mode: tuple[int, ...] | None = field(metadata=dict(static=True), default=None)
    # Indicates the shape and strides will be defined statically. Enabling may enable
    # additional optimization. Kernels that do not support static shapes will generate
    # compile errors if this is enabled so we leave it off by default.
    static: bool = field(metadata=dict(static=True), default=False)
    # Overrides the default pointer alignment. Generally this should not be changed
    # but is left here to provide a hook.
    ptr_assumed_align: int = field(
        metadata=dict(static=True), default=DEFAULT_CUTLASS_DEVICE_BUFFER_ALIGNMENT
    )

    def __post_init__(self):
        if self.mode is not None:
            if len(self.mode) != len(set(self.mode)):
                raise ValueError(
                    f"Invalid mode {self.mode} contains duplicate entries."
                )
            for m in self.mode:
                if m < 0 or m >= len(self.mode):
                    raise ValueError(
                        f"Invalid mode {self.mode} contains out of range entires."
                    )
        if (
            self.ptr_assumed_align <= 0
            or not math.log2(self.ptr_assumed_align).is_integer()
        ):
            raise ValueError(
                f"Invalid pointer alignment {self.ptr_assumed_align} must be power of 2."
            )


def row_major_layout(shaped):
    """Returns a row major layout given a shaped value.

    Row major layout is (N-1, N-2, ... 1, 0) for an N-dimensional tensor.
    """
    return tuple(reversed(range(len(shaped.shape))))


def default_tensor_mode(shaped) -> TensorMode:
    """Returns a default tensor mode given a shaped value.

    Default tensor mode is (0, 1, ... N-2, N-1) for an N_dimensional tensor.
    """
    return TensorMode(tuple(range(len(shaped.shape))))


def jax_to_cutlass_dtype(dtype):
    """Gets the corresponding cutlass dtype given a jax dtype."""
    dtype = jnp.dtype(dtype)
    if dtype not in JAX_DTYPE_TO_CUTLASS_DTYPE:
        raise ValueError(f"Jax dtype [{dtype}] has no equivalent cutlass dtype.")
    return JAX_DTYPE_TO_CUTLASS_DTYPE[dtype]


def from_dlpack(buffer, assumed_align: int = DEFAULT_CUTLASS_DEVICE_BUFFER_ALIGNMENT):
    """Convert device buffer to runtime Tensor."""
    return _from_dlpack(buffer, assumed_align=assumed_align)


class _JaxArrayBase(cute.Pointer):
    """Base class for the JaxArray and JaxRuntimeArray types."""

    def __init__(
        self,
        ptr: cute.Pointer,
        shape: tuple[int, ...],
        order: tuple[int, ...] | None = None,
    ):
        self.ptr = ptr
        self._shape = tuple(shape)
        if order is None:
            order = tuple(reversed(range(len(self._shape))))
        if len(order) != len(shape):
            raise ValueError(f"order must be same length as shape", order, shape)
        for s in order:
            if s < 0 or s > len(self._shape):
                raise ValueError(f"Invalid index {s} in stride order", order, shape)
        if len(tuple(set(order))) != len(order):
            raise ValueError(f"order has duplicate indices", order)
        self._order = tuple(order)

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns physical shape of this jax array."""
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def order(self) -> tuple[int, ...]:
        """Returns stride order (layout) of this jax array."""
        return self._order

    @property
    def dtype(self) -> Type[Numeric]:
        """Returns cute dtype of this jax array."""
        return self.ptr.dtype

    @property
    def memspace(self):
        """Returns the address space of this jax array."""
        return self.ptr.memspace


class JaxArray(_JaxArrayBase):
    """Represents a jax.Array IR value passed to a cute kernel or function.

    The JaxArray is a shaped pointer with physical dimension specified by the Jax program.
    By default the data is assumed to follow row-major layout but a custom order
    (e.g. column-major) can also be used.

    e.g. (8, 4, 2) row-major strides are (8, 2, 1)

    JaxArray always have statically know shapes and strides.
    """

    def __init__(
        self,
        ptr: cute.Pointer,
        shape: tuple[int, ...],
        order: tuple[int, ...] | None = None,
    ):
        """Creates a Jax array from a cute.Pointer and shape/stride information.

        Args:
            ptr: The typed pointer.
            shape: A tuple of shape dimensions associated with the .
            order: An optional ordering of the dimensions in shape. If None the
                shape is assumed to be row-major.
        """
        if not hasattr(ptr, "value") or not isinstance(ptr.value, ir.Value):
            raise ValueError("not an ir.Value", ptr)
        super().__init__(ptr, shape, order)

    #
    # Compile Time IR Value Properties
    #
    # These methods allow JaxArray look like core cute.Pointer. The ptr must
    # be a cute.Pointer value

    @property
    def value(self) -> cute.Pointer:
        return self.ptr.value

    @property
    def type(self) -> ir.Type:
        return self.ptr.type

    @property
    def alignment(self) -> int:
        return self.ptr.alignment

    @property
    def max_alignment(self) -> int:
        return self.ptr.max_alignment

    def llvm_ptr(self, *, loc=None, ip=None) -> ir.Value:
        return self.ptr.llvm_ptr(loc, ip)

    def __add__(self, offset: IntTuple) -> "JaxArray":
        return JaxArray(self.ptr + offset, self._shape, self._order)

    def toint(self, *, loc=None, ip=None):
        return self.ptr.toint()

    def align(self, min_align: int, *, loc=None, ip=None) -> "JaxArray":
        return JaxArray(self.ptr.align(min_align, loc, ip), self._shape, self._order)

    def get_layout(
        self, mode: tuple[int, ...] | TensorMode = None, *, loc=None, ip=None
    ) -> cute.Layout:
        """Create a cute.Layout from this JaxArray.

        Physical: (I, J, K) strides are (J*K, K, 1) in row-major order.

        mode = (2, 0, 1) : shape becomes (K, I, J) strides become (1, J*K, K)
        mode = (1, 2, 0) : shape becomes (J, K, I) strides become (K, 1, J*K)

        :param mode: Maps the physical shape dimension to logical shape dimensions. If not given the physical layout is used.
        :type tuple[int,...]: Tuple that is same size as shape.
        """
        if isinstance(mode, (tuple, list)):
            mode = TensorMode(mode)

        shape = (
            self._shape if mode.static else [cutlass.as_numeric(m) for m in self._shape]
        )
        layout = cute.make_ordered_layout(tuple(shape), self._order, loc=loc, ip=ip)
        if mode is not None and mode.mode is not None:
            layout = cute.select(layout, mode.mode)
        return layout

    def get_tensor(
        self, mode: tuple[int, ...] | TensorMode = None, *, loc=None, ip=None
    ) -> cute.Tensor:
        """Create a cute.Tensor from this JaxArray.

        :param mode: Maps the physical shape dimension to logical shape dimensions. If not given the physical layout is used.
        :type tuple[int,...]: Tuple that is same size as shape.
        :see get_layout
        """
        layout = self.get_layout(mode, loc=loc, ip=ip)
        return cute.make_tensor(self.ptr, layout)

    # Utility methods

    def __str__(self) -> str:
        return f"JaxArray<{self.ptr}:{self.shape}:{self.order}>"

    def __repr__(self) -> str:
        return str(self)

    # DynamicExpression Protocol

    def __extract_mlir_values__(self):
        return [self.ptr.value]

    def __new_from_mlir_values__(self, values):
        return JaxArray(
            self.ptr.__new_from_mlir_values__(values), self._shape, self._order
        )


class JaxRuntimeArray(_JaxArrayBase):
    """Runtime equivalent of jax.Array."""

    def __init__(
        self,
        ptr: cute.Pointer | int,
        shape: tuple[int, ...],
        order: tuple[int, ...] | None = None,
    ):
        super().__init__(ptr, shape, order)

    @property
    def alignment(self) -> int:
        return self.ptr._assumed_align

    def __str__(self) -> str:
        return f"JaxRuntimeArray<{self.ptr}:{self.shape}:{self.order}>"

    def __repr__(self) -> str:
        return str(self)

    # JitArgument Protocol

    def __c_pointers__(self):
        return self.ptr.__c_pointers__()

    def __get_mlir_types__(self):
        return self.ptr.__get_mlir_types__()


def make_runtime_array(
    value: Union[int, ctypes._Pointer],
    dtype: Type[Numeric],
    shape: tuple[int, ...],
    order: tuple[int, ...] | None = None,
    mem_space: AddressSpace = AddressSpace.generic,
    assumed_align=DEFAULT_CUTLASS_DEVICE_BUFFER_ALIGNMENT,
):
    """Creates a JaxRuntimeArray and its underlying pointer."""
    ptr = cute.runtime.make_ptr(dtype, value, mem_space, assumed_align)
    return JaxRuntimeArray(ptr, shape, order)


def make_placeholder_array(
    dtype: Type[Numeric],
    shape: tuple[int, ...],
    order: tuple[int, ...] | None = None,
    mem_space: AddressSpace = AddressSpace.generic,
    assumed_align=DEFAULT_CUTLASS_DEVICE_BUFFER_ALIGNMENT,
):
    """Creates a JaxRuntimeArray that can be used as a placeholder for cute.compile."""
    # n.b. 0 causes issues with c_types so we use a non-zero address that is aligned
    # The value should not matter but we do it for good measure.
    addr = 2 ** int(math.ceil(math.log2(assumed_align)))
    return make_runtime_array(addr, dtype, shape, order, mem_space, assumed_align)


class JaxArrayList:
    """Holds list of JaxArray or JaxRuntimeArray.
    This class facilitates conversion of JaxRuntimeArray to JaxArray when crossing
    the jit boundary.
    """

    def __init__(self, arrays: Sequence[JaxArray]):
        self.arrays = tuple(arrays)

    def __getitem__(self, idx):
        return self.arrays[idx]

    def __len__(self):
        return len(self.arrays)

    def __iter__(self):
        return iter(self.arrays)

    def __c_pointers__(self):
        return [x.__c_pointers__()[0] for x in self.arrays]

    def __get_mlir_types__(self):
        return [x.__get_mlir_types__()[0] for x in self.arrays]

    def __extract_mlir_values__(self):
        return [x.__extract_mlir_values__()[0] for x in self.arrays]

    def __new_from_mlir_values__(self, values):
        return JaxArrayList(
            [JaxArray(v, x.shape, x.order) for x, v in zip(self.arrays, values)]
        )
