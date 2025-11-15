#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools as it
import math
from typing import List, Tuple

import jax

from .util import prime_factors


class PolymorphicMesh:
  def __init__(self, devices: List, primary_shape: Tuple[int], extra_singleton_dims: int = 0):
    assert len(devices) == math.prod(primary_shape), \
      f"Number of devices {len(devices)} must match {primary_shape}."
    self._devices = devices
    self._shape = tuple(it.chain(*[prime_factors(d) for d in primary_shape], [1] * extra_singleton_dims))
    self._axes = [f'd{i}' for i in range(len(self._shape))]
    self._mesh = jax.make_mesh(self._shape, self._axes, devices=devices)

  def __repr__(self):
    return f"{self.__class__.__name__}(shape={self._shape}, axes={self._axes})"

  @property
  def devices(self):
    return self._mesh.devices

  class PolymorphicMeshView:
    def __init__(self, mesh, axes_mapping):
      self._mesh = mesh
      self._axes_mapping = axes_mapping

    def __repr__(self):
      return f"{self.__class__.__name__}(shape={tuple(self._mesh.shape.values())}, axes={self._axes_mapping})"

    def axis(self, name):
      if name is None:
        return None
      else:
        try:
          return tuple([self._mesh.axis_names[i] for i in self._axes_mapping[name]])
        except KeyError as e:
          raise RuntimeError(f"Mesh has no axis named '{name}'. Available axes: {list(self._axes_mapping.keys())}.") from e

    def __enter__(self):
      return self._mesh.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
      return self._mesh.__exit__(exc_type, exc_value, traceback)

    # Delegated special methods
    def __str__(self):
      return str(self._mesh)

    def __eq__(self, other):
      if isinstance(other, PolymorphicMesh.PolymorphicMeshView):
        return self._mesh == other._mesh
      return self._mesh == other

    def __hash__(self):
      return hash(self._mesh)

    def __getattr__(self, name):
      return getattr(self._mesh, name)

    # Delegated properties to mirror jax.sharding.Mesh public interface
    @property
    def devices(self):
      return self._mesh.devices

    @property
    def axis_names(self):
      return self._mesh.axis_names

    @property
    def shape(self):
      return self._mesh.shape

    @property
    def shape_tuple(self):
      return self._mesh.shape_tuple

    @property
    def axis_sizes(self):
      return self._mesh.axis_sizes

    @property
    def size(self):
      return self._mesh.size

    @property
    def empty(self):
      return self._mesh.empty

    @property
    def is_multi_process(self):
      return self._mesh.is_multi_process

    @property
    def local_mesh(self):
      return self._mesh.local_mesh

    @property
    def device_ids(self):
      return self._mesh.device_ids

    @property
    def local_devices(self):
      return self._mesh.local_devices

    @property
    def abstract_mesh(self):
      return self._mesh.abstract_mesh

    @property
    def axis_types(self):
      return self._mesh.axis_types

  def view(self, shape, axis_names):
    assert math.prod(shape) == math.prod(self._shape), \
      f"Cannot reshape mesh of size {math.prod(self._shape)} to {shape}."
    assert len(shape) == len(axis_names), \
      f"Shape {shape} and axes internal_names {axis_names} must have the same length."

    axes_inventory = {size: [] for size in set(self._shape)}
    for i, size in enumerate(self._shape):
      axes_inventory[size].append(i)

    axes_mapping = {}
    for name, size in zip(axis_names, shape):
      sub_axes = []
      for p in prime_factors(size):
        try:
          sub_axes.append(axes_inventory[p].pop(0))
        except (IndexError, KeyError):
          raise RuntimeError(f"Cannot synthesize mesh dimension of size {size} due to no available prime factor dimension {p} for axis '{name}'.")
      axes_mapping[name] = tuple(sub_axes)

    for size, ids in axes_inventory.items():
      if size > 1 and len(ids) > 0:
        raise RuntimeError(f"Leftover mesh dimensions of size {size}: {ids}.")

    return self.PolymorphicMeshView(self._mesh, axes_mapping)
