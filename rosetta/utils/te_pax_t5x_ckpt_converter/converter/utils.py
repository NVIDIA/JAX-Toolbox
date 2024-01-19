# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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


import asyncio
import os
import shutil
from dataclasses import dataclass

from etils import epath
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.array_serialization import serialization
from jax.experimental.array_serialization.serialization import get_tensorstore_spec

# Use host to convert tensors for leveraging the larger host memory.
jax.config.update('jax_platform_name', 'cpu')

MESH = jax.sharding.Mesh(devices=mesh_utils.create_device_mesh((1,)), axis_names=('no_shard',))
SHARDING = jax.sharding.NamedSharding(MESH, jax.sharding.PartitionSpec())


@dataclass
class ModelConfig:
    num_of_layer: int
    embed_dim: int
    num_of_head: int
    head_dim: int
    mlp_intermediate_dim: int
    kernel_chunk_size: int = None


@dataclass
class ConvertPkg:
    target_path: (str | list[str])
    shape: tuple
    chunk_dim: int
    converters: tuple
    extra_src_paths: list[str]
    stack_dim: int
    just_copy: bool


class ConvertHelper:

    def __init__(self, input_path: str, output_path: str, model_config: ModelConfig,
                 weight_only: bool, skip_ln: bool):
        self.input_path = input_path
        self.output_path = output_path
        self.model_config = model_config
        self.weight_only = weight_only
        self.skip_ln = skip_ln

    @property
    def catagories(self):
        raise NotImplementedError

    def _get_convert_pkg(self,
                         target_path,
                         shape,
                         chunk_dim,
                         *converters,
                         extra_src_paths=[],
                         stack_dim=0,
                         just_copy=False):
        return ConvertPkg(target_path, shape, chunk_dim, tuple(converters), extra_src_paths,
                          stack_dim, just_copy)

    def _unpack_convert_pkg(self, pkg):
        return pkg.target_path, pkg.shape, pkg.chunk_dim, pkg.converters, \
               pkg.extra_src_paths, pkg.stack_dim, pkg.just_copy

    def _generate_ckpt_map(self):
        raise NotImplementedError

    def generate_ckpt_map_with_full_name(self):
        ckpt_map = self._generate_ckpt_map()

        def is_ln_weights(key):
            if "layer_norm" in key:
                return True
            if "scale" in key:
                return True
            if "ln_bias" in key:
                return True
            return False

        if self.skip_ln:
            for key in ckpt_map:
                if is_ln_weights(key):
                    ckpt_map.pop(key)

        ckpt_map_with_full_name = {}
        for prefix in self.catagories:
            for src_path in ckpt_map:
                ckpt_pkgs = ckpt_map[src_path]
                if not isinstance(ckpt_pkgs, list):
                    ckpt_pkgs = [ckpt_pkgs]

                ckpt_pkgs_with_full_name = []
                for pkg in ckpt_pkgs:
                    target_path, shape, chunk_dim, converters, \
                    extra_src_paths, stack_dim, just_copy = self._unpack_convert_pkg(pkg)

                    full_src_name = prefix + '.' + src_path
                    full_target_name = prefix + '.' + target_path
                    full_extra_src_names = None if extra_src_paths is None else \
                                           [prefix + '.'+ esp for esp in extra_src_paths]

                    ckpt_pkgs_with_full_name.append(
                        self._get_convert_pkg(full_target_name,
                                              shape,
                                              chunk_dim,
                                              *converters,
                                              extra_src_paths=full_extra_src_names,
                                              stack_dim=stack_dim,
                                              just_copy=just_copy))

                ckpt_map_with_full_name[full_src_name] = ckpt_pkgs_with_full_name

        return ckpt_map_with_full_name

    def convert(self):
        ckpt_map_with_full_path = self.generate_ckpt_map_with_full_name()

        skip_pool = set()
        for folder in ckpt_map_with_full_path:
            src_path = os.path.join(self.input_path, folder)
            assert os.path.exists(src_path), \
                f"{src_path} does not exist."

            for ckpt_pkg in ckpt_map_with_full_path[folder]:
                target_path, shape, chunk_dim, converters, \
                extra_src_paths, stack_dim, just_copy = self._unpack_convert_pkg(ckpt_pkg)

                if just_copy:
                    src_path = os.path.join(self.input_path, folder)
                    target_path = os.path.join(self.output_path, target_path)
                    copy_ckpt(src_path, target_path)
                else:
                    target_path = os.path.join(self.output_path, target_path)

                    jnp_arrs = []
                    for src in [folder, *extra_src_paths]:
                        skip_pool.add(src)
                        src_path = os.path.join(self.input_path, src)
                        jnp_arrs.append(serialize_tensor(src_path, shape))

                    if len(jnp_arrs) == 1:
                        jnp_arr = jnp_arrs[0]
                    else:
                        jnp_arr = jnp.stack(jnp_arrs, axis=stack_dim)

                    for converter in converters:
                        jnp_arr = converter(jnp_arr)

                    deserialize_tensor(target_path, jnp_arr, chunk_dim,
                                       self.model_config.kernel_chunk_size)

        for folder in os.listdir(self.input_path):
            if folder not in ckpt_map_with_full_path and folder not in skip_pool:
                src_path = os.path.join(self.input_path, folder)
                target_path = os.path.join(self.output_path, folder)
                copy_ckpt(src_path, target_path)


def copy_ckpt(src_path, target_path):
    copy_fn = shutil.copytree if os.path.isdir(src_path) else shutil.copyfile
    copy_fn(src_path, target_path)


def get_json_tspec(path):
    """Gets Tensorstore spec in JSON format."""
    path = os.fspath(path)
    tspec = get_tensorstore_spec(path, ocdbt=False)
    return tspec


def get_cast_tspec_deserialize(tspec, dtype):
    """Creates a Tensorstore spec for casting a param during deserialize."""
    if dtype is not None:
        tspec = {
            'base': tspec,
            'driver': 'cast',
            'dtype': jnp.dtype(dtype).name,
        }
    return tspec


def get_cast_tspec_serialize(tspec, value):
    """Creates a Tensorstore spec for casting a param during serialize."""
    tspec = {
        'base': tspec,
        'driver': 'cast',
    }
    # Origin dtype.
    tspec['dtype'] = jnp.dtype(value.dtype).name
    # Destination dtype.
    tspec['base']['dtype'] = jnp.dtype(value.dtype).name
    return tspec


def serialize_tensor(path: str, shape: tuple, dtype=jnp.float32):
    path = epath.Path(path)

    tspec = get_json_tspec(path)
    tspec = get_cast_tspec_deserialize(tspec, dtype)
    jnp_arr = asyncio.run(serialization.async_deserialize(SHARDING, tspec, global_shape=shape))
    return jnp_arr


def deserialize_tensor(path: str, tensor: jnp.ndarray, chunk_dim: int = None, chunk_size=None):
    path = epath.Path(path)

    tspec = get_json_tspec(path)
    tspec['metadata'] = serialization._get_metadata(tensor)
    del tspec['metadata']['dtype']

    if chunk_dim is not None:
        chunk_shape = tuple([
            chunk_size if (i == chunk_dim and chunk_size is not None) else s
            for i, s in enumerate(tensor.shape)
        ])
        tspec['metadata']['chunks'] = chunk_shape

    tspec = get_cast_tspec_serialize(tspec, tensor)
    serialization.run_serialization([tensor], [tspec])
