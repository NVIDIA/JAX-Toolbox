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

import numpy as np
import nvidia.dali.types as types
import pytest
from nvidia.dali import fn, pipeline_def

from rosetta.data import wds_utils
from rosetta.data.dali import BaseDALIPipeline

import jax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils

from rosetta.data.dali_iterator import vit_pipeline
from nvidia.dali.plugin.jax.clu import peekable_data_iterator



def test_dali_iterator():
  config = wds_utils.WebDatasetConfig(
    urls='/opt/rosetta/datasets/imagenet/imagenet-train-{000000..000146}.tar',
    batch_size=128,
    shuffle=False,
    seed=0,
    num_parallel_processes=16,
    prefetch=2,
    index_dir='/opt/rosetta/datasets/imagenet/index/train/'
  )

  ds_shard_id = 0
  num_ds_shards = 1
  total_steps = 10
  seed = config.seed

  num_classes = 1000
  image_shape = (384, 384, 3)
  use_gpu = False
  is_training = False

  global_mesh = Mesh(jax.devices(), axis_names=('data'))
  sharding = NamedSharding(global_mesh, PartitionSpec('data'))
  
  iterator = peekable_data_iterator(
        vit_pipeline,
        output_map=["images", "labels"],
        auto_reset=True,
        size=total_steps * config.batch_size,
        sharding=sharding,
    )(
        batch_size=config.batch_size,
        num_threads=config.num_parallel_processes,
        seed=seed,
        use_gpu=use_gpu,
        wds_config=config,
        num_classes=num_classes,
        image_shape=image_shape,
        is_training=is_training,
        enable_conditionals=True
    )
    
  i = 0
  for batch in iterator:
    print(i)
    i = i + 1
