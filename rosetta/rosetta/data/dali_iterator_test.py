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
from nvidia.dali import fn, pipeline_def

from rosetta.data import dali
from rosetta.data import dali_iterator
from rosetta.data import wds_utils
from rosetta.projects.vit.dali_utils import ViTPipeline


import jax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding

import jax.numpy as jnp

from rosetta.data.dali_iterator import vit_pipeline
from nvidia.dali.plugin.jax.clu import peekable_data_iterator

import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali import fn

from braceexpand import braceexpand
import os

from t5x import partitioning
from functools import partial

eval_urls = '/opt/rosetta/datasets/imagenet/imagenet-val-{000000..000006}.tar'
eval_index_path = '/opt/rosetta/datasets/imagenet/index/eval/'

testing_dataset_urls = '/opt/rosetta/dev/dataset/dataset.tar'

train_urls = '/opt/rosetta/datasets/imagenet/imagenet-train-{000000..000146}.tar'

image_shape = (384, 384, 3)
num_classes = 1000

num_iter = 14


config = wds_utils.WebDatasetConfig(
  urls=testing_dataset_urls,
  batch_size=6,
  shuffle=False,
  seed=0,
  num_parallel_processes=16,
  prefetch=2,
  # index_dir=eval_index_path,
)

ds_shard_id = 0
num_ds_shards = 1
total_steps = num_iter
seed = config.seed

use_gpu = True
is_training = False

partitioner = partitioning.PjitPartitioner(
  num_partitions=1,
  model_parallel_submesh=None,
  logical_axis_rules=partitioning.standard_logical_axis_rules()
)

data_layout = partitioner.get_data_layout(config.batch_size)
ds_shard_id = data_layout.shard_id
num_ds_shards = data_layout.num_shards


def get_dali_dataset_configured(config, image_shape, num_classes, ds_shard_id, num_ds_shards):
  "Wrapper for get_dali_dataset that configures it for ViT. This configuration is normally done in gin files."

  vit_pipeline_configured = partial(ViTPipeline, num_classes=num_classes, image_shape=image_shape, training=is_training)
  iterator = dali.get_dali_dataset(config, ds_shard_id, num_ds_shards, None, vit_pipeline_configured)

  return iterator


def test_sharded_dataset_iterator_for_vit():
  iterator = get_dali_dataset_configured(config, image_shape, num_classes, ds_shard_id, num_ds_shards)
  
  sharded_iterator = dali.create_sharded_iterator(
    iterator, partitioner, None, data_layout)
  
  # TODO(awolant): Why is_nonpadding is not working?
  # Because it is working only for eval pipeline.

  for _ in range(num_iter):
    batch = next(sharded_iterator)
    padding = sharded_iterator.is_nonpadding
    print(padding, type(padding), padding.shape, padding.sharding)
    
    if not jnp.all(padding):
      print("Padding is not all True")


def test_peekable_dataset_iterator_for_vit_eval():
  iterator_args = dali_iterator.get_dali_dataset(
    config, ds_shard_id, num_ds_shards, None, total_steps, is_training)
  
  sharded_iterator = dali_iterator.prepare_dali_iterator(
    iterator_args, partitioner, None, data_layout)
  
  for batch in sharded_iterator:
    # TODO(awolant): Implement this
    padding = sharded_iterator.is_nonpadding
    
    print(padding, type(padding), padding.shape, padding.sharding)
    
    # if not jnp.all(padding):
    #   print("Padding is not all True")


# def test_composite_pipeline_on_validation_dataset():
#     pipeline = ViTPipeline(
#         config,
#         shard_id=0,
#         num_shards=1,
#         num_classes=num_classes,
#         image_shape=image_shape,
#         training=True).get_dali_pipeline()

#     pipeline.build()
    
#     for i in range(num_iter):
#       out = pipeline.run()


# def test_minimal_pipeline_on_validation_dataset():
#   index_paths = (
#         [
#             os.path.join(config.index_dir, f)
#             for f in os.listdir(config.index_dir)
#         ]
#         if config.index_dir
#         else None
#     )
  
  
#   @pipeline_def(
#     batch_size=config.batch_size, num_threads=config.num_parallel_processes, device_id=0)
#   def minimal_pipeline():
#     img, clss = fn.readers.webdataset(
#         paths=list(braceexpand(config.urls)),
#         index_paths=index_paths,
#         ext=['jpg', 'cls'],
#         missing_component_behavior='error',
#         random_shuffle=config.shuffle,
#         shard_id=0,
#         num_shards=1,
#         pad_last_batch=False,
#         name="reader")
    
#     device = "mixed" if use_gpu else "cpu"
#     img = fn.decoders.image(img, device=device, output_type=types.RGB)
    
#     return img, clss
  
#   pipeline = minimal_pipeline()
#   pipeline.build()
  
#   pipeline.epoch_size("reader")
  
#   for i in range(num_iter):
#     pipeline.run()
  

# def test_dali_iterator():
#   global_mesh = Mesh(jax.devices(), axis_names=('data'))
#   sharding = NamedSharding(global_mesh, PartitionSpec('data'))
  
#   iterator = peekable_data_iterator(
#         vit_pipeline,
#         output_map=["images", "labels"],
#         auto_reset=True,
#         size=total_steps * config.batch_size,
#         sharding=sharding,
#     )(
#         batch_size=config.batch_size,
#         num_threads=config.num_parallel_processes,
#         seed=seed,
#         use_gpu=use_gpu,
#         wds_config=config,
#         num_classes=num_classes,
#         image_shape=image_shape,
#         is_training=is_training,
#         enable_conditionals=True
#     )
    

#   for batch in iterator:
#     pass
