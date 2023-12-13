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
import time

from nvidia.dali.plugin.jax.clu import peekable_data_iterator

from jax.experimental import multihost_utils

import jax

import os
import numpy as np
from braceexpand import braceexpand

import nvidia.dali.types as types
from nvidia.dali import fn, pipeline_def
from nvidia.dali.auto_aug import auto_augment


def non_image_preprocessing(raw_text):      
    return np.array([int(bytes(raw_text).decode('utf-8'))])


def vit_pipeline(wds_config, num_classes, image_shape, shard_id=0, num_shards=1, is_training=True, use_gpu=False):
    index_paths = [os.path.join(wds_config.index_dir, f) for f in os.listdir(wds_config.index_dir)] if wds_config.index_dir else None

    img, clss = fn.readers.webdataset(
        paths=list(braceexpand(wds_config.urls)),
        index_paths=index_paths,
        ext=['jpg', 'cls'],
        missing_component_behavior='error',
        random_shuffle=False,
        shard_id=shard_id,
        num_shards=num_shards,
        pad_last_batch=False if is_training else True,
        name='webdataset_reader')

    labels = fn.python_function(clss, function=non_image_preprocessing, num_outputs=1)
    if use_gpu:
        labels = labels.gpu()
    labels = fn.one_hot(labels, num_classes=num_classes)

    device = 'mixed' if use_gpu else 'cpu'
    img = fn.decoders.image(img, device=device, output_type=types.RGB)

    if is_training:
        img = fn.random_resized_crop(img, size=image_shape[:-1])
        img = fn.flip(img, depthwise=0, horizontal=fn.random.coin_flip())

        # color jitter
        brightness = fn.random.uniform(range=[0.6,1.4])
        contrast = fn.random.uniform(range=[0.6,1.4])
        saturation = fn.random.uniform(range=[0.6,1.4])
        hue = fn.random.uniform(range=[0.9,1.1])
        img = fn.color_twist(
            img,
            brightness=brightness,
            contrast=contrast,
            hue=hue,
            saturation=saturation)

        # auto-augment
        # `shape` controls the magnitude of the translation operations
        img = auto_augment.auto_augment_image_net(img)
    else:
        img = fn.resize(img, size=image_shape[:-1])

    ## normalize
    ## https://github.com/NVIDIA/DALI/issues/4469
    mean = np.asarray([0.485, 0.456, 0.406])[None, None, :]
    std = np.asarray([0.229, 0.224, 0.225])[None, None, :]
    scale = 1 / 255.
    img = fn.normalize(
        img,
        mean=mean / scale,
        stddev=std,
        scale=scale,
        dtype=types.FLOAT)

    return img, labels


def get_dali_dataset(
    config,
    ds_shard_id,
    ds_num_shards,
    feature_converter,
    total_steps
):  
    assert not bool(feature_converter), 'Passing `feature_converter_cls` is not supported'
    
    # Note: DALI iterator for multiple GPUs has to now the sharding during the initialization.
    # Therefore, we have to create it in prepare_dali_iterator (conntected to prepare_train_iter_fn in config).
    
    # TODO(awolant): Implement support for multiple hosts
    # assert ds_shard_id == 0, 'DALI does not support multi host training'
    # assert ds_num_shards == 1, 'DALI does not support multi host training'
    
    # For multihost training we have ds_num_shards > 1 and ds_shard_id points to the shard id of the current host.
    # For single host training we have ds_num_shards == 1 and ds_shard_id == 0.
    
    seed = config.seed
    if seed is None:
        # Use a shared timestamp across devices as the seed.
        seed = int(multihost_utils.broadcast_one_to_all(np.int32(time.time())))
    config.seed = seed
    
    print("Local devices: ", jax.local_devices())
    
    
    return (config, ds_shard_id, ds_num_shards, total_steps, seed)


def prepare_dali_iterator(
    iterator_args,
    partitioner,
    checkpoint_cfg,
    data_layout
):
    config, ds_shard_id, ds_num_shards, total_steps, seed = iterator_args
    
    num_classes = 1000
    image_shape = (384,384,3)
    use_gpu = True
    is_training = True
    
    global_mesh = partitioner.mesh
    axes = partitioner.data_partition_spec
    local_devices = global_mesh.local_devices
    local_device_count = jax.local_device_count()
    sharding = jax.sharding.NamedSharding(global_mesh, axes)
    
    # TODO(awolant): Implement support for multiple hosts 
    # jax.process_index() is the index of the current host
    
    # TODO(awolant): Add index support

    iterator = peekable_data_iterator(
        vit_pipeline,
        output_map=['images', 'labels'],
        auto_reset=True,
        size=total_steps*config.batch_size,
        sharding=sharding,
        )(
            enable_conditionals=True,
            batch_size=config.batch_size,
            num_threads=config.num_parallel_processes,
            seed=seed,
            use_gpu=use_gpu,
            wds_config=config,
            # num_shards=ds_num_shards,
            # shard_id=ds_shard_id,
            num_classes=num_classes,
            image_shape=image_shape,
            is_training=is_training)

    return iterator