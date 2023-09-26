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

from rosetta.data import dali, wds_utils
from rosetta.projects.vit.dali_utils import ViTPipeline
from functools import partial

from nvidia.dali.plugin.jax.clu import peekable_data_iterator


# Paths to validation dataset
wds_imagenet_url = '/home/awolant/Projects/datasets/webdataset/validation/{0..6}.tar'
index_dir_path= '/home/awolant/Projects/datasets/webdataset/validation/index/'


# This file contains tests for updated DALI pipelines used for ViT.
# The main goal is to compare the outputs of the new pipelines with the old ones.

#===================================================================================================================
# This section has all code necessary to run DALI pipelines for ViT in the updated approach.

import os
import numpy as np
import nvidia.dali.types as types
from nvidia.dali import fn, pipeline_def
from nvidia.dali.auto_aug import auto_augment

from braceexpand import braceexpand

from nvidia.dali.pipeline import Pipeline


def non_image_preprocessing(raw_text):      
    return np.array([int(bytes(raw_text).decode('utf-8'))])


def vit_pipeline_updated(wds_config, num_classes, image_shape, is_training=True, use_gpu=False):
    index_paths = [os.path.join(wds_config.index_dir, f) for f in os.listdir(wds_config.index_dir)] if wds_config.index_dir else None
    
    img, clss = fn.readers.webdataset(
        paths=list(braceexpand(wds_config.urls)),
        index_paths=index_paths,
        ext=['jpg', 'cls'],
        missing_component_behavior='error',
        random_shuffle=False,
        shard_id=0,
        num_shards=1,
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


def get_dali_pipeline_for_vit_updated(use_gpu=False):
    image_shape = (384,384,3)
    num_classes = 1000
    config = wds_utils.WebDatasetConfig(
        urls=wds_imagenet_url,
        index_dir=index_dir_path,
        batch_size=8,
        shuffle=False,
        seed=0,
        num_parallel_processes=1)
    
    pipeline = pipeline_def(
        vit_pipeline_updated,
        enable_conditionals=True,
        batch_size=config.batch_size,
        num_threads=config.num_parallel_processes,
        seed=0,
        device_id=0 if use_gpu else None)(
            wds_config=config,
            num_classes=num_classes,
            image_shape=image_shape,
            is_training=True)
    pipeline.build()
    
    return pipeline


def get_dali_dataset_updated(use_gpu=False):
    image_shape = (384,384,3)
    num_classes = 1000
    config = wds_utils.WebDatasetConfig(
        urls=wds_imagenet_url,
        index_dir=index_dir_path,
        batch_size=8,
        shuffle=False,
        seed=0,
        num_parallel_processes=1)
    
    iterator = peekable_data_iterator(
        vit_pipeline_updated,
        output_map=['images', 'labels'],
        reader_name='webdataset_reader')(
            enable_conditionals=True,
            batch_size=config.batch_size,
            num_threads=config.num_parallel_processes,
            seed=0,
            device_id=0 if use_gpu else None,
            wds_config=config,
            num_classes=num_classes,
            image_shape=image_shape,
            is_training=True)
    
    return iterator

#===================================================================================================================

def get_dali_pipeline_for_vit():
    "Gets DALI pipeline object for ViT how it is currently implemented."
    
    image_shape = (384,384,3)
    num_classes = 1000
    config = wds_utils.WebDatasetConfig(
        urls=wds_imagenet_url,
        index_dir=index_dir_path,
        batch_size=8,
        shuffle=False,
        seed=0,
        num_parallel_processes=1)
    
    pipeline = ViTPipeline(
        config,
        shard_id=0,
        num_shards=1,
        num_classes=num_classes,
        image_shape=image_shape,
        training=True,
        use_gpu=False,
        device_id=None).get_dali_pipeline()
    
    pipeline.build()
    return pipeline


def test_compare_dali_pipelines_outputs():
    pipeline = get_dali_pipeline_for_vit()
    pipeline_mono = get_dali_pipeline_for_vit_updated()
    
    for i in range(10):
        pipeline_out = pipeline.run()
        pipeline_mono_out = pipeline_mono.run()
        
        for j in range(len(pipeline_out)):
            assert np.array_equal(
                pipeline_out[j].as_array(),
                pipeline_mono_out[j].as_array())
            
            
            
def get_dali_dataset_configured():
    "Wrapper for get_dali_dataset that configures it for ViT. This configuration is normally done in gin files."
    
    image_shape = (384,384,3)
    num_classes = 1000
    config = wds_utils.WebDatasetConfig(
        urls=wds_imagenet_url,
        index_dir=index_dir_path,
        batch_size=8,
        shuffle=False,
        seed=0,
        num_parallel_processes=1)
    
    vit_pipeline_configured = partial(ViTPipeline, num_classes=num_classes, image_shape=image_shape)
    iterator = dali.get_dali_dataset(config, 0, 1, None, vit_pipeline_configured)
    return iterator


def compare_iterators_outpus(out, out_rosetta):
    for key in out.keys():
        assert np.array_equal(
                out[key],
                out_rosetta[key])    
    
        
def test_comapre_dali_rosetta_dataset_with_dali_peekable_iterator():
    iterator = get_dali_dataset_updated()
    iterator_rosetta = get_dali_dataset_configured()
    
    iterator_element_spec = iterator.element_spec
    iterator_rosetta_element_spec = iterator_rosetta.element_spec
    
    assert iterator_element_spec == iterator_rosetta_element_spec
    
    for i in range(10):
        peeked_out = iterator.peek()
        peeked_roseeta_out = iterator_rosetta.peek()
        
        compare_iterators_outpus(peeked_out, peeked_roseeta_out)
        
        out = iterator.next()
        out_rosetta = next(iterator_rosetta)
        
        compare_iterators_outpus(out, out_rosetta)
        compare_iterators_outpus(out, peeked_out)