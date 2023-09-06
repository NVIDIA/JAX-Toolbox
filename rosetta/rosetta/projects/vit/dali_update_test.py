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
import pytest


from rosetta.data import dali, wds_utils
from rosetta.projects.vit.dali_utils import ViTPipeline, ViTPipelineMono
from functools import partial

from nvidia.dali.plugin.jax import DALIGenericPeekableIterator as DALIIterator


wds_imagenet_url = '/home/awolant/Projects/datasets/webdataset/validation/{0..6}.tar'
index_dir_path= '/home/awolant/Projects/datasets/webdataset/validation/index/'


def test_dali_dataset_iterator():
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
        device_id=None)

    # Ten iterator jest opakowaniem na pipeline i ma być kompatybilny z CLU PeekableDatasetIterator
    # https://github.com/google/CommonLoopUtils
    iterator = dali.DALIIterator(pipeline)

    for id, batch in enumerate(iterator):    
        if id == 10:
            break
  

def get_dali_rosetta_dataset():
    image_shape = (384,384,3)
    num_classes = 1000
    config = wds_utils.WebDatasetConfig(
        urls=wds_imagenet_url,
        index_dir=index_dir_path,
        batch_size=128,
        shuffle=False,
        seed=0,
        num_parallel_processes=1)
    
    vit_pipeline_configured = partial(ViTPipeline, num_classes=num_classes, image_shape=image_shape)
    iterator = dali.get_dali_dataset(config, 0, 1, None, vit_pipeline_configured)
    return iterator
  
          
def test_dali_get_dali_dataset():    
    # Właścieiwe API używanym do stworzenia iteratora jest chyba
    # Żeby to działało w Pythonie trzeba sobie zrobić dodatkową konfigurację, bo normalnie te argumenty
    # są ustawiane w plikach konfiguracyjnych .gin. 
    iterator = get_dali_rosetta_dataset()
    
    for id, batch in enumerate(iterator):    
        if id == 10:
            break
        
    # BaseDALIPipeline to jest abstrakcyjny interfejs, który ma być opakowaniem na DALI pipeline
    # Trzyma trochę rzeczy związanych z konfiguracją (czy training czy eval, jaki shard itd.) i ma mieć funkcję
    # która produkuje pipeliney. Ta funkcja musi uzwzględnić tą dziwną zależność, że są potrzebne dwa
    # ViT pipeline to jest przykład implementacji tego
    
    # Ten podział na pipeline jest dlatego, że wds ma labele jako stringi. Trzeba je przekonwertować do intów.
    # Wtedy drugi pipeline ma external source, gdzie woła ten pierwszy


def get_pure_dali_pipeline_for_vit():
    image_shape = (384,384,3)
    num_classes = 1000
    config = wds_utils.WebDatasetConfig(
        urls=wds_imagenet_url,
        index_dir=index_dir_path,
        batch_size=128,
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


def get_pure_dali_pipeline_for_vit_mono(use_gpu=False):
    image_shape = (384,384,3)
    num_classes = 1000
    config = wds_utils.WebDatasetConfig(
        urls=wds_imagenet_url,
        index_dir=index_dir_path,
        batch_size=128,
        shuffle=False,
        seed=0,
        num_parallel_processes=1)
    
    pipeline = ViTPipelineMono(
        config,
        shard_id=0,
        num_shards=1,
        num_classes=num_classes,
        image_shape=image_shape,
        training=True,
        use_gpu=use_gpu,
        device_id=0 if use_gpu else None).get_dali_pipeline()
    
    pipeline.build()
    
    return pipeline


def get_dali_peekable_iterator_for_vit_mono(use_gpu=False):
    pipeline = get_pure_dali_pipeline_for_vit_mono(use_gpu=use_gpu)
    
    iterator = DALIIterator(
        pipelines=[pipeline],
        output_map=['images', 'labels'],
        reader_name='webdataset_reader')
    
    return iterator
    

def test_dali_vit_pipeline_only(): 
    pipeline = get_pure_dali_pipeline_for_vit()
    
    for i in range(10):
        pipeline.run()
        

def test_vit_pipeline_mono():
    pipeline = get_pure_dali_pipeline_for_vit_mono()
    
    for i in range(10):
        out = pipeline.run()
        out[0].source_info()
        

    
def test_compare_dali_pipeline_outputs():
    pipeline = get_pure_dali_pipeline_for_vit()
    pipeline_mono = get_pure_dali_pipeline_for_vit_mono()
    
    for i in range(10):
        pipeline_out = pipeline.run()
        pipeline_mono_out = pipeline_mono.run()
        
        for j in range(len(pipeline_out)):
            assert np.array_equal(
                pipeline_out[j].as_array(),
                pipeline_mono_out[j].as_array())


def test_dali_peekable_iterator():
    iterator = get_dali_peekable_iterator_for_vit_mono()
    
    out = iterator.peek()
    
    for i in range(10):
        out = iterator.next()
    

def compare_iterators_outpus(out, out_rosetta):
    for key in out.keys():
        assert np.array_equal(
                out[key],
                out_rosetta[key])    
    
        
def test_comapre_dali_rosetta_dataset_with_dali_peekable_iterator():
    iterator = get_dali_peekable_iterator_for_vit_mono()
    iterator_rosetta = get_dali_rosetta_dataset()
    
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
