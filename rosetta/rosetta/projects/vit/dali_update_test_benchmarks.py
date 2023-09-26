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

import pytest
from dali_update_test import get_pure_dali_pipeline_for_vit, get_dali_pipeline_for_vit_updated, get_dali_peekable_iterator_for_vit_mono, get_dali_peekable_iterator_for_vit_mono, get_dali_dataset_configured

      
@pytest.mark.benchmark 
def test_benchmark_for_vit_pipeline(benchmark):
    pipeline = get_pure_dali_pipeline_for_vit()
    
    @benchmark
    def run():
        for i in range(10):
            pipeline.run()

        
@pytest.mark.benchmark
def test_benchmark_for_vit_pipeline_mono(benchmark):
    pipeline = get_dali_pipeline_for_vit_updated()
    
    @benchmark
    def run():
        for i in range(10):
            pipeline.run()
    
@pytest.mark.benchmark
def test_benchamrk_for_vit_pipeline_mono_gpu(benchmark):
    pipeline = get_dali_pipeline_for_vit_updated(use_gpu=True)
    
    @benchmark
    def run():
        for i in range(10):
            pipeline.run()
        
        
def test_benchmark_for_dali_peekable_iterator(benchmark):
    iterator = get_dali_peekable_iterator_for_vit_mono()
    
    @benchmark
    def run():
        for i in range(10):
            out = iterator.next()
            
            
def test_benchmark_for_dali_rosetta_dataset(benchmark):
    iterator = get_dali_dataset_configured()
    
    @benchmark
    def run():
        for i in range(10):
            out = next(iterator)
            
            
def test_benchmark_for_dali_peekable_iterator_gpu(benchmark):
    iterator = get_dali_peekable_iterator_for_vit_mono(use_gpu=True)
    
    @benchmark
    def run():
        for i in range(10):
            out = iterator.next()