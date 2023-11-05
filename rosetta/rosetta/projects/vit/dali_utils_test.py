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

import tensorflow_datasets as tfds
from rosetta.data import dali, wds_utils
from rosetta.projects.vit.dali_utils import ViTPipeline


def iter_per_sec(dataset, batch_size: int = 1, num_iter: int | None = None):
    """
    Example Stats:
                 duration  num_examples         avg
    first+lasts  0.169234            15   88.634839
    first        0.103241             3   29.058269
    lasts        0.065993            12  181.837903
    """
    return tfds.benchmark(dataset, num_iter=num_iter, batch_size=batch_size).stats['avg']['lasts']

@pytest.mark.perf
@pytest.mark.data
def test_baseline_dali_iteration_stats(
    dummy_wds_metadata,
):
    """Computes dataset stats for a batched raw webdataset with cls/img elements"""
    img_shape = (dummy_wds_metadata.image_size, dummy_wds_metadata.image_size, dummy_wds_metadata.channels)

    config = wds_utils.WebDatasetConfig(
       urls=dummy_wds_metadata.path,
       batch_size=dummy_wds_metadata.batch_size,
       shuffle=False,
       seed=0,
       num_parallel_processes=1,
    )

    ds_shard_id = 0
    num_ds_shards = 1
    dataset = iter(dali.DALIIterator(ViTPipeline(config,
                                                 ds_shard_id,
                                                 num_ds_shards,
                                                 num_classes=dummy_wds_metadata.num_classes,
                                                 image_shape=img_shape)))

    bps = iter_per_sec(dataset, batch_size=dummy_wds_metadata.batch_size, num_iter=500)

    assert bps > (155 * 0.9)


def test_dali_cls_preprocessing(dummy_wds_metadata):
    config = wds_utils.WebDatasetConfig(
       urls=dummy_wds_metadata.path,
       batch_size=dummy_wds_metadata.batch_size,
       shuffle=False,
       seed=0,
       num_parallel_processes=1,
    )

    img_shape = (dummy_wds_metadata.image_size, dummy_wds_metadata.image_size, dummy_wds_metadata.channels)

    ds_shard_id = 0
    num_ds_shards = 1
    dataset = iter(dali.DALIIterator(
                         ViTPipeline(config,
                                     ds_shard_id,
                                     num_ds_shards,
                                     num_classes=dummy_wds_metadata.num_classes,
                                     image_shape=img_shape)))

    batch = next(dataset)
    class_labels = np.argmax(batch['labels'], -1)
    assert(np.array_equal(class_labels, np.arange(4)))
