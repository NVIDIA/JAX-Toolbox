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


class DummyPipeline(BaseDALIPipeline):

  def __init__(self,
               wds_config,
               shard_id,
               num_shards,
               num_classes,
               image_shape,
              ):

    modalities = [
       wds_utils.ModalityConfig(
            name='image',
            ftype='jpg',
            out_type='float32',
            shape=image_shape,
        ),
         wds_utils.ModalityConfig(
            name='label',
            ftype='cls',
            out_type='int',
            shape=(num_classes,),
        ),
    ]

    super().__init__(wds_config=wds_config,
                     modalities=modalities,
                     shard_id=shard_id,
                     num_shards=num_shards,
                     training=False)

  def get_wds_pipeline(self):
    @pipeline_def(batch_size=self.per_shard_batch_size, num_threads=1, device_id=None)
    def wds_pipeline():
      img, clss = fn.readers.webdataset(
          paths=self.urls,
          index_paths=self.index_paths,
          ext=[m.ftype for m in self.modalities],
          missing_component_behavior='error',
          random_shuffle=self.shuffle,
          shard_id=self.shard_id,
          num_shards=self.num_shards,
          pad_last_batch=False)
      return img, clss
    return wds_pipeline()

  ## non-image preprocessing
  def class_preproc(self, raw_text):
    bs = len(raw_text.shape())
    ascii = [np.asarray(raw_text[i]) for i in range(bs)]

    labels = np.zeros((bs, ))
    for i, el in enumerate(ascii):
      idx = int(bytes(el).decode('utf-8'))
      labels[i] = idx

    return labels

  def data_source(self):
    while True:
      img, clss = self.pipe.run()
      clss = self.class_preproc(clss)
      yield img, clss


  def get_dali_pipeline(self):
    @pipeline_def(batch_size=self.per_shard_batch_size, num_threads=self.num_workers, device_id=None)
    def main_pipeline():
      img, labels = fn.external_source(source=self.data_source(), num_outputs=2)

      img = fn.decoders.image(img, device='cpu', output_type=types.RGB)
      return img, labels

    return main_pipeline()

@pytest.mark.data
def test_baseline_dali_singleprocess_output(
    dummy_wds_metadata,
):
    img_shape = (dummy_wds_metadata.image_size, dummy_wds_metadata.image_size, dummy_wds_metadata.channels)

    config = wds_utils.WebDatasetConfig(
       urls=dummy_wds_metadata.path,
       batch_size=dummy_wds_metadata.batch_size,
       shuffle=False,
       seed=0,
    )

    ds_shard_id = 0
    num_ds_shards = 1

    pipe = DummyPipeline(config,
                         ds_shard_id,
                         num_ds_shards,
                         dummy_wds_metadata.num_classes,
                         img_shape).get_dali_pipeline()
    pipe.build()
    labels = []
    for _ in range(2):
      img, lab = pipe.run()
      labels.extend(lab.as_array())

    assert labels == list(range(8))


@pytest.mark.data
def test_baseline_dali_multiprocess_output(
    dummy_wds_metadata,
):
    img_shape = (dummy_wds_metadata.image_size, dummy_wds_metadata.image_size, dummy_wds_metadata.channels)

    config = wds_utils.WebDatasetConfig(
       urls=dummy_wds_metadata.path,
       batch_size=dummy_wds_metadata.batch_size,
       shuffle=False,
       seed=0,
    )

    ds_shard_id = 0
    num_ds_shards = 2

    first_proc_pipe = DummyPipeline(config,
                         ds_shard_id,
                         num_ds_shards,
                         dummy_wds_metadata.num_classes,
                         img_shape).get_dali_pipeline()
    first_proc_pipe.build()
    labels = []
    source = []
    for _ in range(2):
      img, lab = first_proc_pipe.run()
      labels.extend(lab.as_array())
      source +=  [l.source_info() for l in img]

    assert labels == list(range(4))
    assert (source[i].endswith(f'sample00000{i}.jpg') for i in range(len(source)))

    ds_shard_id = 1

    second_proc_pipe = DummyPipeline(config,
                         ds_shard_id,
                         num_ds_shards,
                         dummy_wds_metadata.num_classes,
                         img_shape).get_dali_pipeline()
    second_proc_pipe.build()
    labels = []
    source = []
    for _ in range(2):
      img, lab = second_proc_pipe.run()
      labels.extend(lab.as_array())
      source +=  [l.source_info() for l in img]

    assert (source[i].endswith(f'sample00000{40+i}.jpg') for i in range(len(source)))
