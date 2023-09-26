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
from nvidia.dali.auto_aug import auto_augment

from rosetta.data.dali import BaseDALIPipeline
from rosetta.data.wds_utils import ModalityConfig


class ViTPipeline(BaseDALIPipeline):

  ## all pipelines are extpected to take wds_config, per_shard_batch_size into constructor
  def __init__(self,
               wds_config,
               shard_id,
               num_shards,
               num_classes,
               image_shape,
               training=True,
               use_gpu=False,
               device_id=None
              ):

    self.num_classes = num_classes
    self.image_shape = image_shape
    self._use_gpu = use_gpu
    self._device_id = None if not use_gpu else device_id
    modalities = [ModalityConfig(name='images',
                                 ftype='jpg',
                                 out_type='float32',
                                 shape=self.image_shape),
                  ModalityConfig(name='labels',
                                 ftype='cls',
                                 out_type='float32',
                                 shape=(self.num_classes,))]

    super().__init__(wds_config=wds_config,
                     modalities=modalities,
                     shard_id=shard_id,
                     num_shards=num_shards,
                     training=training)


  def get_wds_pipeline(self):
    @pipeline_def(batch_size=self.per_shard_batch_size, num_threads=1, device_id=None, seed=self.seed)
    def wds_vit_pipeline():
      ## assumes a particular order to the ftypes
      img, clss = fn.readers.webdataset(
        paths=self.urls,
        index_paths=self.index_paths,
        ext=[m.ftype for m in self.modalities],
        missing_component_behavior='error',
        random_shuffle=self.shuffle,
        shard_id=self.shard_id,
        num_shards=self.num_shards,
        pad_last_batch=False if self.training else True)
      return img, clss
    return wds_vit_pipeline()

  def non_image_preprocessing(self, raw_text, num_classes):
    """ preprocessing of class labels. """
    bs = len(raw_text.shape())
    ascii = [np.asarray(raw_text[i]) for i in range(bs)]

    one_hot = np.zeros((bs, num_classes))
    for i, el in enumerate(ascii):
      idx = int(bytes(el).decode('utf-8'))
      one_hot[i][idx] = 1

    return one_hot

  def data_source(self, num_classes):
    while True:
      preprocessed_img, raw_text = self.pipe.run()
      preprocessed_label = self.non_image_preprocessing(raw_text, num_classes)
      yield preprocessed_img, preprocessed_label


  def get_dali_pipeline(self):

    ## need to enable conditionals for auto-augment
    @pipeline_def(batch_size=self.per_shard_batch_size, num_threads=self.num_workers, device_id=self._device_id, enable_conditionals=True, seed=self.seed, prefetch_queue_depth=self.prefetch)
    def main_vit_pipeline():
      jpegs, labels = fn.external_source(source=self.data_source(self.num_classes), num_outputs=2)
      
      
      device = 'mixed' if self._use_gpu else 'cpu'
      img = fn.decoders.image(jpegs, device=device, output_type=types.RGB)

      if self.training:
        img = fn.random_resized_crop(img, size=self.image_shape[:-1])
        img = fn.flip(img, depthwise=0, horizontal=fn.random.coin_flip())

        # color jitter
        brightness = fn.random.uniform(range=[0.6,1.4])
        contrast = fn.random.uniform(range=[0.6,1.4])
        saturation = fn.random.uniform(range=[0.6,1.4])
        hue = fn.random.uniform(range=[0.9,1.1])
        img = fn.color_twist(img,
                             brightness=brightness,
                             contrast=contrast,
                             hue=hue,
                             saturation=saturation)

        # auto-augment
        # `shape` controls the magnitude of the translation operations
        img = auto_augment.auto_augment_image_net(img)

      else:
        img = fn.resize(img, size=self.image_shape[:-1])

      ## normalize
      ## https://github.com/NVIDIA/DALI/issues/4469
      mean = np.asarray([0.485, 0.456, 0.406])[None, None, :]
      std = np.asarray([0.229, 0.224, 0.225])[None, None, :]
      scale = 1 / 255.
      img = fn.normalize(img,
          mean=mean / scale,
          stddev=std,
          scale=scale,
          dtype=types.FLOAT)

      return img, labels

    return main_vit_pipeline()
