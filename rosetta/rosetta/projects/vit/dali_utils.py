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

def non_image_preprocessing(raw_text):
  """ preprocessing of class labels. """
  return np.array([int(bytes(raw_text).decode('utf-8'))])


class ViTPipeline(BaseDALIPipeline):

  ## all pipelines are extpected to take wds_config, per_shard_batch_size into constructor
  def __init__(self,
               wds_config,
               shard_id,
               num_shards,
               num_classes,
               image_shape,
               training=True,
              ):

    self.num_classes = num_classes
    self.image_shape = image_shape
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


  def get_dali_pipeline(self):

    ## need to enable conditionals for auto-augment
    @pipeline_def(batch_size=self.per_shard_batch_size, num_threads=self.num_workers, device_id=None, enable_conditionals=True, seed=self.seed, prefetch_queue_depth=self.prefetch)
    def main_vit_pipeline():
      jpegs, clss = fn.readers.webdataset(
        paths=self.urls,
        index_paths=self.index_paths,
        ext=[m.ftype for m in self.modalities],
        missing_component_behavior='error',
        random_shuffle=self.shuffle,
        shard_id=self.shard_id,
        num_shards=self.num_shards,
        pad_last_batch=False if self.training else True)
      img = fn.decoders.image(jpegs, device='cpu', output_type=types.RGB)
      
      labels = fn.python_function(clss, function=non_image_preprocessing, num_outputs=1)
      labels = fn.one_hot(labels, num_classes=self.num_classes)

      if self.training:
        img = fn.random_resized_crop(img, size=self.image_shape[:-1], seed=self.seed)
        img = fn.flip(img, depthwise=0, horizontal=fn.random.coin_flip(seed=self.seed))

        ## color jitter
        brightness = fn.random.uniform(range=[0.6,1.4], seed=self.seed)
        contrast = fn.random.uniform(range=[0.6,1.4], seed=self.seed)
        saturation = fn.random.uniform(range=[0.6,1.4], seed=self.seed)
        hue = fn.random.uniform(range=[0.9,1.1], seed=self.seed)
        img = fn.color_twist(img,
                             brightness=brightness,
                             contrast=contrast,
                             hue=hue,
                             saturation=saturation)

        ## auto-augment
        ## `shape` controls the magnitude of the translation operations
        img = auto_augment.auto_augment_image_net(img, seed=self.seed)

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

