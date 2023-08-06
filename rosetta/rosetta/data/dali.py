# Copyright (c) 2023, The T5X Authors.
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

import abc
import os
import re
import threading
import time
from typing import Any

import jax
import numpy as np
from braceexpand import braceexpand
from clu import asynclib
from clu.data.dataset_iterator import ArraySpec, Element, ElementSpec
from jax.experimental import multihost_utils

from rosetta.data import wds_utils
from t5x import partitioning, utils


PyTree = Any

def type_proc(dtype:str):
  if dtype == 'float32':
    return np.float32
  elif dtype == 'int':
    return np.int32
  elif dtype == 'float16':
    return np.float16
  elif dtype == 'bfloat16':
    return jax.numpy.bfloat16
  else:
    raise ValueError('Could not parse dtype: %s' % dtype)

class BaseDALIPipeline(abc.ABC):

  def __init__(self,
               wds_config: wds_utils.WebDatasetConfig,
               modalities: wds_utils.ModalityConfig,
               shard_id: int,
               num_shards: int,
               training: bool=True):

    """Abstract class for defining a DALI pipeline for t5x models.

       Attributes:
         wds_config: a WebDatasetConfig instance containing the dataloading configuration. See `wds_utils.py` for more information
         modalities: a ModalityConfig instance containing information about the modalities present in the dataset. See `wds_utils.py` for more information
         shard_id: dataset shard index
         num_shards: number of dataset shards
         training: whether data is being loaded in training or evaluation mode.
    """

    index_dir = wds_config.index_dir
    index_paths = [os.path.join(index_dir, f) for f in os.listdir(index_dir)] if index_dir else None
    self.index_paths = sorted(index_paths, key=lambda x: int(re.split('_|\\.', x)[-2])) if index_paths else None

    self.urls = list(braceexpand(wds_config.urls))
    self.modalities = modalities
    self.shard_id = shard_id
    self.num_shards = num_shards
    self.seed = wds_config.seed
    self.per_shard_batch_size = wds_config.batch_size // num_shards
    self.shuffle = wds_config.shuffle
    self.num_workers = wds_config.num_parallel_processes
    self.prefetch = wds_config.prefetch
    self.training = training

    ## set up the wds reader
    self.pipe = self.get_wds_pipeline()
    self.pipe.build()

    ## dataset metadata
    meta_dict = self.pipe.reader_meta()
    assert(len(meta_dict) == 1), 'Pipeline has multiple readers but is expected to have only one'
    self.meta = list(meta_dict.values())[0]

  @abc.abstractmethod
  def get_wds_pipeline(self):
    """Returns the pipeline which loads the wds files.

       Expected to have the following format:

         @pipeline_def(batch_size=self.per_shard_batch_size, num_threads=1, device_id=None)
         def wds_pipeline():
           outputs = fn.readers.webdataset(
             ...)
           return outputs
         return wds_pipeline()

       See ViT's `dali_utils.py` for an example

    """
    pass

  @abc.abstractmethod
  def get_dali_pipeline(self):
    """Returns a DALI pipeline instance

       In general, pipelines should have the following structure:

          def text_preprocessing(text):
            ## non-image preprocessing ##

          def data_source(num_classes):

            while True:
              img, text = self.pipe.run()
              text = text_preprocessing(text)
              yield img, text

          @pipeline_def
          def main_vit_pipeline():
            img, text = fn.external_source(source=data_source(), num_outputs=...)

            ## image preprocessing ##

            return img, text

        See ViT's `dali_utils.py` for an example

    """

    pass

class DALIIterator:

  """Wrapper around BaseDaliPipeline that makes iterator compatible
       with clu's PeekableDatasetIterator API."""

  def __init__(self, dali_wrapped_pipeline: BaseDALIPipeline):
    self.wrapped_pipeline = dali_wrapped_pipeline
    self.pipeline = dali_wrapped_pipeline.get_dali_pipeline()
    self.pipeline.build()

    self.training = dali_wrapped_pipeline.training

    ## from clu
    self._mutex = threading.Lock()
    self._peek: Element | None = None
    self._pool = None
    self._peek_future = None

    ## has source info about the current batch
    self.num_unique_examples = 0
    self.source2idx = {}
    self.source_info = None
    self.last_source = None

  def __iter__(self):
    return self

  def get_next_element(self):
    out = self.pipeline.run()

    ## stores source information for the current batch
    ## NOTE: source_info gets lost when using certain preprocessing fns
    ## but for eval, preprocessing is simple enough that this works

    ## Update source_info.
    ## Needed to keep track of padding examples during eval
    if not self.training:
      if self.source_info:
        self.last_source = self.source_info[-1]
      self.source_info = []
      for ex in out[0]:
        info = ex.source_info()
        if info not in self.source2idx:
          self.source2idx[info] = self.num_unique_examples
          self.num_unique_examples += 1
        self.source_info.append(self.source2idx[info])

    return {m.name: out[i].as_array() for i, m in enumerate(self.wrapped_pipeline.modalities)}

  def __next__(self):
    with self._mutex:
      if self._peek is None:
        return self.get_next_element()
      peek = self._peek
      self._peek = None
      return peek

  ## "peek" and "peek_async" taken from clu
  def peek(self):
    """Returns the next element without consuming it.
    This will get the next element from the underlying iterator. The element
    is stored and return on the next call of __next__().

    Returns:
        The next element.
    """
    if self._peek is None:
      self._peek = next(self)
    return self._peek

  def peek_async(self):
    """Same as peek() but returns the Future of the element.
    Users can call this to warm up the iterator.

    Returns:
        Future with the next element. The element is also kept and returned on the
    next call of __next__().
    """
    with self._mutex:
      if self._peek_future is None:
        if self._pool is None:
          self._pool = asynclib.Pool(max_workers=1)
        self._peek_future = self._pool(self.peek)()
      return self._peek_future



  @property
  def element_spec(self) -> ElementSpec:

    batch_size = self.wrapped_pipeline.per_shard_batch_size

    return {
      m.name: ArraySpec(dtype=type_proc(m.out_type), shape=(batch_size, *m.shape))
      for m in self.wrapped_pipeline.modalities
    }


def get_dali_dataset(cfg,
                     ds_shard_id,
                     num_ds_shards,
                     feature_converter_cls,
                     pipeline = None,
                     ):

  assert not bool(feature_converter_cls), 'Passing `feature_converter_cls` is not supported'

  seed = cfg.seed
  if seed is None:
    # Use a shared timestamp across devices as the seed.
    seed = int(multihost_utils.broadcast_one_to_all(np.int32(time.time())))
  cfg.seed = seed

  return iter(DALIIterator(pipeline(cfg, ds_shard_id, num_ds_shards)))

def get_dali_eval_dataset(cfg,
                     ds_shard_id,
                     num_ds_shards,
                     feature_converter_cls,
                     eval_steps = None,
                     pipeline = None,
                     ):

  assert not bool(feature_converter_cls), 'Passing `feature_converter_cls` is not supported'

  ds = iter(DALIIterator(pipeline(cfg, ds_shard_id, num_ds_shards)))

  datasets = {'validation': ds}
  return datasets


class ShardedDatasetIterator:
  """A wrapper iterator that returns sharded arrays."""

  def __init__(
      self,
      iterator: DALIIterator,
      partitioner: partitioning.BasePartitioner,
      global_shapes: PyTree,
  ):
    self._iterator = iterator
    self._global_shapes = global_shapes
    self._partitioner = partitioner

  def __next__(self):
    return utils._create_sharded_array(
        self._partitioner, self._global_shapes, next(self._iterator),
    )

  @property
  def element_spec(self):
    return self._iterator.element_spec

  @property
  def is_nonpadding(self):
    """ Returns a boolean array indicating which examples in the batch
       are not padding examples. """
    bs = self._global_shapes[next(iter(self._global_shapes))][0]

    source_info = self._iterator.source_info
    source_shift_right = [self._iterator.last_source] + source_info[:-1]
    is_nonpadding = (1-(np.array(source_info)==np.array(source_shift_right))).astype(bool)

    return utils._create_sharded_array(
        self._partitioner, {'source': (bs,)}, {'source': np.array(is_nonpadding)},
    )['source']

  @property
  def iterator(self):
    return self._iterator

  def __iter__(self):
    return iter(self._iterator)

  def peek(self):
    return self._iterator.peek()

  def peek_async(self):
    return self._iterator.peek_async()


def create_sharded_iterator(train_iter,
    partitioner,
    checkpoint_cfg,
    data_layout):

  input_shapes = jax.tree_map(
      lambda x: (data_layout.batch_size, *x.shape[1:]), train_iter.element_spec,
  )

  return ShardedDatasetIterator(train_iter, partitioner, input_shapes)

