# Copyright (c) 2022-2023 NVIDIA Corporation
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
# Copyright 2022 The T5X Authors.
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

"""General utility functions for diffusion/wds """
from jax.experimental import multihost_utils
import tensorflow as tf
import typing_extensions
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Type, Union, List
from flax.linen import partitioning as flax_partitioning
from t5x import partitioning
import jax
import jax.numpy as jnp
import numpy as np
import seqio
import time

from absl import logging
import dataclasses
import collections
import collections.abc
from concurrent.futures import thread
import contextlib
import dataclasses
import functools
import importlib
import inspect
import os
import re
import time
import typing
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Type, Union
import warnings
import gc

from absl import logging
import clu.data
from flax import traverse_util
import flax.core
from flax.core import scope as flax_scope
from flax.linen import partitioning as flax_partitioning
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import seqio
from t5x import checkpoints
from t5x import optimizers
from t5x import partitioning
from t5x import state_utils
from t5x import train_state as train_state_lib
import tensorflow as tf
from tensorflow.io import gfile
import typing_extensions
from rosetta.projects.diffusion import wds_utils
from rosetta.projects.diffusion.denoisers import DenoisingFunctionCallableWithParams, DenoisingFunctionCallable

Array = Union[np.ndarray, jnp.ndarray, tf.Tensor]
PyTreeDef = type(jax.tree_structure(None))
PartitionSpec = partitioning.PartitionSpec
DType = Union[np.dtype, type(jnp.bfloat16)]
Shape = Tuple[int, ...]

# -----------------------------------------------------------------------------
# SeqIO utility functions.
# -----------------------------------------------------------------------------


def import_module(module: str):
  """Imports the given module at runtime."""
  logging.info('Importing %s.', module)
  try:
    importlib.import_module(module)
  except RuntimeError as e:
    if (str(e) ==
        'Attempted to add a new configurable after the config was locked.'):
      raise RuntimeError(
          'Your Task/Mixture module contains gin configurables that must be '
          'loaded before gin flag parsing. One fix is to add '
          f"'import {module}' in your gin file.")
    raise e

class ShardInfo:
    def __init__(self, id, ct):
        self.id = id
        self.ct = ct

def get_dataset(cfg: wds_utils.WebDatasetConfig,
                shard_id: int,
                num_shards: int,
                feature_converter_cls: Type[seqio.FeatureConverter],
                num_epochs: Optional[int] = None,
                continue_from_last_checkpoint: bool = False, should_batch=True) -> tf.data.Dataset:
  """Returns a dataset from webdataset based on a `WebDatasetConfig`."""
  if continue_from_last_checkpoint:
    raise ValueError(
        '`continue_from_last_checkpoint` must be set to False as this is not '
        'supported by this dataset fn.')
  del continue_from_last_checkpoint

  if cfg.batch_size % num_shards:
    raise ValueError(
        f'Batch size ({cfg.batch_size}) must be divisible by number of '
        f'shards ({num_shards}).')

  if cfg.seed is None:
    # Use a shared timestamp across devices as the seed.
    seed = multihost_utils.broadcast_one_to_all(np.int32(time.time()))
  else:
    seed = cfg.seed

  shard_info = ShardInfo(shard_id, num_shards)
  return get_dataset_inner(cfg, shard_info, feature_converter_cls, seed,
                           num_epochs, should_batch)

def get_dataset_inner(cfg: wds_utils.WebDatasetConfig,
                      shard_info: ShardInfo,
                      feature_converter_cls: Callable[...,
                                                      seqio.FeatureConverter],
                      seed: Optional[int] = None,
                      num_epochs: Optional[int] = None,
                      should_batch=True):
  """Internal fn to load a dataset from WebDataset based on a `WebDatasetConfig`."""
  batch_size = cfg.batch_size // shard_info.ct
  if seed is not None:
    multihost_utils.assert_equal(
        np.array(seed),
        f'`seed` is not same across hosts; {jax.process_index} has a seed of '
        f'{seed}')
    logging.info(
        "Initializing dataset for task '%s' with a replica batch size of %d and "
        'a seed of %d', cfg.mixture_or_task_name, batch_size, seed)

  # should_batch implies that we will add a batch dimension ourselves to the loaded data
  print("SHOULD BATCH?", should_batch)
  ds, out_shapes, out_types = wds_utils.get_mm_wds_from_urls(cfg, batch_size=batch_size if should_batch else -1)
  if should_batch:
      for k in out_shapes.keys():
          out_shapes[k] = (batch_size,) + out_shapes[k]

  ds = tf.data.Dataset.from_generator(
        generator=ds.__iter__, output_types=out_types, output_shapes=out_shapes
       )
  if cfg.samples:
    add_fake_length_method(ds, cfg.samples)
  return ds

class GetDatasetCallable(typing_extensions.Protocol):

  def __call__(self,
               cfg: wds_utils.WebDatasetConfig,
               shard_id: int,
               num_shards: int,
               feature_converter_cls: Callable[..., seqio.FeatureConverter],
               num_epochs: Optional[int] = None,
               continue_from_last_checkpoint: bool = True) -> tf.data.Dataset:
    ...

def multihost_assert_equal(input_tree, fail_message: str = ''):
  """Verifies that all the hosts have the same tree of values."""
  # Internal mock TPU handling
  multihost_utils.assert_equal(input_tree, fail_message)
class InferStepWithRngCallable(typing_extensions.Protocol):

  def __call__(self,
               params: Mapping[str, Any],
               batch: Mapping[str, jnp.ndarray],
               rng: jnp.ndarray = None) -> PyTreeDef:
    """Runs an inference step returning a prediction or score."""
    ...


class InferStepWithoutRngCallable(typing_extensions.Protocol):

  def __call__(self, params: Mapping[str, Any],
               batch: Mapping[str, jnp.ndarray]) -> PyTreeDef:
    """Runs an inference step returning a prediction or score."""
    ...


InferStepCallable = Union[InferStepWithRngCallable, InferStepWithoutRngCallable]

# NOTE: We're not more prescriptive than PyTreeDef because that's what
# InferStepCallable expects.
_InferFnResult = Sequence[Tuple[int, PyTreeDef]]
_InferFnWithAuxResult = Tuple[_InferFnResult, Mapping[str, Sequence[Any]]]


class InferFnCallable(typing_extensions.Protocol):

  def __call__(
      self,
      ds: tf.data.Dataset,
      train_state: train_state_lib.TrainState,
      rng: Optional[jnp.ndarray] = None
  ) -> Union[_InferFnResult, _InferFnWithAuxResult]:
    """Runs inference on the dataset."""
    ...


def _remove_padding(all_inferences, all_indices):
  """Remove padded examples.

  Args:
    all_inferences: PyTree[total_examples + padding_count, ...].
    all_indices: [total_examples + padding_count].

  Returns:
    all_inferences in shape PyTree[total_examples, ...].
    all_indices in shape [total_exmamples].
  """
  non_pad_idxs = np.where(all_indices >= 0)
  all_indices = all_indices[non_pad_idxs]
  all_inferences = jax.tree_map(lambda x: x[non_pad_idxs], all_inferences)
  return all_inferences, all_indices


def get_infer_fn(infer_step: InferStepCallable, batch_size: int,
                 train_state_axes: train_state_lib.TrainState,
                 partitioner: partitioning.BasePartitioner, num_samples:Optional[int]=None,
                 return_batch_keys:Optional[List[str]]=None) -> InferFnCallable:
  """Get prediction function for the SeqIO evaluator.

  The returned prediction function should take in an enumerated dataset, make
  predictions and return in an enumerated form with the original indices and
  examples zipped together. This ensures that the predictions are compared to
  the targets in a correct order even if the dataset is sharded across
  multiple hosts and gathered in a nondeterministic way.

  jax.process_index == 0 is used as a "main host", i.e., it gathers all
  inference results and returns.

  Shape notation:
    Per replica set num replicas: R
    Per replica set batch size: B
    Number of replica sets: H
    Length: L

    Some transformations have shape transformation annotation, e.g.,
    [B, L] -> [R, B/R, L].

  Args:
    infer_step: a callable that executes one prediction step. Should not yet be
      partitioned or pmapped.
    batch_size: the number of examples in the global infer batch.
    train_state_axes: Partitioning info for the train state object.
    partitioner: partitioner to use.

  Returns:
    predict_fn: a callable which takes in the enumerated infer dataset and an
      optimizer and runs the prediction.
  """

  return_batch = return_batch_keys is not None
  def infer_step_with_indices(params, batch, rng, indices):
    if 'rng' in inspect.signature(infer_step).parameters:
      res = typing.cast(InferStepWithRngCallable, infer_step)(params, batch,
                                                              rng)
    else:
      res = typing.cast(InferStepWithoutRngCallable, infer_step)(params, batch)
    if return_batch:
        return indices, res, batch
    else:
        return indices, res

  outs = (None, ) * 3 if return_batch else (None, ) * 2
  partitioned_infer_step = partitioner.partition(
      infer_step_with_indices,
      in_axis_resources=(train_state_axes.params,
                         partitioner.data_partition_spec, None,
                         partitioner.data_partition_spec),
      out_axis_resources=outs)

  data_layout = partitioner.get_data_layout(batch_size)
  shard_id = data_layout.shard_id
  num_shards = data_layout.num_shards

  per_shard_batch_size = batch_size // num_shards
  num_batches = num_samples // batch_size 

  def infer_fn(ds: tf.data.Dataset,
               train_state: train_state_lib.TrainState,
               rng: Optional[jnp.ndarray] = None):
    ds_shapes = jax.tree_map(lambda x: jnp.array(x.shape), ds.element_spec)
    ds = ds.enumerate()
    multihost_assert_equal(
        ds_shapes, 'Dataset element shapes do not agree across hosts. '
        'This could be an indication that the dataset is nondeterministic.')
    try:
      original_ds_length = num_samples #len(ds)
      dataset_remainder = original_ds_length % batch_size  # pytype:disable=wrong-arg-types
      logging.info('length of dataset = %s', num_samples)# len(ds))
    except TypeError as e:
      if str(e) == 'dataset length is unknown.':
        logging.warning(
            'The following error is likely due to the use of TensorFlow v1 in '
            'your dataset pipeline. Verify you are not importing from '
            '`tf.compat.v1` as part of your pipeline.')
      raise e

    if dataset_remainder:
      dataset_pad_amt = batch_size - dataset_remainder
      logging.info(
          'Padding infer dataset with %d examples for even per-replica shards.',
          dataset_pad_amt)
      # Pad with the first example using an index of -1 so seqio will ignore.
      pad_ds = ds.take(1).map(lambda i, x: (np.int64(-1), x)).repeat(
          dataset_pad_amt)
      ds = ds.concatenate(pad_ds)

    # Shard the infer dataset across replica sets.
    sharded_ds = ds.shard(num_shards, shard_id).batch(
        per_shard_batch_size, drop_remainder=True)
    # multihost_assert_equal(
       # jnp.array(len(sharded_ds)),
       # 'Dataset lengths do not agree across hosts.')

    logging.info(
        'The infer dataset is sharded into %d shards with per-shard '
        'batch size of %d', num_shards, per_shard_batch_size)

    # Run inference for each replica set.
    batched_results, all_indices, batched_return = [], [], []
    for batch_idx, (index, infer_batch) in enumerate(sharded_ds.as_numpy_iterator()):
      logging.info(str(index))
      if batch_idx >= num_batches:
          break
      if rng is None:
        step_rng = None
      else:
        step_rng, rng = jax.random.split(rng)
      # Run fast inference on batch.
      # [B, ...] -> [B * shard_count, ...]
      # partitioned_infer_step executes infer_step on sharded batched data, and
      # returns de-sharded batched indices and result replicated on all hosts.

      if jax.config.jax_array and jax.process_count() > 1:
        logging.info('in array conf. array shape is ' + str(jax.tree_map(lambda x: x.shape, infer_batch)))
        inputs = multihost_utils.host_local_array_to_global_array(
            (infer_batch, step_rng, index), partitioner.mesh,
            (partitioner.data_partition_spec, None,
             partitioner.data_partition_spec))
        logging.info('input batch shape' +  str(tree_shape(inputs[0])))
        if return_batch:
            batch_indices, batch_result, batch_ret = partitioned_infer_step(
                train_state.params, *inputs)
            logging.info('out batch shape' +  str(tree_shape(batch_ret)))
            batch_indices, batch_result, batch_ret = multihost_utils.global_array_to_host_local_array(
                (batch_indices, batch_result, batch_ret), partitioner.mesh, (None, None, None))

        else:
            batch_indices, batch_result = partitioned_infer_step(
                train_state.params, *inputs)

            batch_indices, batch_result = multihost_utils.global_array_to_host_local_array(
                (batch_indices, batch_result), partitioner.mesh, (None, None))

        logging.info('out shape' + str(jax.tree_map(lambda x: x.shape, batch_result)))
        logging.info('out idx shape' + str(jax.tree_map(lambda x: x.shape, batch_indices)))
      else:
        if return_batch:
            batch_indices, batch_result, batch_ret = partitioned_infer_step(
                train_state.params, infer_batch, step_rng, index)
        else:
            batch_indices, batch_result = partitioned_infer_step(
                train_state.params, infer_batch, step_rng, index)
      logging.info('Inference of batch %s done.', index)


      def _copy_to_host_async(x):
        if hasattr(x, 'addressable_data'):
          # Array is fully replicated.
          x.addressable_data(0).copy_to_host_async()
          return x.addressable_data(0)
        else:
          x.copy_to_host_async()
          return x

      try:
        logging.info("full result " + str(jax.tree_map(lambda x: x.shape, batch_result)))
        batch_result = jax.tree_map(_copy_to_host_async, batch_result)
        if return_batch_keys:
            if return_batch_keys == True:
                ret = batch_ret
            else:
                ret = {}
                for k in return_batch_keys:
                    ret[k] = batch_ret[k]
            batch_return = ret 
        else:
            batch_return = None
        batch_indices = jax.tree_map(_copy_to_host_async, batch_indices)
      except AttributeError:
        # Similar to jax.device_get, we skip transfers for non DeviceArrays.
        pass

      logging.info('out idx shape after copy' + str(jax.tree_map(lambda x: x.shape, batch_indices)))

      batched_results.append(batch_result)
      if return_batch_keys:
          batched_return.append(batch_return)
      all_indices.append(batch_indices)
      logging.info('returns' + str(tree_shape(batched_return)))

    logging.info('Inference of all batches done.')
    all_inferences = batched_results

    # List[B * shard_count, ...] -> [B * shard_count * batch_count, ...]
    all_inferences = jax.tree_map(lambda *args: np.concatenate(args),
                                  *all_inferences)
    all_indices = np.concatenate(all_indices)
    logging.info(str(tree_shape(all_inferences)) + str(tree_shape(all_indices)))

    all_inferences, all_indices = _remove_padding(all_inferences, all_indices)

    # Results are returned from infer_step out of order due to shard operation.
    # Note: remove padding first, as -1 indices would mess up this operation.
    # Note: all_inferences may be a PyTree, not just an array, e.g. if
    # `infer_step` is `model.predict_batch_with_aux`.
    logging.info(str(tree_shape(all_inferences)) + str(tree_shape(all_indices)))
    if return_batch_keys:
        all_batches = jax.tree_map(lambda *args: np.concatenate(args),
                                   *batched_return)

    # aux_values is supposed to be a dictionary that maps strings to a set of
    # auxiliary values.
    #
    # We don't want to flatten/unflatten the aux values. We want to preserve the
    # unflattened values with the type List[Mapping[str, Sequence[Any]]]. We do
    # this as a memory optimization to avoid lots of redundant keys if we'd
    # instead had List[Mapping[str, Any]].
    #
    # It has shape Mapping[str, [B * shard_count * batch_count, ...]]. That is,
    # the first dimension of each of the values in aux_values is equal to
    # len(all_inferences).
    aux_values = None
    if (isinstance(all_inferences, tuple) and len(all_inferences) == 2 and
        isinstance(all_inferences[1], Mapping)):
      all_inferences, aux_values = all_inferences

    # Translate to List[...] by flattening inferences making sure to
    # preserve structure of individual elements (inferences are not assumed to
    # be simple np.array). Finally, zip inferences with corresponding indices
    # and convert leaf np.arrays into lists.
    if return_batch_keys:
        indices_and_outputs = (all_indices, all_inferences, all_batches)

    else:
        indices_and_outputs = (all_indices, all_inferences)

    logging.info('final idxes ' + str(all_indices))
    logging.info('final out ' + str(tree_shape(indices_and_outputs)))
    if indices_and_outputs[0].shape[0] != original_ds_length:
      raise ValueError(
          'Size of indices_and_outputs does not match length of original '
          'dataset: %d versus %d' %
          (indices_and_outputs[0].shape[0], original_ds_length))

    if aux_values is None:
      return indices_and_outputs
    else:
      aux_values = jax.tree_map(lambda x: np.array(x).tolist(), aux_values)
      return indices_and_outputs, aux_values

  return infer_fn

class DiffusionSamplingEvaluator:
  def __init__(self, dataset_cfg, dataset, log_dir=None, fixed_rng=True):
    self.dataset_cfg = dataset_cfg
    self.dataset = dataset
    self.log_dir = log_dir
    self.rng = jax.random.PRNGKey(0)
    self.keep_random = fixed_rng
    self.eval_tasks = [1] #non empty
  
  def evaluate(self, 
              compute_metrics: bool,
              step: int,
              predict_fn: Optional[Callable] = None,
              score_fn: Optional[Callable] = None,
              predict_with_aux_fn: Optional[Callable] = None,
              ):
    samples = predict_fn(self.dataset)

    save_dir = '{}/samples/{}_trainsteps'.format(self.log_dir, step)
    try:
        os.makedirs(save_dir)
    except:
        pass

    if jax.process_index() == 0:
        import matplotlib.image as matimg
        logging.info('Saving samples to {}'.format(save_dir))
        for i in range(len(samples)):
            np_arr = np.clip(samples[i][1], a_min = 0, a_max = 1)
            matimg.imsave(os.path.join(save_dir, '{}-{}.png'.format(jax.process_index(), i)), np_arr)
            if len(samples) == 3:
                np_arr_batch = (samples[2]['low_res_images'][i] + 1) / 2.
                logging.info(str(np_arr_batch))
                np_arr_batch = np.clip(np_arr_batch, a_min = 0, a_max = 1)
                matimg.imsave(os.path.join(save_dir, 'dataset-{}-{}.png'.format(jax.process_index(), i)), np_arr_batch)

    multihost_utils.sync_global_devices('eval')
    return None, None

def expand_dims_like(target, source):
    return jnp.reshape(target, target.shape + (1, ) * (len(source.shape) - len(target.shape)))

def tree_shape(tree):
    return jax.tree_map(lambda x: x.shape, tree)

def add_fake_length_method(obj, size):
    def length(self):
        return size

    Combined = type(
        obj.__class__.__name__ + "_Length",
        (obj.__class__,),
        {"__len__": length},
    )
    obj.__class__ = Combined
    return obj

def _copy_to_host_async(x):
        if hasattr(x, 'addressable_data'):
          # Array is fully replicated.
          x.addressable_data(0).copy_to_host_async()
          return x.addressable_data(0)
        else:
          x.copy_to_host_async()
          return x
