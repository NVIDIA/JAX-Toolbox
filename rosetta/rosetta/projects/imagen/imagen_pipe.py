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

# Runs inference of an Imagen base model and a 64->256 superresolution model
import dataclasses
import os
import re
import functools
from typing import Mapping, Any, Optional, Callable, Sequence
import logging

import numpy as np
import jax
import jax.numpy as jnp
import seqio
from t5x import partitioning
from t5x import utils
from t5x import models as t5x_models
from seqio.vocabularies import PAD_ID
from rosetta.projects.diffusion import models
from rosetta.projects.diffusion import samplers
import matplotlib.image as matimg

from rosetta.projects.diffusion.mm_utils import expand_dims_like
# Automatically search for gin files relative to the T5X package.
_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]

@dataclasses.dataclass
class DiffusionModelSetupData:
    model: models.DenoisingDiffusionModel
    sampling_cfg: samplers.SamplingConfig
    restore_checkpoint_cfg: utils.RestoreCheckpointConfig
    partitioner: partitioning.BasePartitioner
    input_shapes: Mapping[str, Any]
    input_types: Mapping[str, Any]

def pad_right(tokens, seq_len, eos_id,pad_id):
  padded, tok_lengths = [], []
  for t in tokens:
    diff = seq_len - (len(t) + 1)
    #assert diff >= 0
    if diff < 0:
        padded.append(t[:seq_len - 1] + [eos_id])
        tok_lengths.append(seq_len)
    else:
        padded.append(t + [eos_id] + [pad_id] * diff)
        tok_lengths.append(len(t) + 1)

  return jnp.array(padded, dtype=jnp.int32), tok_lengths, seq_len

def seqio_preprocessing(mbatch, vocab:Any, seq_len:int=128):
  return pad_right(vocab.encode(mbatch), seq_len=seq_len, eos_id=vocab.eos_id, pad_id=PAD_ID)

def setup_text_enc(model: t5x_models.BaseTransformerModel,
                   restore_checkpoint_cfg: utils.RestoreCheckpointConfig,
                   partitioner: partitioning.BasePartitioner,
                   batch_size=1, seq_len=128, vocab=None,
                   ):
  input_shapes = {'encoder_input_tokens': (batch_size, seq_len)}

  train_state_initializer = utils.TrainStateInitializer(
      optimizer_def=None,  # Do not load optimizer state.
      init_fn=model.get_initial_variables,
      input_shapes=input_shapes,
      partitioner=partitioner)
  train_state_axes = train_state_initializer.train_state_axes

  # Disable strictness since we are dropping the optimizer state.
  restore_checkpoint_cfg.strict = False

  fallback_init_rng = None

  if fallback_init_rng is not None:
    fallback_init_rng = jax.random.PRNGKey(fallback_init_rng)
  train_state = list(train_state_initializer.from_checkpoints([restore_checkpoint_cfg], init_rng=fallback_init_rng))[0]
  logging.warning(f'Restored from Checkpoint: {train_state[1]}')
  train_state = train_state[0]

  partitioned_fn = partitioner.partition(
    model.score_batch,
    in_axis_resources=(train_state_axes.params, partitioning.PartitionSpec('data',)),
    out_axis_resources=None)

  def infer_fn(inputs: Sequence[str]):
      tokenized_padded, batch_len, curr_seqlen = seqio_preprocessing(inputs, vocab, seq_len=seq_len)
      results = partitioned_fn(train_state.params, {"encoder_input_tokens": tokenized_padded}).astype(jnp.float16)

      bs = len(inputs)
      individual_shape = results[0].shape
      padded_output = np.zeros((bs, *individual_shape), dtype=np.float16)
      for idx, (tensor, true_len) in enumerate(zip(results, batch_len)):
          padded_output[idx, :true_len] = tensor[:true_len]

      return padded_output, jnp.array(batch_len, dtype=np.int32)


  return infer_fn

def get_sample_fn(model_setup:DiffusionModelSetupData):
  train_state_initializer = utils.TrainStateInitializer(
      optimizer_def=None,  # Do not load optimizer state.
      init_fn=model_setup.model.get_initial_variables,
      input_shapes=model_setup.input_shapes,
      input_types=model_setup.input_types,
      partitioner=model_setup.partitioner)

  train_state_axes = train_state_initializer.train_state_axes
  
  # Disable strictness since we are dropping the optimizer state.
  model_setup.restore_checkpoint_cfg.strict = False
  
  fallback_init_rng = None
  
  if fallback_init_rng is not None:
    fallback_init_rng = jax.random.PRNGKey(fallback_init_rng)
  train_state = list(train_state_initializer.from_checkpoints([model_setup.restore_checkpoint_cfg], init_rng=fallback_init_rng))[0]
  logging.warning(f'Restored from Checkpoint: {train_state[1]}')
  train_state = train_state[0]
  
  model_pred = functools.partial(model_setup.model.predict_batch, sampling_cfg=model_setup.sampling_cfg)
  partitioned_fn = model_setup.partitioner.partition(
    model_pred,
    in_axis_resources=(train_state_axes.params, model_setup.partitioner.data_partition_spec, None),
    out_axis_resources=model_setup.partitioner.data_partition_spec)

  return train_state.params, partitioned_fn

def sanitize_filename(filename):
  # Remove leading and trailing whitespaces
  filename = filename.strip()
  
  # Replace spaces with underscores
  filename = filename.replace(" ", "_")
  
  # Remove any other characters that are not allowed in a Linux filename
  filename = re.sub(r'[^\w\.-]', '', filename)
  
  # Remove forward slashes
  filename = filename.replace("/", "")
  
  return filename

def sample(
  base_setupdata: DiffusionModelSetupData,
  sr256_setupdata: DiffusionModelSetupData,
  out_dir: str,
  sr1024_setupdata: DiffusionModelSetupData=None,
  gen_per_prompt: int = 1,
  text_enc_infer = Callable,
  prompt_file=None,
  batch_size=32,
  max_images=50000000,
  base_img_size=(64, 64, 3),
  sr256_img_size=(256, 256, 3),
  sr1024_img_size=(1024, 1024, 3),
  noise_conditioning_aug=0.002,
  resume_from=0
  ):
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  base_dir = os.path.join(out_dir, 'base')
  sr_dir = os.path.join(out_dir, 'sr')
  sr2_dir = os.path.join(out_dir, 'sr2')
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
  if not os.path.exists(sr_dir):
    os.makedirs(sr_dir)
  if sr1024_setupdata is not None and not os.path.exists(sr_dir):
    os.makedirs(sr_dir)

  with open(prompt_file, 'r') as f:
      prompts = f.readlines()
      prompt_ct = len(prompts)

  # Set up models
  base_params, base_fn = get_sample_fn(base_setupdata)
  sr256_params, sr256_fn = get_sample_fn(sr256_setupdata)
  if sr1024_setupdata is not None:
    sr1024_params, sr1024_fn = get_sample_fn(sr1024_setupdata)
  text_encoder = text_enc_infer

  sampled_ctr = 0
  rng = jax.random.PRNGKey(0)
  for start_idx in range(resume_from, max_images, batch_size // gen_per_prompt):
      if start_idx > prompt_ct:
          break
      prompt_batch = prompts[start_idx: start_idx + (batch_size // gen_per_prompt)] * gen_per_prompt
      rng, rng_base, rng_sr, rng_sr2, rng_aug = jax.random.split(rng, 5)

      # Encode Text
      encoded_text, text_lens = text_encoder(prompt_batch)
      text_mask = np.zeros(encoded_text.shape[:2])
      for i in range(text_lens.shape[0]):
          text_mask[i][:text_lens[i]] = 1

      # Base model generation
      base_img_inputs = jnp.zeros((len(prompt_batch), *base_img_size))
      sr256_img_inputs = jnp.zeros((len(prompt_batch), *sr256_img_size))
      sr1024_img_inputs = jnp.zeros((len(prompt_batch), *sr1024_img_size))
      base_batch = {'samples': base_img_inputs, 'text': encoded_text, 'text_mask': text_mask}
      base_out = base_fn(base_params, base_batch, rng_base)
      for i in range(base_out.shape[0]):
          matimg.imsave(os.path.join(base_dir, sanitize_filename(f'{prompt_batch[i]}_{sampled_ctr + i}.png')), np.clip(base_out[i], a_min=0, a_max=1))

      # Stage 2: Super Resolution (64-> 256)
      base_aug = (base_out * 2 - 1)
      noise_aug_level = expand_dims_like(jnp.ones((base_aug.shape[0], )) * noise_conditioning_aug, base_aug)
      sr256_batch = {'samples': sr256_img_inputs, 'text':encoded_text, 'text_mask':text_mask, 'low_res_images': base_aug, 'noise_aug_level': noise_aug_level}
      sr_out = sr256_fn(sr256_params, sr256_batch, rng_sr)
      sr_out = jnp.clip(sr_out, a_min = 0, a_max = 1)
      for i in range(sr_out.shape[0]):
          matimg.imsave(os.path.join(sr_dir, sanitize_filename(f'{prompt_batch[i]}_{sampled_ctr + i}.png')), sr_out[i])

      # Stage 3: Super Resolution (256-> 1024)
      if sr1024_setupdata is not None:
        sr_aug = (sr_out * 2 - 1)
        noise_aug_level = expand_dims_like(jnp.ones((sr_aug.shape[0], )) * noise_conditioning_aug, base_aug)
        sr1024_batch = {'samples': sr1024_img_inputs, 'text':encoded_text, 'text_mask':text_mask, 'low_res_images': sr_aug, 'noise_aug_level': noise_aug_level}
        sr_out = sr1024_fn(sr1024_params, sr1024_batch, rng_sr2)
        sr_out = jnp.clip(sr_out, a_min = 0, a_max = 1)
        for i in range(sr_out.shape[0]):
            matimg.imsave(os.path.join(sr2_dir, sanitize_filename(f'{prompt_batch[i]}_{sampled_ctr + i}.png')), sr_out[i])

      sampled_ctr += sr_out.shape[0]
  

if __name__ == '__main__':
  # pylint: disable=g-import-not-at-top
  from absl import app
  from absl import flags
  import gin
  from t5x import gin_utils
  import tensorflow as tf
  # pylint: enable=g-import-not-at-top
  FLAGS = flags.FLAGS

  jax.config.parse_flags_with_absl()

  flags.DEFINE_multi_string(
      'gin_file',
      default=None,
      help='Path to gin configuration file. Multiple paths may be passed and '
      'will be imported in the given order, with later configurations  '
      'overriding earlier ones.')

  flags.DEFINE_multi_string(
      'gin_bindings', default=[], help='Individual gin bindings.')

  flags.DEFINE_list(
      'gin_search_paths',
      default=['.'],
      help='Comma-separated list of gin config path prefixes to be prepended '
      'to suffixes given via `--gin_file`. If a file appears in. Only the '
      'first prefix that produces a valid path for each suffix will be '
      'used.')

  flags.DEFINE_boolean(
      'multiprocess_gpu',
      False,
      help='Initialize JAX distributed system for multi-host GPU, using '
      '`coordinator_address`, `process_count`, and `process_index`.')

  flags.DEFINE_string(
      'coordinator_address',
      None,
      help='IP address:port for multi-host GPU coordinator.')

  flags.DEFINE_integer(
      'process_count', None, help='Number of processes for multi-host GPU.')

  flags.DEFINE_integer('process_index', None, help='Index of this process.')


  def main(argv: Sequence[str]):
    """Wrapper for pdb post mortems."""
    _main(argv)

  def _main(argv: Sequence[str]):
    """True main function."""
    if len(argv) > 1:
      raise app.UsageError('Too many command-line arguments.')

    # OOM fix. Prevents TF from seeing GPUs to stop conflict with JAX.
    # This must go after InitGoogle(), which is called by
    # gin_utils.run(main).
    tf.config.experimental.set_visible_devices([], 'GPU')


    if FLAGS.multiprocess_gpu:
      logging.info(
          'Initializing distributed system for multi-host GPU:\n'
          '  coordinator_address: %s\n  process_count: %s\n  process_index: %s',
          FLAGS.coordinator_address, FLAGS.process_count, FLAGS.process_index)

      jax.distributed.initialize(FLAGS.coordinator_address, FLAGS.process_count,
                                 FLAGS.process_index)

    # Create gin-configurable version of `train`.
    sample_using_gin = gin.configurable(sample)

    gin_utils.parse_gin_flags(
        # User-provided gin paths take precedence if relative paths conflict.
        FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
        FLAGS.gin_file,
        FLAGS.gin_bindings)
    sample_using_gin()
    jax.effects_barrier()


  gin_utils.run(main)
