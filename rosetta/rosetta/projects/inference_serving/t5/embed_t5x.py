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

import numpy as np
import jax 
import jax.numpy as jnp

import seqio
from t5x import gin_utils
from t5x import models
from t5x import partitioning
from t5x import utils
from seqio.vocabularies import PAD_ID

import logging
import time
import os
from typing import Any, Callable, Sequence
import pickle as pkl
import zmq
from rosetta.projects.inference_serving import server_utils
from rosetta.projects.inference_serving.shared_numpy import SharedNPDict

_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'

def get_singleton_batch(batch_size: int):
    in_singleton = ['the quick brown fox jumped over the lazy dog. the quick brown fox jumped over the lazy dog. \
                    the quick brown fox jumped over the lazy dog. the quick brown fox jumped over the lazy dog. \
                    the quick brown fox jumped over the lazy dog. the quick brown fox jumped over the lazy dog. \
                    the quick brown fox jumped over the lazy dog. the quick brown fox jumped over the lazy dog. \
                    the quick brown fox jumped over the lazy dog. the quick brown fox jumped over the lazy dog. \
                    the quick brown fox jumped over the lazy dog. the quick brown fox jumped over the lazy dog. \
                    the quick brown fox jumped over the lazy dog. the quick brown fox jumped over the lazy dog. \
                    the quick brown fox jumped over the lazy dog. the quick brown fox jumped over the lazy dog. \
                    the quick brown fox jumped over the lazy dog. the quick brown fox jumped over the lazy dog. \
                    the quick brown fox jumped over the lazy dog. the quick brown fox jumped over the lazy dog.']
    batch = in_singleton * batch_size
    return {'batch': server_utils.triton_textencode(batch)}


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

def seqio_preprocessing(mbatch, vocab:Any=None, seq_len:int=128):
  return pad_right(vocab.encode(mbatch), seq_len=seq_len, eos_id=vocab.eos_id, pad_id=PAD_ID)

def pow2upper(n: int):
    i = 1
    while i < n:
        i *= 2
    return i

def pow_2_pad_right(tokens_batch, seq_len, eos_id, pad_id):
  padded, tok_lengths = [], []
  max_seq_len = max([len(t) for t in tokens_batch]) + 1
  seq_len = min(pow2upper(max_seq_len), seq_len)

  for t in tokens_batch:
    diff = seq_len - (len(t) + 1)
    # assert diff >= 0
    if diff < 0:
        padded.append(t[:seq_len - 1] + [eos_id])
        tok_lengths.append(seq_len)
    else:
        padded.append(t + [eos_id] + [pad_id] * diff)
        tok_lengths.append(len(t) + 1)

  return jnp.array(padded, dtype=jnp.int32), tok_lengths, seq_len

def seqio_preprocessing_pow2(mbatch, vocab:Any=None, seq_len:int=128):
  return pow_2_pad_right(vocab.encode(mbatch), seq_len=seq_len, eos_id=vocab.eos_id, pad_id=PAD_ID)

def get_infer_fn(
        *,
        model: models.BaseTransformerModel,
        vocab: Any,
        restore_checkpoint_cfg: utils.RestoreCheckpointConfig,
        partitioner: partitioning.BasePartitioner,
        output_dir: str,
        preproc_fn: Callable,
        batch_size: int,
        seq_len: int):

  input_shapes = {'encoder_input_tokens': (batch_size, seq_len)}

  train_state_initializer = utils.TrainStateInitializer(
      optimizer_def=None,  # Do not load optimizer state.
      init_fn=model.get_initial_variables,
      input_shapes=input_shapes,
      partitioner=partitioner)
  train_state_axes = train_state_initializer.train_state_axes
  # Log the variable shapes information and write to a file.
  log_file = os.path.join(output_dir, 'model-info.txt')
  utils.log_model_info(log_file,
                       train_state_initializer.global_train_state_shape,
                       partitioner)

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

  CUDA_VIS = os.getenv('CUDA_VISIBLE_DEVICES')

  def infer_fn(**inputs: np.ndarray):
      start_time = time.time()
      (sequence_batch,) = inputs.values()
      batch = np.array([i[0] for i in sequence_batch])
      sequence_batch = to_str_list(np.char.decode(batch.astype("bytes"), "utf-8"))

      tokenized_padded, batch_len, curr_seqlen = preproc_fn(sequence_batch)
      results = partitioned_fn(train_state.params, {"encoder_input_tokens": tokenized_padded}).astype(jnp.float16)

      results.block_until_ready()
      bs = batch.shape[0]
      pre_pad_time = time.time()
      individual_shape = results[0].shape
      padded_output = np.zeros((bs, *individual_shape), dtype=np.float16)
      for idx, (tensor, true_len) in enumerate(zip(results, batch_len)):
          padded_output[idx, :true_len] = tensor[:true_len]

      logging.info('Throughput (seq/sec): {}, bs: {}, devices: {}, seqlen: {}, throughput w/o pad {}'.format(bs / (time.time() - start_time), bs, CUDA_VIS, curr_seqlen, bs/(pre_pad_time - start_time)))
      # return sliced_output
      return padded_output, np.array(batch_len, dtype=np.int32)

    
  return infer_fn

def to_str_list(batch):
    b = []
    for i in batch:
        b.append(str(i))
    return b

def zmq_run(socket, infer_fn: Callable):
    logging.info("Starting ZMQ Server")

    while True:
        socket_in = socket.recv_pyobj()
        # logging.warning(f"Recieved from socket, {socket_in}")
        localized_inputs = SharedNPDict(metadata=socket_in).localize(close_shared=True)
        # logging.warning("Localized socket_in")
        if isinstance(localized_inputs, dict):
            if 'singleton' in localized_inputs.keys():
                count = localized_inputs['singleton']
                localized_inputs = get_singleton_batch(localized_inputs['singleton'])
                logging.info(f"Recieved singleton command {count}")
            else:
                for k, v in localized_inputs.items():
                    localized_inputs[k] = pkl.loads(v[0])
        try:
            padded_outs, seqlens = infer_fn(**localized_inputs)
            # logging.info("created out")
            outputs_shared = SharedNPDict(dict_to_share={'encodings_padded': padded_outs, 'encodings_seqlens': seqlens})
            logging.info("Shared out")
            outputs = outputs_shared.get_metas()
            outputs_shared.close()
            # logging.info("closed out")
        except Exception as e:
            outputs = str(e)
        
        socket.send_pyobj(outputs)

if __name__ == '__main__':
  from absl import app
  from absl import flags
  import gin

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

  def main(argv: Sequence[str]):
    """Wrapper for pdb post mortems."""

    if jax.process_index() == 0:
      main_fn = lambda: _main(argv)
      _main(argv)
    else:
      _main(argv)

  def _main(argv: Sequence[str]):
    """True main function."""
    if len(argv) > 1:
      raise app.UsageError('Too many command-line arguments.')

    socket_name = os.environ.get('SOCKET_ADDRESS')
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.REP)
    socket.connect(socket_name)

    # Create gin-configurable version of `eval`.
    # tr = functools.partial(triton_run, port=FLAGS.port)
    run_using_gin = gin.configurable(zmq_run)

    gin_utils.parse_gin_flags(
        # User-provided gin paths take precedence if relative paths conflict.
        FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
        FLAGS.gin_file,
        FLAGS.gin_bindings)
    run_using_gin(socket)

  gin_utils.run(main)
