# Copyright 2022 Google LLC.
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

import os
from collections.abc import Sequence

import flax
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from absl import logging

from rosetta.projects.vit import models
from t5x import checkpoints, partitioning, utils


_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
]

#### from https://github.com/google-research/vision_transformer/blob/62a446f1b3bb9e470db5689bfd7407a8d91bae8a/vit_jax/checkpoint.py
def interpolate_posembed(posemb, num_tokens: int, has_class_token: bool):
  """Interpolate given positional embedding parameters into a new shape.
  Args:
    posemb: positional embedding parameters.
    num_tokens: desired number of tokens.
    has_class_token: True if the positional embedding parameters contain a
      class token.
  Returns:
    Positional embedding parameters interpolated into the new shape.
  """
  assert posemb.shape[0] == 1
  if has_class_token:
    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
    num_tokens -= 1
  else:
    posemb_tok, posemb_grid = posemb[:, :0], posemb[0, 0:]

  gs_old = int(np.sqrt(len(posemb_grid)))
  gs_new = int(np.sqrt(num_tokens))
  logging.info('interpolate_posembed: grid-size from %s to %s', gs_old, gs_new)
  assert gs_old ** 2 == len(posemb_grid), f'{gs_old ** 2} != {len(posemb_grid)}'
  assert gs_new ** 2 == num_tokens, f'{gs_new ** 2} != {num_tokens}'
  posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

  zoom = (gs_new / gs_old, gs_new / gs_old, 1)
  posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
  posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
  return jnp.array(np.concatenate([posemb_tok, posemb_grid], axis=1))


def convert_t5x_finetune_to_pretrain(
    pretrain_ckpt_dir: str,
    finetune_ckpt_dir: str,
    pretrained_model: models.ViTModel,
    finetune_resolution: int,
    partitioner: partitioning.PjitPartitioner,
):

    pt_input_shapes = {'images': (1, pretrained_model.module.config.image_size, pretrained_model.module.config.image_size, 3)}
    input_dtypes = {'images': jnp.float32}

    pt_train_state_initializer = utils.TrainStateInitializer(
          optimizer_def=None,  # Do not load optimizer state.
          init_fn=pretrained_model.get_initial_variables,
          input_shapes=pt_input_shapes,
          input_types=input_dtypes,
          partitioner=partitioner)

    # Start by filling t5x state with initialized model
    pt_init_ts = pt_train_state_initializer.from_scratch(jax.random.PRNGKey(0))
    pt_checkpointer = checkpoints.Checkpointer(
        pt_init_ts,
        partitioner,
        pretrain_ckpt_dir,
    )
    restored_state = pt_checkpointer.restore()

    ft_input_shapes = {'images': (1, finetune_resolution, finetune_resolution, 3)}

    utils.TrainStateInitializer(
          optimizer_def=None,  # Do not load optimizer state.
          init_fn=pretrained_model.get_initial_variables,
          input_shapes=ft_input_shapes,
          input_types=input_dtypes,
          partitioner=partitioner)

    ft_init_ts = pt_train_state_initializer.from_scratch(jax.random.PRNGKey(0))

    pt_posemb = restored_state.params['vision_model']['VisionTransformer']['Transformer']['posembed_input']['pos_embedding']
    pt_posemb = jnp.expand_dims(pt_posemb, 0)

    new_shape = 1 + (finetune_resolution // pretrained_model.module.config.patch_size)**2
    ft_posemb = interpolate_posembed(
          pt_posemb, new_shape, has_class_token=True)

    ft_posemb = jnp.squeeze(ft_posemb, 0)

    pt_params = restored_state.params.unfreeze()
    pt_params['vision_model']['VisionTransformer']['Transformer']['posembed_input']['pos_embedding'] = ft_posemb
    ## drop head
    pt_params['vision_model']['VisionTransformer']['pre_logits'] = {}
    pt_params['vision_model']['head'] = ft_init_ts.params['vision_model']['head']

    ft_init_ts = ft_init_ts.replace_params(pt_params)

    ft_checkpointer = checkpoints.Checkpointer(
        ft_init_ts,
        partitioner,
        finetune_ckpt_dir,
    )
    ft_checkpointer.save(ft_init_ts)
    print(f'Saved to {finetune_ckpt_dir}')

    # Verify that the state transition worked
    flat_ft_params = flax.traverse_util.flatten_dict(ft_init_ts.params, sep='/')
    flat_pt_params = flax.traverse_util.flatten_dict(restored_state.params, sep='/')

    for n in flat_ft_params.keys():
        if 'vision_model/head' in n or 'posembed_input' in n:
            continue
        np.testing.assert_allclose(flat_ft_params[n], flat_pt_params[n])

        ### MLP head should be reinitialized from scratch
        if 'vision_model/head' in n:
            np.testing.assert_allclose(flat_ft_params[n], np.zeros_like(flat_ft_params[n]))

if __name__ == '__main__':
    # pylint: disable=g-import-not-at-top
    import gin
    from absl import app, flags

    from t5x import gin_utils
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

    def main(argv: Sequence[str]):
        """Wrapper for pdb post mortems."""
        _main(argv)

    def _main(argv: Sequence[str]):
        """True main function."""
        if len(argv) > 1:
          raise app.UsageError('Too many command-line arguments.')

        save_using_gin = gin.configurable(convert_t5x_finetune_to_pretrain)

        gin_utils.parse_gin_flags(
            # User-provided gin paths take precedence if relative paths conflict.
            FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
            FLAGS.gin_file,
            FLAGS.gin_bindings)
        save_using_gin()
        jax.effects_barrier()

    gin_utils.run(main)
