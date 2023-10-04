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
"""Tests for rosetta.projects.diffusion.augmentations."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import sys

class AugTest(absltest.TestCase):

  def test_text_cond_aug_array(self):
      in_arr = jnp.ones((3, 4, 2))
      rng_in = jax.random.PRNGKey(0)
      masked, rng = augmentations.text_conditioning_dropout(in_arr, rng_in, dropout_rate=0.5)

      output = jnp.ones((3,4,2))
      output = output.at[0, :].set(0)
      output = output.at[1, :].set(0)
      assert jnp.allclose(output, masked), f'expected: {output}, got: {masked}'
      assert (rng_in != rng).all()
      assert rng is not None

  def test_text_cond_aug_mapping(self):
      in_arr = {'text_mask': jnp.ones((3, 4, 2)), 'another_key':jnp.ones((1, 4, 2))}
      rng = jax.random.PRNGKey(0)
      masked, rng = augmentations.text_conditioning_dropout(in_arr, rng, dropout_rate=0.5)

      output = jnp.ones((3,4,2))
      output = output.at[0, :].set(0)
      output = output.at[1, :].set(0)
      masked_arr = masked['text_mask']
      assert jnp.allclose(output, masked_arr), f'expected: {output}, got: {masked_arr}'
      masked_ones = masked['another_key']
      assert jnp.allclose(masked_ones, jnp.ones((1, 4, 2))), f'expected ones, got: {masked_ones}'
      assert list(masked.keys()) == ['text_mask', 'another_key'], 'modified keys in batch'

  def test_text_cond_aug_array_preserved(self):
      in_arr = jnp.ones((3, 4, 2))
      in_arr = in_arr.at[2, 2:, 1:].set(0)
      rng = jax.random.PRNGKey(0)
      masked, rng = augmentations.text_conditioning_dropout(in_arr, rng, dropout_rate=0.5)

      output = jnp.ones((3,4,2))
      output = output.at[0, :].set(0)
      output = output.at[1, :].set(0)
      output = output.at[2, 2:, 1:].set(0)
      assert jnp.allclose(output, masked)

  def test_text_cond_aug_jit(self):
      in_arr = jnp.ones((3, 4, 2))
      in_arr = in_arr.at[2, 2:, 1:].set(0)
      rng = jax.random.PRNGKey(0)
      masked, rng = jax.jit(augmentations.text_conditioning_dropout)(in_arr, rng, dropout_rate=0.5)
      output = jnp.ones((3,4,2))
      output = output.at[0, :].set(0)
      output = output.at[1, :].set(0)
      output = output.at[2, 2:, 1:].set(0)
      assert jnp.allclose(output, masked)


if __name__ == '__main__':
  sys.path.append('../')
  import rosetta.projects.diffusion.augmentations as augmentations
  absltest.main()