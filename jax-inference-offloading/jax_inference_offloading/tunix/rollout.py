#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rollout worker with offloading to vLLM."""
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import jaxtyping

import jax_inference_offloading.api.controller_pb2 as ctrl
from jax_inference_offloading.jax import OffloadingBridge
from jax_inference_offloading.timer import Timer
from tunix.rl.rollout.base_rollout import BaseRollout, RolloutConfig, RolloutOutput


class VllmGPURollout(BaseRollout):

  def __init__(
    self,
    gateway_url,
    model_name,
    *,
    rollout_actor,  # AKA rollout model
    tokenizer,
    mesh,
    rollout_config,
    extra_stop_tokens: list[str] | None = None,
    transfer_mode: str = 'fused',
    timer: Any | None = None,
  ):
    self._timer = timer or Timer()
    self._tokenizer = tokenizer
    self._bridge = OffloadingBridge(
      gateway_url=gateway_url,
      model_name=model_name,
      mesh=mesh,
      transfer_mode=transfer_mode,
      timer=self._timer,
    )
    self._extra_stop_token_ids = []
    for t in extra_stop_tokens or []:
      i = self._tokenizer.encode(t)
      assert len(i) == 1, f"Stop token {t} must be a single token, got {i}"
      self._extra_stop_token_ids.extend(i)

  def generate(
      self,
      prompts: list[str],
      rollout_config: RolloutConfig,
  ):
    """Generates samples from the model."""
    with self._timer.section("rollout.generate"):
      remote_rollout_config = ctrl.RolloutConfig(
        top_p=rollout_config.top_p,
        top_k=rollout_config.top_k,
        temperature=rollout_config.temperature,
        max_tokens=rollout_config.max_tokens_to_generate,
        seed=rollout_config.seed,
      )
      if rollout_config.eos_tokens is not None:
        remote_rollout_config.stop_token_ids.extend(rollout_config.eos_tokens)
      else:
        remote_rollout_config.stop_token_ids.extend([self._tokenizer.eos_id()])
      remote_rollout_config.stop_token_ids.extend(self._extra_stop_token_ids)

      with self._timer.section("inference"):
        response = self._bridge.gateway.inference([str(p) for p in prompts], config=remote_rollout_config)

      with self._timer.section("process_outputs"):
        generated_text = []
        input_tokens = []
        output_tokens = []

        def pad_to_left(original, length, pad_value):
          assert len(original) <= length
          return [pad_value] * (length - len(original)) + original

        def pad_to_right(original, length, pad_value):
          assert len(original) <= length
          return original + [pad_value] * (length - len(original))

        for i, output in enumerate(response.outputs):
          if i < 1:
            print(f"# Rollout {i} of {len(prompts)}")
            print(f"## Prompt:\n{prompts[i]}")
            print(f"## Response:\n{output.generated_text}")
            print("-" * 80)
          generated_text.append(output.generated_text)
          input_tokens.append(
            pad_to_left(list(output.tokenized_prompt.ids), rollout_config.max_prompt_length, self._tokenizer.pad_id())
          )
          output_tokens.append(
            pad_to_right(list(output.generated_tokens.ids), rollout_config.max_tokens_to_generate, self._tokenizer.pad_id())
          )

    return RolloutOutput(
      text=generated_text,
      logits=[],  # not needed for GRPO
      tokens=jnp.array(output_tokens, dtype=jnp.int32),
      left_padded_prompt_tokens=jnp.array(input_tokens, dtype=jnp.int32),
      logprobs=None,  # needed for GRPO, GRPOLearner will recalc
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      completion_mask: jax.Array | None = None,
  ) -> jax.Array:
    raise NotImplementedError()

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    """Updates the rollout model parameters."""
    with self._timer.section("rollout.update_params"):
      self._bridge.transfer(params)

  def pad_id(self) -> int:
    return self._tokenizer.pad_id()

  def eos_id(self) -> int:
    return self._tokenizer.eos_id()

  def model(self):
    return None

  def shutdown(self) -> None:
    """Gracefully shutdown the remote gateway if available."""
    try:
      self._bridge.gateway.shutdown()
    except Exception:
      # Ignore shutdown errors; process teardown or remote unavailability is expected.
      pass

  def __del__(self):
    try:
      self.shutdown()
    except Exception:
      # Suppress destructor-time errors during interpreter shutdown.
      pass
