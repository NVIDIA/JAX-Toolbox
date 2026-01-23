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
"""Tunix adapter for vLLM rollout offloading.

This module provides VllmGPURollout, a Tunix BaseRollout implementation
that delegates to the framework-agnostic VLLMRolloutEngine.
"""
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jaxtyping

from jax_inference_offloading.api import InferenceConfig
from jax_inference_offloading.engines import VLLMRolloutEngine
from jax_inference_offloading.timer import Timer
from tunix.rl.rollout.base_rollout import BaseRollout, RolloutConfig, RolloutOutput


class VllmGPURollout(BaseRollout):
    """Tunix adapter wrapping VLLMRolloutEngine.

    This class implements Tunix's BaseRollout interface by delegating
    to the framework-agnostic VLLMRolloutEngine. It handles the conversion
    between Tunix's RolloutConfig/RolloutOutput and the bridge's
    InferenceConfig/InferenceOutput.
    """

    def __init__(
        self,
        gateway_url: str,
        model_name: str,
        *,
        rollout_actor,  # AKA rollout model (unused for remote engine)
        tokenizer,
        mesh: jax.sharding.Mesh,
        rollout_config: RolloutConfig,  # Initial config (unused, passed per-call)
        extra_stop_tokens: List[str] | None = None,
        transfer_mode: str = "fused",
        timer: Any | None = None,
    ):
        """Initialize the Tunix vLLM rollout adapter.

        Args:
            gateway_url: URL of the gateway server.
            model_name: HuggingFace model name for tensor mapping.
            rollout_actor: The rollout model (unused for remote engine).
            tokenizer: Tunix tokenizer for encoding/decoding.
            mesh: JAX device mesh.
            rollout_config: Initial rollout config (unused, provided per generate call).
            extra_stop_tokens: Additional stop tokens as strings.
            transfer_mode: Weight transfer mode ('fused', 'unfused', 'grouped').
            timer: Optional timer for profiling.
        """
        del rollout_actor  # Not used for remote engine
        del rollout_config  # Config passed per generate() call

        self._timer = timer or Timer()
        self._tokenizer = tokenizer

        # Resolve extra stop tokens to IDs
        self._extra_stop_token_ids: List[int] = []
        for t in extra_stop_tokens or []:
            token_ids = self._tokenizer.encode(t)
            assert len(token_ids) == 1, f"Stop token {t} must be a single token, got {token_ids}"
            self._extra_stop_token_ids.extend(token_ids)

        # Delegate to the framework-agnostic engine
        self._engine = VLLMRolloutEngine(
            gateway_url=gateway_url,
            model_name=model_name,
            mesh=mesh,
            transfer_mode=transfer_mode,
            timer=self._timer,
        )

    def generate(
        self,
        prompts: List[str],
        rollout_config: RolloutConfig,
    ) -> RolloutOutput:
        """Generate completions for the given prompts.

        Args:
            prompts: List of text prompts.
            rollout_config: Tunix rollout configuration.

        Returns:
            Tunix RolloutOutput with generated samples.
        """
        with self._timer.section("rollout.generate"):
            # Convert Tunix RolloutConfig -> InferenceConfig
            stop_token_ids = list(self._extra_stop_token_ids)
            if rollout_config.eos_tokens is not None:
                stop_token_ids = list(rollout_config.eos_tokens) + stop_token_ids
            else:
                stop_token_ids = [self._tokenizer.eos_id()] + stop_token_ids

            config = InferenceConfig(
                max_tokens=rollout_config.max_tokens_to_generate,
                temperature=rollout_config.temperature,
                top_p=rollout_config.top_p if rollout_config.top_p is not None else 1.0,
                top_k=rollout_config.top_k if rollout_config.top_k is not None else -1,
                seed=rollout_config.seed,
                stop_token_ids=stop_token_ids,
            )

            # Call engine
            with self._timer.section("inference"):
                output = self._engine.generate([str(p) for p in prompts], config)

            # Convert InferenceOutput -> Tunix RolloutOutput
            with self._timer.section("process_outputs"):
                generated_text = []
                input_tokens = []
                output_tokens = []

                def pad_to_left(original: List[int], length: int, pad_value: int) -> List[int]:
                    assert len(original) <= length, f"Sequence too long: {len(original)} > {length}"
                    return [pad_value] * (length - len(original)) + original

                def pad_to_right(original: List[int], length: int, pad_value: int) -> List[int]:
                    assert len(original) <= length, f"Sequence too long: {len(original)} > {length}"
                    return original + [pad_value] * (length - len(original))

                for i, completion in enumerate(output.completions):
                    if i < 1:
                        print(f"# Rollout {i} of {len(prompts)}")
                        print(f"## Prompt:\n{prompts[i]}")
                        print(f"## Response:\n{completion.text}")
                        print("-" * 80)

                    generated_text.append(completion.text)
                    input_tokens.append(
                        pad_to_left(
                            completion.prompt_token_ids or [],
                            rollout_config.max_prompt_length,
                            self._tokenizer.pad_id(),
                        )
                    )
                    output_tokens.append(
                        pad_to_right(
                            completion.token_ids,
                            rollout_config.max_tokens_to_generate,
                            self._tokenizer.pad_id(),
                        )
                    )

        return RolloutOutput(
            text=generated_text,
            logits=[],  # Not needed for GRPO
            tokens=jnp.array(output_tokens, dtype=jnp.int32),
            left_padded_prompt_tokens=jnp.array(input_tokens, dtype=jnp.int32),
            logprobs=None,  # GRPOLearner will recalculate
        )

    def get_per_token_logps(
        self,
        prompt_tokens: jax.Array,
        completion_tokens: jax.Array,
        completion_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Get per-token log probabilities.

        Not implemented for remote engine - use GRPOLearner's recalculation.
        """
        raise NotImplementedError(
            "get_per_token_logps is not supported for remote vLLM engine. "
            "Use GRPOLearner which recalculates logprobs locally."
        )

    def update_params(
        self,
        params: jaxtyping.PyTree,
        filter_types: Optional[Tuple[Any, ...]] = None,
    ) -> None:
        """Update the rollout model parameters.

        Args:
            params: Model parameters to transfer.
            filter_types: Unused for remote engine.
        """
        del filter_types  # Not used for remote engine
        with self._timer.section("rollout.update_params"):
            self._engine.update_weights(params)

    def pad_id(self) -> int:
        """Return the padding token ID."""
        return self._tokenizer.pad_id()

    def eos_id(self) -> int:
        """Return the end-of-sequence token ID."""
        return self._tokenizer.eos_id()

    def model(self):
        """Return the local model (None for remote engine)."""
        return None

    def shutdown(self) -> None:
        """Gracefully shutdown the remote gateway."""
        self._engine.shutdown()

    def __del__(self):
        """Destructor - attempt graceful shutdown."""
        try:
            self.shutdown()
        except Exception:
            # Suppress destructor-time errors during interpreter shutdown.
            pass
