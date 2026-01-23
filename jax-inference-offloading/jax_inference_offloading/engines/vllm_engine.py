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
"""vLLM-based rollout engine using the JAX-vLLM offloading bridge."""

from typing import Dict, List, Optional, Union

import jax

from jax_inference_offloading.api.types import (
    CompletionOutput,
    InferenceConfig,
    InferenceOutput,
)
from jax_inference_offloading.jax import OffloadingBridge
from jax_inference_offloading.models import flatten_state, get_named_parameters
from jax_inference_offloading.timer import Timer
import jax_inference_offloading.api.controller_pb2 as ctrl


class VLLMRolloutEngine:
    """vLLM-based rollout engine for inference offloading.

    This is the main entry point for users who want to use vLLM
    for inference offloading without depending on Tunix or other
    RL frameworks.

    Example:
        >>> engine = VLLMRolloutEngine(
        ...     gateway_url="localhost:50051",
        ...     model_name="meta-llama/Llama-3.1-8B-Instruct",
        ...     mesh=jax.make_mesh((8,), ("tp",)),
        ... )
        >>>
        >>> # Transfer weights from JAX model
        >>> engine.update_weights(my_jax_params)
        >>>
        >>> # Generate completions
        >>> config = InferenceConfig(max_tokens=128, temperature=0.9)
        >>> output = engine.generate(["What is 2+2?"], config)
        >>> print(output.texts[0])
    """

    def __init__(
        self,
        gateway_url: str,
        model_name: str,
        mesh: jax.sharding.Mesh,
        *,
        transfer_mode: str = "grouped",
        timer: Optional[Timer] = None,
    ):
        """Initialize the vLLM rollout engine.

        Args:
            gateway_url: URL of the gateway server (e.g., "localhost:50051").
            model_name: HuggingFace model name for tensor mapping resolution.
            mesh: JAX device mesh for sharded parameter handling.
            transfer_mode: Weight transfer mode ('fused', 'unfused', 'grouped').
            timer: Optional timer for performance profiling.
        """
        self._timer = timer or Timer()
        self._bridge = OffloadingBridge(
            gateway_url=gateway_url,
            model_name=model_name,
            mesh=mesh,
            transfer_mode=transfer_mode,
            timer=self._timer,
        )

    def generate(
        self,
        prompts: Union[List[str], List[List[int]]],
        config: InferenceConfig,
    ) -> InferenceOutput:
        """Generate completions using vLLM.

        Args:
            prompts: Text prompts or pre-tokenized prompts.
            config: Inference configuration.

        Returns:
            InferenceOutput with generated completions.
        """
        # Build protobuf config
        proto_config = ctrl.RolloutConfig(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            num_outputs=config.n,
            seed=config.seed or 42,
        )
        proto_config.stop_token_ids.extend(config.stop_token_ids)

        # Call gateway
        with self._timer.section("inference"):
            response = self._bridge.gateway.inference(prompts, config=proto_config)

        # Convert response to framework-agnostic output
        completions = []
        for output in response.outputs:
            completions.append(
                CompletionOutput(
                    text=output.generated_text,
                    token_ids=list(output.generated_tokens.ids),
                    logprobs=(
                        list(output.generated_token_logps)
                        if output.generated_token_logps
                        else None
                    ),
                    prompt_token_ids=(
                        list(output.tokenized_prompt.ids)
                        if output.tokenized_prompt.ids
                        else None
                    ),
                )
            )

        return InferenceOutput(completions=completions)

    def update_weights(
        self,
        params: Union[Dict[str, jax.Array], "nnx.State", "nnx.Module"],  # noqa: F821
        *,
        block: bool = True,
    ) -> None:
        """Transfer model weights to vLLM.

        Args:
            params: Model parameters in various formats:
                - Dict[str, jax.Array]: Direct flattened params
                - flax.nnx.State: Flax state object
                - flax.nnx.Module: Flax module (state extracted automatically)
            block: If True, wait for transfer completion (always True currently).
        """
        del block  # Currently always blocking

        with self._timer.section("update_weights"):
            # Handle different input formats
            if isinstance(params, dict):
                named_params = params
            else:
                # Try flax.nnx formats
                try:
                    from flax import nnx

                    if isinstance(params, nnx.Module):
                        named_params = get_named_parameters(params)
                    elif isinstance(params, nnx.State):
                        named_params = flatten_state(params)
                    else:
                        raise TypeError(f"Unsupported params type: {type(params)}")
                except ImportError:
                    raise TypeError(
                        f"Unsupported params type: {type(params)}. "
                        "Expected Dict[str, jax.Array] or install flax for nnx support."
                    )

            # Transfer via bridge
            self._bridge.transfer(named_params)

    def shutdown(self) -> None:
        """Shutdown the gateway connection."""
        try:
            self._bridge.gateway.shutdown()
        except Exception:
            pass  # Ignore shutdown errors

    @property
    def gateway(self):
        """Access the underlying gateway client for advanced usage."""
        return self._bridge.gateway

    @property
    def timer(self) -> Timer:
        """Access the timer for performance analysis."""
        return self._timer

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures shutdown is called."""
        self.shutdown()
        return False
