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
"""Framework-agnostic types for inference offloading."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class InferenceConfig:
    """Framework-agnostic inference configuration.

    Maps to vLLM SamplingParams.

    Attributes:
        max_tokens: Maximum number of tokens to generate per output sequence.
        temperature: Temperature for sampling. 0.0 = greedy, higher = more random.
        top_p: Top-p (nucleus) sampling. 1.0 = no filtering.
        top_k: Top-k sampling. -1 = no filtering.
        n: Number of output sequences per prompt (for best-of-n, GRPO groups, etc.).
        seed: Random seed for reproducibility.
        stop_token_ids: Stop token IDs (e.g., EOS tokens).
    """

    max_tokens: int = 64
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = -1
    n: int = 1
    seed: Optional[int] = None
    stop_token_ids: List[int] = field(default_factory=list)


@dataclass
class CompletionOutput:
    """Single completion/output from the model.

    Attributes:
        text: Generated text.
        token_ids: Generated token IDs.
        logprobs: Log probabilities per generated token (optional).
        prompt_token_ids: Prompt token IDs (useful for log-prob calculations).
    """

    text: str
    token_ids: List[int]
    logprobs: Optional[List[float]] = None
    prompt_token_ids: Optional[List[int]] = None


@dataclass
class InferenceOutput:
    """Output from inference/rollout generation.

    Contains one or more CompletionOutput per prompt (based on config.n).

    Attributes:
        completions: List of completions, flattened across all prompts.
            Length = num_prompts * config.n
    """

    completions: List[CompletionOutput]

    @property
    def texts(self) -> List[str]:
        """Get all generated texts."""
        return [c.text for c in self.completions]

    @property
    def token_ids(self) -> List[List[int]]:
        """Get all generated token ID sequences."""
        return [c.token_ids for c in self.completions]

    def to_arrays(
        self,
        max_prompt_length: int,
        max_completion_length: int,
        pad_id: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Convert to padded numpy arrays for training.

        Args:
            max_prompt_length: Maximum prompt length (for left-padding).
            max_completion_length: Maximum completion length (for right-padding).
            pad_id: Padding token ID.

        Returns:
            dict with keys:
                - 'prompt_tokens': [batch, max_prompt_length] left-padded
                - 'completion_tokens': [batch, max_completion_length] right-padded
                - 'completion_logprobs': [batch, max_completion_length] if available
        """

        def pad_left(seq: List[int], length: int, pad_value: int) -> List[int]:
            seq = seq[:length]  # Truncate if too long
            return [pad_value] * (length - len(seq)) + seq

        def pad_right(seq: List[Any], length: int, pad_value: Any) -> List[Any]:
            seq = seq[:length]  # Truncate if too long
            return seq + [pad_value] * (length - len(seq))

        result: Dict[str, np.ndarray] = {
            "prompt_tokens": np.array(
                [
                    pad_left(c.prompt_token_ids or [], max_prompt_length, pad_id)
                    for c in self.completions
                ],
                dtype=np.int32,
            ),
            "completion_tokens": np.array(
                [
                    pad_right(c.token_ids, max_completion_length, pad_id)
                    for c in self.completions
                ],
                dtype=np.int32,
            ),
        }

        if all(c.logprobs is not None for c in self.completions):
            result["completion_logprobs"] = np.array(
                [
                    pad_right(c.logprobs or [], max_completion_length, 0.0)
                    for c in self.completions
                ],
                dtype=np.float32,
            )

        return result
