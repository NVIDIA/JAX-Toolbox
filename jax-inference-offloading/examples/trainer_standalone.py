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
"""Standalone example: Using jax-inference-offloading without Tunix.

This example demonstrates how to use the VLLMRolloutEngine directly
without depending on the Tunix RL framework. This is useful for:
- Custom RL training loops
- Integration with other frameworks (OpenRLHF, TRL, etc.)
- Testing and benchmarking

Prerequisites:
- Gateway server running (python -m jax_inference_offloading.controller.gateway)
- vLLM rollout worker running (python examples/rollout.py)

Environment variables:
- GATEWAY_URL: URL of the gateway server (e.g., "localhost:50051")
- MODEL_NAME: HuggingFace model name (default: "meta-llama/Llama-3.1-8B-Instruct")
- MODEL_PATH: Path to model checkpoint (optional, uses dummy weights if not set)
"""

import os

import jax
import jax.numpy as jnp

# Framework-agnostic imports from jax-inference-offloading
from jax_inference_offloading import (
    InferenceConfig,
    VLLMRolloutEngine,
)
from jax_inference_offloading.timer import Timer

# For model loading, we still use the Tunix integration helper
# In a real custom setup, you would load your model with your own code
from jax_inference_offloading.integrations.tunix.load_model import load_model
from jax_inference_offloading.models import get_named_parameters

from transformers import AutoTokenizer

timer = Timer()

# --- Configuration ---
model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
model_path = os.environ.get("MODEL_PATH", None)
gateway_url = os.environ.get("GATEWAY_URL", "localhost:50051")
transfer_mode = os.environ.get("TRANSFER_MODE", "grouped")

# Load tokenizer for pad_id
tokenizer = AutoTokenizer.from_pretrained(model_path or model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Create mesh
mesh = jax.make_mesh((jax.process_count(), jax.local_device_count()), ("fsdp", "tp"))

# --- Load Model ---
if jax.process_index() == 0:
    print(f"Loading JAX model {model_name} @ {model_path}")

with timer.section("load_model"):
    model = load_model(
        model_name,
        mesh,
        checkpoint_path=model_path,
        dtype=jnp.bfloat16,
        random_seed=42,
    )

# Extract named parameters for transfer
params = get_named_parameters(model)

# --- Create VLLMRolloutEngine (framework-agnostic) ---
if jax.process_index() == 0:
    print(f"Creating VLLMRolloutEngine with gateway_url={gateway_url}")

with timer.section("create_engine"):
    engine = VLLMRolloutEngine(
        gateway_url=gateway_url,
        model_name=model_name,
        mesh=mesh,
        transfer_mode=transfer_mode,
        timer=timer,
    )

# --- Transfer Weights ---
if jax.process_index() == 0:
    print("Transferring weights to vLLM...")

with timer.section("warmup_transfer"):
    engine.update_weights(params)

if jax.process_index() == 0:
    print("Weights transferred successfully!")

# --- Benchmark weight transfer ---
for r in range(3):
    with timer.section(f"transfer.run{r}"):
        engine.update_weights(params)

# --- Generate Completions ---
if jax.process_index() == 0:
    print("\n" + "=" * 80)
    print("Generating completions...")
    print("=" * 80)

    # Example 1: Simple text prompt
    config = InferenceConfig(
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
    )
    output = engine.generate(["Quick facts about the moon:"], config)

    print("\n--- Text Prompt ---")
    print(f"Prompt: Quick facts about the moon:")
    print(f"Response: {output.texts[0]}")

    # Example 2: Multiple prompts with multiple outputs per prompt
    config_multi = InferenceConfig(
        max_tokens=100,
        temperature=0.9,
        top_p=0.95,
        n=2,  # Generate 2 completions per prompt
    )
    prompts = [
        "What is 2 + 2?",
        "Name a color:",
    ]
    output = engine.generate(prompts, config_multi)

    print("\n--- Multiple Prompts (n=2) ---")
    for i, completion in enumerate(output.completions):
        print(f"\nCompletion {i + 1}:")
        print(f"  Text: {completion.text[:100]}...")
        print(f"  Token count: {len(completion.token_ids)}")

    # Example 3: Using to_arrays() for training
    arrays = output.to_arrays(
        max_prompt_length=64,
        max_completion_length=100,
        pad_id=tokenizer.pad_token_id,
    )
    print("\n--- Arrays for Training ---")
    print(f"  prompt_tokens shape: {arrays['prompt_tokens'].shape}")
    print(f"  completion_tokens shape: {arrays['completion_tokens'].shape}")

# --- Print timing summary ---
if jax.process_index() == 0:
    print("\n" + "=" * 80)
    print("Timing Summary")
    print("=" * 80)
    timer.summary(sort_by="name", precision=3)

# --- Shutdown ---
if jax.process_index() == 0:
    engine.shutdown()
    print("\nEngine shutdown complete. Exiting.")
