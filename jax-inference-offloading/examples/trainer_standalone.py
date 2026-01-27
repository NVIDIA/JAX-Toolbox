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
- MODEL_NAME: HuggingFace model name (e.g., "meta-llama/Llama-3.2-1B-Instruct")
- MODEL_PATH: Path to HuggingFace model checkpoint (optional, uses dummy weights if not set)
- PARAM_MAPPING_PATH: Path to JSON parameter mapping file (optional, uses hardcoded mappings if not set)
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
from jax_inference_offloading.models import get_named_parameters

from transformers import AutoTokenizer

# =============================================================================
# CUSTOM MODEL LOADING
# =============================================================================
# IMPORTANT: This example uses the model implementation in Tunix, Thus
# the model loading below uses Tunix's load_model function.
# When integrating with your own RL framework (OpenRLHF, TRL, custom, etc.),
# you MUST replace this with your own model loading code that interfaces with
# your own model implementation.
#
# NOTE: Tunix's load_model requires MODEL_NAME because it uses regex matching
# on the model name to determine which architecture/config to use. A custom
# loader could instead read the model type from checkpoint_path/config.json.
#
# Your custom load_model function should:
# 1. Load the model architecture specific to your framework
# 2. Load checkpoint weights from MODEL_PATH (HuggingFace safetensors format)
# 3. Return a model object from which parameters can be extracted
#
# The key requirement is that parameters must have JAX-compatible shapes that
# match the parameter mapping (see examples/mappings/ for JSON mapping format).
# The mapping defines how JAX parameter shapes are transformed to vLLM shapes
# during weight transfer.
#
# Example custom implementation:
#
#   def load_model(checkpoint_path, mesh, dtype):
#       # 1. Create your model architecture (could infer from config.json)
#       config = json.load(open(f"{checkpoint_path}/config.json"))
#       model = MyCustomModel(config)
#
#       # 2. Load weights from checkpoint
#       model = load_weights_from_safetensors(model, checkpoint_path)
#
#       # 3. Shard across mesh if needed
#       model = shard_model(model, mesh)
#
#       return model
#
# =============================================================================
from jax_inference_offloading.integrations.tunix.load_model import load_model

timer = Timer()

# --- Configuration ---
model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
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
# NOTE: Replace this section with your own model loading code.
# The load_model function must return a model compatible with get_named_parameters().
with timer.section("load_model"):
    model = load_model(
        model_name,
        mesh,
        checkpoint_path=model_path,
        dtype=jnp.bfloat16,
        random_seed=42,
    )

# Extract named parameters for transfer
# This flattens the model's parameter tree into a dict with dot-separated keys
# e.g., {"model.layers.0.attn.q_proj.w": array(...), ...}
params = get_named_parameters(model)

# --- Create VLLMRolloutEngine (framework-agnostic) ---
if jax.process_index() == 0:
    print(f"Creating VLLMRolloutEngine with gateway_url={gateway_url}")

with timer.section("create_engine"):
    engine = VLLMRolloutEngine(
        gateway_url=gateway_url,
        mesh=mesh,
        model_name=model_name,
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
