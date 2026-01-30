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

This example demonstrates how to use OffloadingSession, VLLMTransferEngine,
and VLLMRolloutEngine directly without depending on any external framework.
This is useful for:
- Custom RL training loops
- Integration with other frameworks (OpenRLHF, TRL, custom, etc.)
- Testing and benchmarking

The architecture separates concerns:
- MappingConfig: Loads mesh_axes and parameter mappings from JSON
- load_checkpoint_to_jax: Loads HuggingFace checkpoints into JAX arrays
- OffloadingSession: Handles gRPC connection and handshake
- VLLMTransferEngine: Handles weight transfer from JAX to vLLM
- VLLMRolloutEngine: Handles inference/rollout generation

Prerequisites:
- Gateway server running (python -m jax_inference_offloading.controller.gateway)
- vLLM rollout worker running (python examples/rollout.py)

Environment variables:
- GATEWAY_URL: URL of the gateway server (e.g., "localhost:50051")
- MODEL_PATH: Path to HuggingFace model checkpoint (required)
- PARAM_MAPPING_PATH: Path to JSON parameter mapping file (required)
"""

import os

import jax
import jax.numpy as jnp

# Framework-agnostic imports from jax-inference-offloading
from jax_inference_offloading import (
    InferenceConfig,
    OffloadingSession,
    VLLMRolloutEngine,
    VLLMTransferEngine,
)
from jax_inference_offloading.timer import Timer
from jax_inference_offloading.models.checkpoint import (
    load_mapping_config,
    load_checkpoint_to_jax,
)

from transformers import AutoTokenizer

timer = Timer()

# --- Configuration ---
model_path = os.environ.get("MODEL_PATH", None)
param_mapping_path = os.environ.get("PARAM_MAPPING_PATH", None)
gateway_url = os.environ.get("GATEWAY_URL", "localhost:50051")
transfer_mode = os.environ.get("TRANSFER_MODE", "grouped")

# Validate: both model_path and param_mapping_path are required
if model_path is None:
    raise ValueError("MODEL_PATH environment variable is required")
if param_mapping_path is None:
    raise ValueError("PARAM_MAPPING_PATH environment variable is required")

# Load tokenizer for pad_id
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# --- Load Mapping Config ---
# This reads the JSON mapping file to get mesh_axes and parameter mappings
if jax.process_index() == 0:
    print(f"Loading mapping config from {param_mapping_path}")

with timer.section("load_mapping_config"):
    mapping_config = load_mapping_config(param_mapping_path)

# Create mesh using axis names from the mapping config
mesh_shape = (jax.process_count(), jax.local_device_count())
mesh_axes = tuple(mapping_config.mesh_axes)
if len(mesh_shape) != len(mesh_axes):
    raise ValueError(
        f"Mesh shape {mesh_shape} does not match mesh_axes {mesh_axes}. "
        f"Expected {len(mesh_shape)} axes, got {len(mesh_axes)}."
    )
mesh = jax.make_mesh(mesh_shape, mesh_axes)

if jax.process_index() == 0:
    print(f"Created mesh with shape {mesh_shape} and axes {mesh_axes}")

# --- Load Checkpoint ---
# Load HuggingFace checkpoint directly into JAX arrays using the mapping
if jax.process_index() == 0:
    print(f"Loading checkpoint from {model_path}")

with timer.section("load_checkpoint"):
    params = load_checkpoint_to_jax(
        checkpoint_path=model_path,
        mapping_specs=mapping_config.mapping_specs,
        mesh=mesh,
        dtype=jnp.bfloat16,
    )

if jax.process_index() == 0:
    print(f"Loaded {len(params)} parameters")

# --- Create OffloadingSession and Engines ---
if jax.process_index() == 0:
    print(f"Creating OffloadingSession with gateway_url={gateway_url}")

with timer.section("create_session"):
    session = OffloadingSession(
        gateway_url=gateway_url,
        mesh=mesh,
        model_path=model_path,
        param_mapping_path=param_mapping_path,
    )

if jax.process_index() == 0:
    print("Creating VLLMTransferEngine and VLLMRolloutEngine...")

with timer.section("create_engines"):
    transfer_engine = VLLMTransferEngine(
        session=session,
        transfer_mode=transfer_mode,
        timer=timer,
    )
    rollout_engine = VLLMRolloutEngine(
        session=session,
        timer=timer,
    )

# --- Transfer Weights ---
SKIP_TRANSFER = os.environ.get("SKIP_TRANSFER", "0") == "1"

if SKIP_TRANSFER:
    if jax.process_index() == 0:
        print("SKIPPING weight transfer (SKIP_TRANSFER=1)")
else:
    if jax.process_index() == 0:
        print("Transferring weights to vLLM...")

    with timer.section("warmup_transfer"):
        transfer_engine.update_weights(params)

    if jax.process_index() == 0:
        print("Weights transferred successfully!")

    # --- Benchmark weight transfer ---
    for r in range(3):
        with timer.section(f"transfer.run{r}"):
            transfer_engine.update_weights(params)

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
    output = rollout_engine.generate(["Messi's barcelona career:"], config)

    print("\n--- Text Prompt ---")
    print(f"Prompt: Messi's barcelona career:")
    print(f"Response: {output.texts[0]}")

    # Example 2: Multiple prompts with multiple outputs per prompt
    config_multi = InferenceConfig(
        max_tokens=100,
        temperature=0.9,
        top_p=0.95,
        n=2,  # Generate 2 completions per prompt
    )
    prompts = [
        "Messi's barcelona career:",
        "Name a color:",
    ]
    output = rollout_engine.generate(prompts, config_multi)

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
rollout_engine.shutdown()
session.shutdown()
if jax.process_index() == 0:
    print("\nShutdown complete. Exiting.")
