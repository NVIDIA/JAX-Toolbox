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
"""JAX-vLLM GRPO workflow example (JAX-side)"""

import logging
import os
import re
import tempfile
from functools import partial

import grain
import jax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds
from flax import nnx
from tunix.generate.tokenizer_adapter import Tokenizer
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rl_cluster import ClusterConfig, RLCluster, RLTrainingConfig
from tunix.rl.rl_cluster import Role as ClusterRole
from tunix.rl.rollout import base_rollout

from jax_inference_offloading.timer import Timer
from jax_inference_offloading.tunix.load_model import load_model
from jax_inference_offloading.tunix.rollout import VllmGPURollout

logger = logging.getLogger(__name__)
timer = Timer()

ROLLOUT_ENGINE = os.environ.get('ROLLOUT_ENGINE', 'vllm_gpu')  # or 'vanilla'
TRANSFER_MODE = os.environ.get('TRANSFER_MODE', 'grouped')  # 'grouped' | 'fused' | 'unfused' 

#-------------------------- Config & Hyperparameters --------------------------#

# Helper for hyperparameter overrides via environment variables
def get_env(name, default):
  raw = os.environ.pop(name, None)
  return type(default)(raw) if raw is not None else default

# ====== Data ======
SCRATCHDIR = os.environ.get("SCRATCHDIR", "/content")
DATASET_DIR = os.environ.get("DATASET_DIR")
if not DATASET_DIR:
  TEMPDIR = tempfile.mkdtemp()
  DATASET_DIR = os.path.join(TEMPDIR, "tfds")
TRAIN_FRACTION = get_env("GRPO_DATA_TRAIN_FRACTION", 1.0)

# ====== Sharding ======
MESH = [(jax.process_count(), jax.local_device_count()), ("fsdp", "tp")]

# ====== Rollout ======
MAX_PROMPT_LENGTH = get_env("GRPO_MAX_PROMPT_LENGTH", 256)
TOTAL_GENERATION_STEPS = get_env("GRPO_TOTAL_GENERATION_STEPS", 512)
TEMPERATURE = get_env("GRPO_TEMPERATURE", 0.9)
TOP_P = get_env("GRPO_TOP_P", 0.95)
TOP_K = get_env("GRPO_TOP_K", 50)

# === GRPO hyperparameters ===
# number of inner-loop policy update steps per batch of data, (𝜇 in GRPO algo 1).
NUM_ITERATIONS = get_env("GRPO_NUM_ITERATIONS", 1)
# The coefficient for the KL divergence penalty (𝛽) in the GRPO loss function.
BETA = get_env("GRPO_BETA", 0.03)
# Epsilon value for clipping (𝜀 in GRPO loss in paper).
EPSILON = get_env("GRPO_EPSILON", 0.15)

# ====== Training ======
TRAIN_MICRO_BATCH_SIZE = get_env("GRPO_TRAIN_MICRO_BATCH_SIZE", 8)
NUM_BATCHES = get_env("GRPO_NUM_BATCHES", 5)
# Number of rollouts per prompt per step (`G` in Algo 1); 'group' in GRPO.
NUM_GENERATIONS = get_env("GRPO_NUM_GENERATIONS", 8)
EVAL_EVERY_N_STEPS = get_env("GRPO_EVAL_EVERY_N_STEPS", 10)
NUM_EPOCHS = get_env("GRPO_NUM_EPOCHS", 1)

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# === Optimizer hyperparameters ===
LEARNING_RATE = get_env("GRPO_LEARNING_RATE", 5e-6)
B1 = get_env("GRPO_B1", 0.9)
B2 = get_env("GRPO_B2", 0.95)
WEIGHT_DECAY = get_env("GRPO_WEIGHT_DECAY", 0.02) 

#--------------------------- Load Model & Tokenizer ---------------------------#

model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
model_path = os.environ.get("MODEL_PATH", None)

with timer.section("load_checkpoint"):
  with timer.section("model"):
    # Reference model
    mesh = jax.make_mesh(*MESH)
    ref_model = load_model(
      model_name, mesh=mesh, checkpoint_path=model_path, dtype=jnp.bfloat16, random_seed=42
    )
    # Policy model
    policy_model = nnx.clone(ref_model)
  with timer.section("tokenizer"):
    tokenizer = Tokenizer(
      tokenizer_type='huggingface',
      tokenizer_path=model_path,
    )

#----------------------------- Dataset Preparation ----------------------------#

# Instruct the model to reason using special tokens

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""

# GSM8K dataset of grade school math word problems.

def get_dataset(data_dir, split="train") -> grain.MapDataset:
  # Prefer using existing dataset without downloading; fall back to download if missing
  try:
    data = tfds.data_source(
      "gsm8k",
      split=split,
      data_dir=data_dir,
      download=False,
    )
  except Exception:
    data = tfds.data_source(
      "gsm8k",
      split=split,
      data_dir=data_dir,
      builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
      download=True,
    )

  def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
      return None
    return text.split("####")[1].strip()

  dataset = (
    grain.MapDataset.source(data)
    .shuffle(seed=42)
    .map(
      lambda x: {
        # for forward pass
        "prompts": tokenizer.apply_chat_template(
          [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"].decode("utf-8")},
          ],
          add_generation_prompt=True,
          tokenize=False,
        ),
        # for reward functions
        "question": x["question"].decode("utf-8"),
        # for reward functions
        "answer": extract_hash_answer(x["answer"].decode("utf-8")),
      }
    )
  )
  return dataset

with timer.section("load_dataset"):
  ds = get_dataset(DATASET_DIR, "train").batch(TRAIN_MICRO_BATCH_SIZE)
  dataset = ds[:NUM_BATCHES]
  train_dataset = dataset.repeat(NUM_EPOCHS)

#--------------------------- Define Reward Functions --------------------------#

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)

def reward_final_answer(prompts, completions, answer, **kwargs):
  extracted = [
      m.group(1).strip() if (m := match_numbers.search(r)) is not None else None
      for r in completions
  ]
  scores = []
  for guess, true in zip(extracted, answer):
    try:
      scores.append(1.5 if float(guess) == float(true.strip()) else 0.0)
    except:
      scores.append(0.0)
  return scores 

#-------------------------- Training Setup & Execution ------------------------#

# Create Optimizer
optimizer = optax.adamw(
    learning_rate=LEARNING_RATE, b1=B1, b2=B2, weight_decay=WEIGHT_DECAY,
)

# choose rollout engine
if ROLLOUT_ENGINE == 'vllm_gpu':
  rollout_engine = partial(
    VllmGPURollout,
    os.environ.get("GATEWAY_URL"),
    model_name,
    transfer_mode=TRANSFER_MODE,
    timer=timer,
  )
elif ROLLOUT_ENGINE == 'vanilla':
  rollout_engine = 'vanilla'

# Create the RL cluster
rl_cluster = RLCluster(
  actor=policy_model,
  reference=ref_model,
  tokenizer=tokenizer,
  cluster_config=ClusterConfig(
    role_to_mesh={
      ClusterRole.ACTOR: mesh,
      ClusterRole.REFERENCE: mesh,
      ClusterRole.ROLLOUT: mesh,
    },
    rollout_engine=rollout_engine,
    offload_to_cpu=False,
    training_config=RLTrainingConfig(
      actor_optimizer=optimizer,
      eval_every_n_steps=EVAL_EVERY_N_STEPS,
      max_steps=MAX_STEPS,
      mini_batch_size=TRAIN_MICRO_BATCH_SIZE,
      train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
    ),
    rollout_config=base_rollout.RolloutConfig(
      max_tokens_to_generate=TOTAL_GENERATION_STEPS,
      max_prompt_length=MAX_PROMPT_LENGTH,
      kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
      temperature=TEMPERATURE,
      top_p=TOP_P,
      top_k=TOP_K,
    ),
  ),
)
rl_cluster.sync_weights()

#------------------------------ Training with GRPO ---------------------------#

grpo_trainer = GRPOLearner(
  rl_cluster=rl_cluster,
  reward_fns=[reward_final_answer],
  grpo_config=GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    beta=BETA,
    epsilon=EPSILON,
  ),
)

# Start training
with timer.section("training"):
  with mesh:
    grpo_trainer.train(train_dataset)

#------------------------------ Report & Finalize ----------------------------#

timer.summary(sort_by='name', precision=1)

# Gracefully shutdown the gateway and vLLM (process 0 only)
if ROLLOUT_ENGINE == 'vllm_gpu' and jax.process_index() == 0:
  try:
    rl_cluster.rollout.shutdown()
  except Exception as e:
    logger.warning(f"Failed to shutdown gateway: {e}")
