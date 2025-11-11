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
"""Multi-process JAX-vLLM coupling example (JAX side).
Assumes:
- JAX and vLLM are running in different processes on the same physical node.
- JAX and vLLM occupy different GPUs.
"""

import logging
import os
import re
import socket
import tempfile
from functools import partial
from pprint import pprint

import grain
import jax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds
from flax import nnx
from orbax import checkpoint as ocp
from tqdm.auto import tqdm
from tunix.generate import sampler as sampler_lib
from tunix.generate.tokenizer_adapter import Tokenizer
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rl_cluster import ClusterConfig, RLCluster, RLTrainingConfig
from tunix.rl.rl_cluster import Role as ClusterRole
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger

from jax_inference_offloading.timer import Timer
from jax_inference_offloading.tunix.load_model import load_model
from jax_inference_offloading.tunix.rollout import VllmGPURollout

logger = logging.getLogger(__name__)
timer = Timer()

ROLLOUT_ENGINE = os.environ.get('ROLLOUT_ENGINE', 'vllm_gpu')  # or 'vanilla'
RUN_MODE = os.environ.get('RUN_MODE', 'timing')  # or 'production'
TRANSFER_MODE = os.environ.get('TRANSFER_MODE', 'grouped')  # 'grouped' | 'fused' | 'unfused'

with timer.section("jax-distributed-initialize"):
  num_processes = int(os.environ.get("JAX_NUM_PROCESSES", "1"))
  if num_processes > 1:
    if "JAX_PROCESS_INDEX" in os.environ:
      process_id = int(os.environ["JAX_PROCESS_INDEX"])
    else:
      hostlist = [h for h in os.environ.get("JAX_NODELIST", "").split(",") if h]
      me = socket.gethostname().split(".")[0]
      if me not in hostlist:
        raise RuntimeError(f"Hostname {me} not found in JAX_NODELIST: {hostlist}")
      process_id = hostlist.index(me)
    coordinator_addr = os.environ.get("JAX_COORDINATOR_ADDR")
    coordinator_port = int(os.environ.get("JAX_COORDINATOR_PORT", "12345"))
    local_ids = [int(x) for x in os.environ.get("JAX_LOCAL_DEVICE_IDS", "").split(",") if x] or None
    print(f'Initializing JAX distributed: process_id={process_id}, '
          f'num_processes={num_processes}, '
          f'coordinator_addr={coordinator_addr}, '
          f'coordinator_port={coordinator_port}, '
          f'local_ids={local_ids}')
    jax.distributed.initialize(
      f"{coordinator_addr}:{coordinator_port}",
      num_processes=num_processes,
      process_id=process_id,
      local_device_ids=local_ids,
    )
  if jax.process_index() == 0:
    for d in jax.devices():
      print(f"JAX device {d.id}: {d.device_kind}")

###############################################################################
# ## Hyperparameters
#
# Let's define the configuration we are going to use. Note that this is by no
# means a "perfect" set of hyperparameters. To get good results, you might have
# to train the model for longer.
###############################################################################

# Helper for hyperparameter overrides via environment variables
def get_env(name, default):
  raw = os.environ.pop(name, None)
  if raw is None:
    return default
  if isinstance(default, bool):
    return str(raw).lower() in {"1", "true", "yes", "y"}
  if isinstance(default, int):
    return int(raw)
  if isinstance(default, float):
    return float(raw)
  return str(raw)

# ====== Data ======
SCRATCHDIR = os.environ.get("SCRATCHDIR", "/content")
DATASET_DIR = os.environ.get("DATASET_DIR")
if not DATASET_DIR:
  TEMPDIR = tempfile.mkdtemp()
  DATASET_DIR = os.path.join(TEMPDIR, "tfds")
TRAIN_FRACTION = 1.0

# ====== Sharding ======
MESH = [(jax.process_count(), jax.local_device_count()), ("fsdp", "tp")]

# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = get_env("GRPO_MAX_PROMPT_LENGTH", 256)
TOTAL_GENERATION_STEPS = get_env("GRPO_TOTAL_GENERATION_STEPS", 512)
# Important to keep a high-ish temperature for varied, diverse responses during
# training.
TEMPERATURE = get_env("GRPO_TEMPERATURE", 0.9)
TOP_P = get_env("GRPO_TOP_P", 0.95)
TOP_K = get_env("GRPO_TOP_K", 50)

# === other GRPO configs ===
# number of inner-loop policy update steps per batch of data
# (𝜇 in GRPO algo 1).
NUM_ITERATIONS = get_env("GRPO_NUM_ITERATIONS", 1)
# The coefficient for the KL divergence penalty (𝛽) in the GRPO loss function.
# Important to keep a high enough value for this, otherwise, the KL divergence
# can increase unchecked.
BETA = get_env("GRPO_BETA", 0.03)
# Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to PPO, for
# stable updates.
EPSILON = get_env("GRPO_EPSILON", 0.15)

# ====== Training ======
# Use 'timing' defaults regardless of RUN_MODE (overridable via GRPO_*)
TRAIN_MICRO_BATCH_SIZE = get_env("GRPO_TRAIN_MICRO_BATCH_SIZE", 16)
NUM_WARMUP_BATCHES = get_env("GRPO_NUM_WARMUP_BATCHES", 1)
NUM_BATCHES = get_env("GRPO_NUM_BATCHES", 5)
# Number of rollouts per prompt to be generated by the policy model during
# a single training step. This corresponds to `G` in Algorithm 1 in the
# paper. The "group" in GRPO comes from here.
NUM_GENERATIONS = get_env("GRPO_NUM_GENERATIONS", 16)
# Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly. It can be
# increased to a max. of 330 (if batch size is 4).
NUM_TEST_BATCHES = get_env("GRPO_NUM_TEST_BATCHES", 1)

EVAL_EVERY_N_STEPS = get_env("GRPO_EVAL_EVERY_N_STEPS", 10)  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
NUM_EPOCHS = get_env("GRPO_NUM_EPOCHS", 1)  # can potentially train for more epochs

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = get_env("GRPO_LEARNING_RATE", 5e-6)
B1 = get_env("GRPO_B1", 0.9)
B2 = get_env("GRPO_B2", 0.95)
WEIGHT_DECAY = get_env("GRPO_WEIGHT_DECAY", 0.02)
# == Cosine decay with warmup scheduler ==
# Linearly increase learning rate from 0. to 5e-6 in the first 10% training
# steps, and then gradually decrease the learning rate to 0 using cosine
# scheduler.
WARMUP_STEPS = get_env("GRPO_WARMUP_STEPS", int(0.06 * MAX_STEPS))
# == Grad clipping ==
# Grad clipping to prevent large gradients. Found this
# important to keep KL divergence in check.
MAX_GRAD_NORM = get_env("GRPO_MAX_GRAD_NORM", 1.0)

# Checkpoint saving
INTERMEDIATE_CKPT_DIR = get_env("GRPO_INTERMEDIATE_CKPT_DIR", os.path.join(SCRATCHDIR, "intermediate_ckpt"))
CKPT_DIR = get_env("GRPO_CKPT_DIR", os.path.join(SCRATCHDIR, "ckpts"))
SAVE_INTERVAL_STEPS = get_env("GRPO_SAVE_INTERVAL_STEPS", 100)
MAX_TO_KEEP = get_env("GRPO_MAX_TO_KEEP", 10)

# Fail fast on unknown GRPO_* env vars (helps catch typos)
_unknown_grpo = [k for k in os.environ.keys() if k.startswith("GRPO_")]
if _unknown_grpo:
  raise ValueError(f"Unrecognized GRPO_* env vars: {', '.join(sorted(_unknown_grpo))}")

# Inference
GENERATION_CONFIGS = {
  # greedy search
  "greedy": {"temperature": 1e-4, "top_k": 1, "top_p": 1.0},
  # some randomness
  "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
  # liberal
  "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}

###############################################################################
# ## Load the reference/policy model and tokenizer
#
# The policy model is the model which is actually trained and whose weights are
# updated. The reference model is the model with which we compute KL divergence.
# This is to ensure that the policy updates are not huge and that it does not
# deviate too much from the reference model.
#
# We load a LLaMA-3 model and its tokenizer from SafeTensors format.
###############################################################################

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

###############################################################################
# ## Dataset preparation
#
# We instruct the model to first reason between the `<reasoning>` and
# `</reasoning>` tokens. After reasoning, we expect it to provide the answer
# between the `<answer>` and `</answer>` tokens.
###############################################################################

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""

##############################################################################
# We use OpenAI's [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k),
# which comprises grade school math word problems.
##############################################################################

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
        # passed to model forward pass
        "prompts": tokenizer.apply_chat_template(
          [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"].decode("utf-8")},
          ],
          add_generation_prompt=True,
          tokenize=False,
        ),
        # passed to reward functions
        "question": x["question"].decode("utf-8"),
        # passed to reward functions
        "answer": extract_hash_answer(x["answer"].decode("utf-8")),
      }
    )
  )
  return dataset

with timer.section("load_dataset"):
  with timer.section("train"):
    ds = get_dataset(DATASET_DIR, "train").batch(TRAIN_MICRO_BATCH_SIZE)
    warmup_dataset = ds[:NUM_WARMUP_BATCHES]
    dataset = ds[NUM_WARMUP_BATCHES:NUM_WARMUP_BATCHES + NUM_BATCHES]
    if TRAIN_FRACTION == 1.0:
      train_dataset = dataset.repeat(NUM_EPOCHS)
      val_dataset = None
    else:
      train_dataset = dataset[: int(len(dataset) * TRAIN_FRACTION)]
      train_dataset = train_dataset.repeat(NUM_EPOCHS)
      val_dataset = dataset[int(len(dataset) * TRAIN_FRACTION) :].repeat(NUM_EPOCHS)
  with timer.section("test"):
    test_dataset = get_dataset(DATASET_DIR, "test").batch(TRAIN_MICRO_BATCH_SIZE)[:NUM_TEST_BATCHES]

dataset_lengths = {
  'warmup': len(warmup_dataset),
  'train': len(train_dataset),
  'val': len(val_dataset) if val_dataset is not None else 0,
  'test': len(test_dataset),
}
print(
  f"dataset contains "
  f"{dataset_lengths['warmup']} warmup batches, "
  f"{dataset_lengths['train']} training batches, "
  f"{dataset_lengths['val']} validation batches, "
  f"{dataset_lengths['test']} test batches."
)
print("Sample batch from dataset:")
for ele in dataset[:1]:
  pprint(ele)

###############################################################################
# ## Define reward functions
# We define four reward functions:
# - reward if the format of the output exactly matches the instruction given in
# `SYSTEM_PROMPT`;
# - reward if the format of the output approximately matches the instruction given
# in `SYSTEM_PROMPT`;
# - reward if the answer is correct/partially correct;
# - Sometimes, the text between `<answer>`, `</answer>` might not be one
#   number. So, we extract the number, and reward the model if the answer is correct.
# The reward functions are inspired from
# [here](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb).
# First off, let's define a RegEx for checking whether the format matches.
###############################################################################

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

match_format.search(
    f"{reasoning_start}Let me"
    f" think!{reasoning_end}{solution_start}2{solution_end}",
)

# Give the model a reward of 3 points if the format matches exactly.
def match_format_exactly(prompts, completions, **kwargs):
  return [
      0 if match_format.search(response) is None else 3.0
      for response in completions
  ]

# Also reward the model if the format of the output matches partially.
def match_format_approximately(prompts, completions, **kwargs):
  scores = []

  for completion in completions:
    score = 0
    response = completion
    # Count how many keywords are seen - we penalize if too many!
    # If we see 1, then plus some points!
    score += 0.5 if response.count(reasoning_start) == 1 else -0.5
    score += 0.5 if response.count(reasoning_end) == 1 else -0.5
    score += 0.5 if response.count(solution_start) == 1 else -0.5
    score += 0.5 if response.count(solution_end) == 1 else -0.5
    scores.append(score)
  return scores

# Reward the model if the answer is correct. A reward is also given if the answer
# does not match exactly, i.e., based on how close the answer is to the correct
# value.
def check_answer(prompts, completions, answer, **kwargs):
  responses = completions

  extracted_responses = [
      guess.group(1) if (guess := match_format.search(r)) is not None else None
      for r in responses
  ]

  scores = []
  assert len(extracted_responses) == len(
      answer
  ), f"{extracted_responses} and {answer} have mismatching length"
  for guess, true_answer in zip(extracted_responses, answer):
    score = 0
    if guess is None:
      scores.append(0)
      continue
    # Correct answer gets 3 points!
    if guess == true_answer:
      score += 3.0
    # Match if spaces are seen
    elif guess.strip() == true_answer.strip():
      score += 1.5
    else:
      # We also reward it if the answer is close via ratios!
      # Ie if the answer is within some range, reward it!
      try:
        ratio = float(guess) / float(true_answer)
        if ratio >= 0.9 and ratio <= 1.1:
          score += 0.5
        elif ratio >= 0.8 and ratio <= 1.2:
          score += 0.25
        else:
          score -= 1.0  # Penalize wrong answers
      except:
        score -= 0.5  # Penalize
    scores.append(score)
  return scores

# Sometimes, the text between `<answer>` and `</answer>` might not be one
# number; it can be a sentence. So, we extract the number and compare the answer.
match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)
match_numbers.findall(f"{solution_start}  0.34  {solution_end}")

def check_numbers(prompts, completions, answer, **kwargs):
  question = kwargs["question"]
  responses = completions

  extracted_responses = [
      guess.group(1) if (guess := match_numbers.search(r)) is not None else None
      for r in responses
  ]

  scores = []
  print("START ============================")
  print(f"Question: {question[0]}")
  print(f"Answer: {answer[0]}")
  print(f"Response: {responses[0]}")
  print(f"Extracted: {extracted_responses[0]}")
  print("END ==============================")
  for guess, true_answer in zip(extracted_responses, answer):
    if guess is None:
      scores.append(0)
      continue
    # Convert to numbers
    try:
      true_answer = float(true_answer.strip())
      guess = float(guess.strip())
      scores.append(1.5 if guess == true_answer else 0.0)
    except:
      scores.append(0)
      continue
  return scores


###############################################################################
# ## Evaluate
#
# Before we train the model, let's evaluate the model on the test set so we can
# see the improvement post training.
#
# We evaluate it in two ways:
#
# **Quantitative**
#
# * **Answer Accuracy**: percentage of samples for which the model predicts the
# correct final numerical answer
# * **Answer (Partial) Accuracy**: percentage of samples for which the model
# predicts a final numerical answer such that the \`model answer / answer\`
# ratio lies between 0.9 and 1.1.
# * **Format Accuracy**: percentage of samples for which the model outputs the
# correct format, i.e., reasoning between the reasoning special tokens, and the
# final answer between the \`\<start\_answer\>\`, \`\<end\_answer\>\` tokens.
#
# **Qualitative**
#
# We'll also print outputs for a few given questions so that we can compare the
# generated output later.
###############################################################################

def generate(
    question, sampler, temperature=0.7, top_k=50, top_p=0.95, seed=None
):
  """Given prompt, generates text."""
  if isinstance(question, str):
    question = [question]

  input_batch = [
    tokenizer.apply_chat_template(
      [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
      ],
      add_generation_prompt=True,
      tokenize=False,
    )
    for q in question
  ]

  out_data = sampler(
      input_strings=input_batch,
      max_generation_steps=768,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
      echo=False,
      seed=seed if seed is not None else None,
  )

  output = out_data.text
  if isinstance(question, str):
    return output[0]
  return output

def evaluate(
    dataset,
    sampler,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_passes=1,
    corr_lst=False,
    make_lst=False,
):
  """Computes accuracy and percentage of outputs matching the format."""
  response_lst = []
  corr = 0
  partially_corr = 0
  corr_format = 0
  total = 0

  for batch in tqdm(dataset):
    answers = batch["answer"]
    questions = batch["question"]

    multiple_call_responses = [[] for _ in range(len(questions))]
    for p in range(num_passes):
      responses = generate(
          questions, sampler, temperature, top_k, top_p, seed=p
      )
      for idx, response in enumerate(responses):
        multiple_call_responses[idx].append(response)

    for question, multiple_call_response, answer in zip(
        questions, multiple_call_responses, answers
    ):
      # check answer
      corr_ctr_per_question = 0
      partially_corr_per_question = 0
      corr_format_per_question = 0
      for response in multiple_call_response:
        extracted_response = (
            guess.group(1)
            if (guess := match_numbers.search(response)) is not None
            else "-1000000"
        )
        try:
          if float(extracted_response.strip()) == float(answer.strip()):
            corr_ctr_per_question += 1

          ratio = float(extracted_response.strip()) / float(answer.strip())
          if ratio >= 0.9 and ratio <= 1.1:
            partially_corr_per_question += 1
        except:
          print("SKIPPED")

        # check format
        if match_format.search(response) is not None:
          corr_format_per_question += 1

        if (
            corr_ctr_per_question > 0
            and partially_corr_per_question > 0
            and corr_format_per_question > 0
        ):
          break

      if corr_ctr_per_question > 0:
        corr += 1
        if corr_lst and make_lst:
          response_lst.append((question, answer, multiple_call_response))
      else:
        if not corr_lst and make_lst:
          response_lst.append((question, answer, multiple_call_response))
      if partially_corr_per_question > 0:
        partially_corr += 1
      if corr_format_per_question > 0:
        corr_format += 1

      total += 1
      if total % 10 == 0:
        print(
            f"===> {corr=}, {total=}, {corr / total * 100=}, "
            f"{partially_corr / total * 100=}, {corr_format / total * 100=}"
        )

  to_return = (
      corr,
      total,
      corr / total * 100,
      partially_corr / total * 100,
      corr_format / total * 100,
  )
  if make_lst:
    return to_return, response_lst
  return to_return

sampler = sampler_lib.Sampler(
    transformer=ref_model,
    tokenizer=tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        num_layers=ref_model.config.num_layers,
        num_kv_heads=ref_model.config.num_kv_heads,
        head_dim=ref_model.config.head_dim,
    ),
)

###############################################################################
# Now let's see how the original model does on the test set.
# You can see the percentages of the mode outputs that are fully correct,
# partially correct and just correct in format. The following step might
# take couple of minutes to finish.
###############################################################################

if RUN_MODE == 'production':

  # using Tunix built-in sampler
  with timer.section("initial_evaluation"):
    (corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
        test_dataset,
        sampler,
        **GENERATION_CONFIGS["greedy"],
    )
    print(
        f"{corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
        f" {format_accuracy=}%"
    )

###############################################################################
# ## Train

# Let's set up all the configs first - checkpointing, metric logging and training.
# We then train the model.
###############################################################################

# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(
  save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
  log_dir=os.path.join(SCRATCHDIR, "tensorboard/grpo"),
  flush_every_n_steps=20
)

# Create Optimizer with learning rate scheduler and (optionally) gradient clipping
optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=MAX_STEPS,
        end_value=0.0,
    ),
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
)
if MAX_GRAD_NORM is not None:
  optimizer = optax.chain(
      optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
      optimizer,
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
cluster_config = ClusterConfig(
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
        # metrics logging
        metrics_logging_options=metrics_logging_options,
        # checkpoint saving
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=TOTAL_GENERATION_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    ),
)

# RL cluster
rl_cluster = RLCluster(
    actor=policy_model,
    reference=ref_model,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)
rl_cluster.sync_weights()

###########################################################################
### Setting Up the GRPO Trainer
#
# Now we initialize our system for training. First, we create an `RLCluster`
# instance, which brings together the **policy model (`actor`)**, a
# **reference model (`reference`)**, and a **tokenizer**. Our `actor` is a
# trainable LoRA model, while the `reference` is a fixed base model that we
# use to guide the training.
#
# We then create a `GRPOLearner`, the specialized trainer that uses a list
# of **reward functions** to evaluate and optimize the model's output,
# completing the RL training setup.
#
# Tunix trainers are integrated with [Weights & Biases](https://wandb.ai/)
# to help you visualize the training progress. You can choose how you want
# to use it:
#
# **Option 1 (Type 1)**: If you're running a quick experiment or just
# testing things out, choose this. It creates a temporary, private dashboard
# right in your browser without requiring you to log in or create an account.
#
# **Option 2 (Type 2)**: If you have an existing W&B account and want to
# save your project's history to your personal dashboard, choose this.
# You'll be prompted to enter your API key or log in.
###########################################################################

grpo_config = GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    beta=BETA,
    epsilon=EPSILON,
)

# Warmup
if len(warmup_dataset) > 0:
  rl_cluster.rollout.generate(
    next(iter(warmup_dataset))["prompts"],
    cluster_config.rollout_config,
  )

# GRPO Trainer
grpo_trainer = GRPOLearner(
  rl_cluster=rl_cluster,
  reward_fns=[
    match_format_exactly,
    match_format_approximately,
    check_answer,
    check_numbers,
  ],
  grpo_config=grpo_config,
)

# Start training
# The first couple of training step might take up to 5 minutes to finish.
with timer.section("training"):
  with mesh:
    grpo_trainer.train(train_dataset)

###########################################################################
## Evaluate
# Let's evaluate our finetuned model!
###########################################################################

if RUN_MODE == 'production':

  # Load checkpoint first.
  trained_ckpt_path = os.path.join(
      CKPT_DIR, "actor", str(MAX_STEPS), "model_params"
  )

  abs_params = jax.tree.map(
      lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
      nnx.state(policy_model, nnx.Param),
  )
  checkpointer = ocp.StandardCheckpointer()
  trained_params = checkpointer.restore(trained_ckpt_path, target=abs_params)

  nnx.update(
      policy_model,
      jax.tree.map(
          lambda a, b: b,
          nnx.state(policy_model, nnx.Param),
          trained_params,
      ),
  )

  sampler = sampler_lib.Sampler(
      transformer=policy_model,
      tokenizer=tokenizer,
      cache_config=sampler_lib.CacheConfig(
          cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
          num_layers=ref_model.config.num_layers,
          num_kv_heads=ref_model.config.num_kv_heads,
          head_dim=ref_model.config.head_dim,
      ),
  )

  with timer.section("final_evaluation"):
    # The evaluation might take up to couple of minutes to finish.
    (corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
        test_dataset,
        sampler,
        **GENERATION_CONFIGS["greedy"],
    )
    print(
        f"{corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
        f" {format_accuracy=}%"
    )

###########################################################################
## Report timing
###########################################################################

timer.summary(sort_by='name', precision=1)

# Gracefully shutdown the vLLM gateway (process 0 only)
if ROLLOUT_ENGINE == 'vllm_gpu' and jax.process_index() == 0:
  try:
    rl_cluster.rollout.shutdown()
  except Exception as e:
    logger.warning(f"Failed to shutdown gateway: {e}")
