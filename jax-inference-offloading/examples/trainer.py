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
"""JAX-vLLM coupling example (JAX side)."""

import logging
import os
import socket
import time

import cupy.cuda.runtime as cudart
import jax
import jax.numpy as jnp

import jax_inference_offloading.api.controller_pb2 as ctrl
from jax_inference_offloading.jax import OffloadingBridge
from jax_inference_offloading.sharding import PolymorphicMesh
from jax_inference_offloading.timer import Timer
from jax_inference_offloading.tunix.load_model import load_model
from jax_inference_offloading.models import get_named_parameters

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
timer = Timer()

# Optional multi-process initialization (one process per node)
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
    local_ids = [
      int(x) for x in os.environ.get("JAX_LOCAL_DEVICE_IDS", "").split(",") if x
    ] or None
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

# for JAX before 286ad054d89 (Sep 29, 4:05 PM PDT 2025),
# buffer_callback requires input tensors to be sharded on the context mesh
# our polymorphic mesh is implemented to allow input tensors to be sharded on a different mesh
# from the context mesh, due to the difference between JAX and vLLM parallelism
USE_POLYMORPHIC_MESH = os.environ.get('USE_POLYMORPHIC_MESH', '0') == '1'

if USE_POLYMORPHIC_MESH:
  main_mesh = PolymorphicMesh(
    devices=jax.devices(),
    primary_shape=(1, len(jax.devices())),
    extra_singleton_dims=0,
  )
  training_mesh = main_mesh.view((1, len(jax.devices())), ("fsdp", "tp"))
else:
  # main_mesh = jax.make_mesh((1, len(jax.devices())), ("fsdp", "tp"))
  main_mesh = jax.make_mesh((jax.process_count(), jax.local_device_count()), ("fsdp", "tp"))
  training_mesh = main_mesh

# Load model and tokenizer
model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
model_path = os.environ.get('MODEL_PATH', None)

if jax.process_index() == 0:
  print(f"Loading JAX model {model_name} @ {model_path}")

with timer.section("load_model"):
  t0 = time.time()
  model = load_model(
    model_name,
    training_mesh,
    checkpoint_path=model_path,
    dtype=jnp.bfloat16,
    random_seed=42
  )
  t1 = time.time()
  print(f'Model loaded in {t1 - t0:.2f} seconds')

with timer.section("create_bridge"):
  gateway_url = os.environ.get("GATEWAY_URL")
  transfer_mode = os.environ.get('TRANSFER_MODE', 'grouped')  # 'fused' or 'unfused'
  bridge = OffloadingBridge(
    gateway_url=gateway_url,
    model_name=model_name,
    mesh=main_mesh,
    transfer_mode=transfer_mode,
    timer=timer,
  )

# send model to vLLM
# the first run includes compilation overhead
state = get_named_parameters(model)
with timer.section("warmup"):
  bridge.transfer(state)
# send model a second time to benchmark
for r in range(5):
  with timer.section(f"transport.run{r}"):
    cudart.profilerStart()
    bridge.transfer(state)
    cudart.profilerStop()

rollout_config = ctrl.RolloutConfig()
rollout_config.max_tokens = 500

# string prompt
if jax.process_index() == 0:
  response = bridge.gateway.inference(prompts="Quick facts about the moon:", config=rollout_config)
  print("=" * 80)
  print(response.outputs[0].generated_text)
  print("=" * 80)

# chat prompt
if jax.process_index() == 0:
  response = bridge.gateway.inference(
    prompts=[
      {"role": "system", "content": "Always start the reply with Ho-ho-ho!"},
      {"role": "user", "content": "Quick facts about the moon"},
    ],
    config=rollout_config,
  )
  print("=" * 80)
  print(response.outputs[0].generated_text)
  print("=" * 80)

if jax.process_index() == 0:
  timer.summary(sort_by='name', precision=3)
  for metric_name, metric_key in [
    ('JIO_METRIC_TRANSFER', r'transport\.run\d+$'),
    ('JIO_METRIC_HANDSHAKE', r'create_bridge\.handshake$'),
    ('JIO_METRIC_LOADMODEL', r'load_model$'),
  ]:
    print(timer.ci_metric(
      metric_name,
      timer.node_stat(metric_key, ('mean', 'std', 'min', 'max')),
      unit='s'
    ))

if jax.process_index() == 0:
  bridge.gateway.shutdown()

if jax.process_index() == 0:
  print('JAX client exiting')
