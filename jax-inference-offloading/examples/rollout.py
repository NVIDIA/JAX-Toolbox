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
"""Multi-process JAX-vLLM coupling example (vLLM side).
Assumes:
- JAX and vLLM are running in different processes on the same physical node.
- JAX and vLLM occupy different GPUs.
"""

import logging
import os


def main():
  from jax_inference_offloading.controller.rollout_client import make_rollout_client
  from jax_inference_offloading.vllm import LLM

  os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

  # parse environment variables
  enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1"
  gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
  cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
  tensor_parallel_size = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", len(cuda_visible_devices)))
  distributed_backend = os.environ.get("VLLM_DISTRIBUTED_BACKEND", "mp")
  load_format = os.environ.get("VLLM_LOAD_FORMAT", "dummy")
  model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
  model_path = os.environ.get("MODEL_PATH", None)
  model = model_path or model_name

  logging.basicConfig(level=logging.INFO)

  gateway_url = os.environ.get("GATEWAY_URL")

  # create offline inference server
  llm = LLM(
    model=model,
    enforce_eager=enforce_eager,
    tensor_parallel_size=tensor_parallel_size,
    distributed_executor_backend=distributed_backend,
    load_format=load_format,
    gpu_memory_utilization=gpu_memory_utilization,
    max_model_len=1024,
  )

  # subscribe to control messages from the gateway
  rollout_client = make_rollout_client(gateway_url)
  rollout_client.subscribe_to_control_messages(llm)


if __name__ == "__main__":
  import torch.multiprocessing as mp
  try:
    mp.set_start_method("spawn", force=True)
  except RuntimeError:
    pass
  main()
