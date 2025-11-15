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
import functools
import os
from importlib import import_module

from packaging.version import Version

from vllm import LLM as UpstreamLLM
from vllm import __version__ as vllm_version

from .extension import VLLMWorkerExtension

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


class LLM(UpstreamLLM):
  """A thin wrapper around vllm.LLM that injects a custom worker extension for
  handling JAX-vLLM weight transfer and rollout requests.
  """

  @functools.wraps(UpstreamLLM.__init__)
  def __init__(self, *args, **kwargs):
    # Inject our custom worker extension.
    # If the user also specifies a custom extension, we create a new
    # class that inherits from both ours and the user's.
    if "worker_extension_cls" not in kwargs:
      WorkerExtension = VLLMWorkerExtension
    else:
      try:
        path, _, name = kwargs["worker_extension_cls"].rpartition('.')
        user_extension_cls = getattr(import_module(path), name)
      except AttributeError | ModuleNotFoundError:
        raise ValueError(f"Cannot find worker extension class {kwargs['worker_extension_cls']}")
      class WorkerExtension(VLLMWorkerExtension, user_extension_cls):
        pass
    kwargs["worker_extension_cls"] = ".".join(
      [WorkerExtension.__module__, WorkerExtension.__name__]
    )
    super().__init__(*args, **kwargs)

    # The dummy weights loader incorrectly initializes the normalizer with a random
    # value, so we reset it here. Can be removed when
    # https://github.com/vllm-project/vllm/pull/19788 is landed.
    if 'gemma' in self.llm_engine.model_config.model and Version(vllm_version) <= Version("0.9.1"):
      self.collective_rpc(
        lambda self: repr(
          self.worker.model_runner.model.model.normalizer.copy_(
            torch.tensor(
              self.worker.model_runner.model.model.config.hidden_size**0.5,
              device=self.worker.model_runner.model.model.normalizer.device,
              dtype=self.worker.model_runner.model.model.normalizer.dtype,
            )
          )
        )
      )
