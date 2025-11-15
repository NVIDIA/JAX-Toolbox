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
from .nccl_star import NcclStarTransport

_TRANSPORT_CLS = {}
_TRANSPORT_CLS["NCCL"] = NcclStarTransport

def get_transport_class(name):
  try:
    return _TRANSPORT_CLS[name]
  except KeyError:
    raise ValueError(
      f"Unknown transport protocol: {name}. "
      f"Available options are: {list(_TRANSPORT_CLS.keys())}"
    ) from KeyError
