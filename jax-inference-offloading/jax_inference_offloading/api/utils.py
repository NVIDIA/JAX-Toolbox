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
from dataclasses import make_dataclass
from typing import Generic, Protocol, TypeVar

from google.protobuf import json_format

ProtoT = TypeVar("ProtoT")

# for use as a type hint
class DataclassFor(Protocol, Generic[ProtoT]):
    ...


def proto_to_dataclass(proto, cls_name):
  def _as_datacls(j, cls_name):
    if isinstance(j, dict):
      cls = make_dataclass(cls_name, j.keys(), frozen=True)
      values = [_as_datacls(v, k) for k, v in j.items()]
      return cls(*values)
    elif isinstance(j, list):
      return tuple(map(lambda item: _as_datacls(item, cls_name), j))
    else:
      return j

  return _as_datacls(
    json_format.MessageToDict(
      proto,
      always_print_fields_with_no_presence=True,
      preserving_proto_field_name=True,
    ),
    cls_name
  )
