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
import re
import time
import numpy as np
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Callable


class Timer:
  """Hierarchical wall‑time profiler.

  `sort_by='name'`  – default, sections ordered by total time (desc).  
  `sort_by='time'`  – alphabetical, shown as an indented tree.

  Example:
  -------
  >>> t = Timer()
  >>> with t.section("load"):
  ...     heavy_io()
  ...     with t.section("parse.json"):
  ...         parse_json()
  >>> with t.section("train"):
  ...     train()
  >>> t.summary(sort_by='name')
  load               : 0.5012 s
    parse            : 0.3017 s
      json           : 0.3017 s
  train              : 2.8344 s
  """

  def __init__(self) -> None:
    self._times: dict[str, float] = defaultdict(float)
    self._stack: deque[str] = deque()

  @contextmanager
  def section(self, name: str):
    """Context‑manager that records wall time spent in *name*.
    Supports nesting; inner sections are recorded under
    parent.child.grandchild... paths.
    """
    names = name.split('.')
    self._stack.extend(names)
    start = time.perf_counter()
    try:
      yield
    finally:
      elapsed = time.perf_counter() - start
      for _ in names:
        self._times[".".join(self._stack)] += elapsed
        self._stack.pop()

  def summary(self, *, sort_by: str = "name", precision: int = 2, col_sep: str = ' : ') -> None:
    """Print the timing summary.

    Parameters
    ----------
    sort_by : {'time', 'name'}
        'time': sort by total elapsed seconds (descending).
        'name': sort alphabetically and render a tree.
    precision : int
        Number of decimal places for seconds.

    """
    if not self._times:
      print("Timer summary: no sections recorded.")
      return

    if sort_by not in ("time", "name"):
      raise ValueError("sort_by must be 'time' or 'name'.")

    if sort_by == "time":
      self._print_by_time(precision, col_sep)
    else:
      self._print_by_name(precision, col_sep)

  def _print_by_time(self, precision: int, col_sep: str) -> None:
    col_width_name = max(len(name) for name in self._times)
    col_width_time = len(f'{max(t for t in self._times.values()):.{precision}f}')
    col_width_sep = len(col_sep)
    total_width = col_width_name + col_width_sep + col_width_time
    items = sorted(self._times.items(), key=lambda kv: (kv[1], -len(kv[0])), reverse=True)
    fmt = f"{{:<{col_width_name}}}{col_sep}{{:{col_width_time}.{precision}f}}"
    print("Timer summary (sorted by time), seconds")
    print("-" * total_width)
    for name, t in items:
      print(fmt.format(name, t))

  def _print_by_name(self, precision: int, col_sep: str) -> None:
    indent = ' ' * 2
    col_width_name = max((len(name.split('.')) - 1) * len(indent) + len(name.split('.')[-1]) for name in self._times)
    col_width_time = len(f'{max(t for t in self._times.values()):.{precision}f}')
    col_width_sep = len(col_sep)
    total_width = col_width_name + col_width_sep + col_width_time
    print("Timer summary (tree view), seconds")
    print("-" * total_width)
    for full_name, t in sorted(self._times.items(), key=lambda kv: kv[0]):
      depth = full_name.count(".")
      segment = full_name.split(".")[-1]
      indent = "  " * depth
      print(f"{indent + segment:<{col_width_name}}{col_sep}{t:{col_width_time}.{precision}f}")

  def node_stat(
      self,
      pattern: str,
      stats: str | Callable[[list[float]], float] | list[str | Callable[[list[float]], float]] = 'mean'
  ) -> float | tuple[float, ...] | None:
    matcher = re.compile(pattern)
    vals = [t for name, t in self._times.items() if matcher.fullmatch(name)]
    if not vals:
      return None

    def get_op(name):
      try:
        return getattr(np, name)
      except AttributeError:
        raise ValueError(f"Unsupported statistic function name: {name}")    

    if isinstance(stats, (list, tuple)):
      return {s: get_op(s)(vals) for s in stats}
    else:
      return {stats: get_op(stats)(vals)}

  def ci_metric(
      self,
      tag,
      stats: dict[str, float],
      unit: str = 's',
      sep: str = '|',
  ):
    return sep.join(
        [tag] + [f'{k}={v:.6f}' for k, v in stats.items()] + [f'unit={unit}']
    )

  def reset(self) -> None:
    """Clear all recorded data."""
    self._times.clear()
    self._stack.clear()
