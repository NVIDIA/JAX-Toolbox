# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import groupby
from contextlib import contextmanager
from collections import defaultdict

import jax.numpy as jnp


def cupti_profile(f):
    """Profiles a callable `f` and returns CUPTI profiled timings.

    Returns (pytree_result_f, timings)
    """

    from jax._src.lib import mosaic_gpu as mosaic_gpu_lib

    def wrapped(*args):
        result = None
        timings = None
        try:
            ext = mosaic_gpu_lib._mosaic_gpu_ext
            ext._cupti_init()
            result = f(*args)
        finally:
            timings = ext._cupti_get_timings(True)
        return result, timings

    return wrapped


class BenchmarkRunner:
    def __init__(self, num_iterations):
        self.num_iterations = num_iterations

    @property
    def enabled(self):
        return self.num_iterations > 0

    def __call__(self, fn, *args, **kwargs):
        """Calls the given function num_iterations times."""
        out = None  # keep only last output
        for _ in range(self.num_iterations):
            out = fn(*args, **kwargs)
        return out

    def __iter__(self):
        """Returns an iterable for num_iterations."""
        return iter(range(self.num_iterations))


@contextmanager
def cupti_benchmark_profiler_runner_context(request, filename, collector, iter_count):
    """A context manager for collecting benchmark data with CUPTI for a number of iterations."""
    try:
        from jax._src.lib import mosaic_gpu as mosaic_gpu_lib

        if iter_count > 0:
            ext = mosaic_gpu_lib._mosaic_gpu_ext
            ext._cupti_init()
        yield BenchmarkRunner(iter_count)
    finally:
        if collector.enabled:
            timings = ext._cupti_get_timings(True)
            collector.record_timings(request, filename, timings)


class BenchmarkCollector:
    def __init__(self, enabled, default_benchmark_iters=16):
        # file name -> result dict
        self.enabled = enabled
        self.results = defaultdict(lambda: defaultdict(list))
        self.default_benchmark_iters = default_benchmark_iters
        self.request = None

    def set_current_request(self, request):
        """Sets the current pytest request."""
        self.request = request

    def _write_one_benchmark_result_csv(self, filename, results):
        with open(filename, "w") as fp:
            for key in results:
                gkey = lambda x: x[0]
                for kernel_key, group in groupby(
                    sorted(results[key], key=gkey), key=gkey
                ):
                    # header
                    for key_entry in key:
                        if not isinstance(key_entry[1], (list, tuple)):
                            fp.write(f"{key_entry[1]},")
                        else:
                            for x in key_entry[1]:
                                if isinstance(x, jnp.dtype):
                                    fp.write(f"{x.__nane__},")
                                else:
                                    fp.write(f"{x},")
                    # data
                    values = list([x[1] for x in group])
                    total, count, minv, maxv = (
                        sum(values),
                        len(values),
                        min(values),
                        max(values),
                    )
                    fp.write(f"{kernel_key},{count},{total/count},{minv},{maxv}\n")

    def save_csv(self):
        """Save the recorded benchmark data files."""
        for filename in self.results:
            self._write_one_benchmark_result_csv(filename, self.results[filename])

    def _benchmark_key(self, request):
        key = [("name", request.node.name)]
        arg_names = sorted(list(request.node.callspec.params.keys()))
        for arg in arg_names:
            key.append((arg, request.node.callspec.params[arg]))
        return tuple(key)

    def record_timings(self, request, filename, timings):
        """Records the timings from the request to a specific file."""
        if not self.enabled:
            raise RuntimeError("Collection is not enabled.")
        key = self._benchmark_key(request)
        self.results[filename][key].extend(timings)

    def runner(self, filename, num_iters=None):
        """Returns a `cupti_benchmark_profiler_runner_context` for collecting.

        with benchmark.runner(request, "blackwell_dense_gemm.txt") as runner:
            runner(launch, a, b)
        """
        if self.request is None:
            raise RuntimeError("No request was set.")
        if num_iters is None:
            num_iters = self.default_benchmark_iters
        if not self.enabled:
            num_iters = 0
        return cupti_benchmark_profiler_runner_context(
            self.request, filename, self, num_iters
        )
