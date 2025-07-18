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

import pytest
import jax

from jax_cutlass import release_compile_cache
from .benchmark import cupti_profile, BenchmarkCollector


def pytest_addoption(parser):
    parser.addoption("--benchmark_iters", default=16, action="store", type=int)
    parser.addoption("--benchmark", action="store_true")
    parser.addoption("--check_tracer_leaks", action="store_true")


def pytest_sessionstart(session):
    session.stash["collector"] = BenchmarkCollector(
        session.config.option.benchmark, session.config.option.benchmark_iters
    )

    if session.config.option.check_tracer_leaks:
        jax.check_tracer_leaks(True)


def pytest_sessionfinish(session):
    session.stash["collector"].save_csv()


@pytest.fixture
def benchmark(request):
    collector = request.session.stash["collector"]
    collector.set_current_request(request)
    yield collector
    collector.set_current_request(None)


@pytest.fixture(scope="function", autouse=True)
def clear_cache_and_live_arrays_after_test():
    yield
    jax.clear_caches()
    release_compile_cache()
    for a in jax.live_arrays():
        a.delete()
