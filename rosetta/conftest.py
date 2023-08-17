# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import os


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.30"  # assumed parallelism: 3 (A6000 49GB VRAM)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # disable preallocation behavior
import runpy
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List
from unittest.mock import patch

import numpy as np
import pytest
import webdataset as wds
from absl import logging


MANUAL_MARKERS = {
    "integration": "used for integration tests (may be slow)",
    "data": "used for dataset utility tests",
    "perf": "used for speed tests",
    "convergence": "used for tests where a certain loss/accuracy is validated",
    "manual": "these are manual tests, e.g., things that require manually standing up a server or require certain datasets to be present",
}


def pytest_configure(config):
    for marker_name, marker_desc in MANUAL_MARKERS.items():
        config.addinivalue_line("markers", f"{marker_name}: {marker_desc}")


def pytest_addoption(parser):
    # TODO(terry): use these options to implement conjunctive/disjunctive selections
    for marker_name in MANUAL_MARKERS:
        parser.addoption(
            f"--{marker_name}",
            action="store_true",
            default=False,
            help=f"Run {marker_name} tests",
        )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    # trylast=True, is so we can let pytest filter out marks before we try to skip
    if not config.getoption("-m"):
        # We will disable integration, convergence and manual tests if markers aren't given
        for marker_name in ('integration', 'convergence', 'manual'):
            skipper = pytest.mark.skip(reason=f"Only run {marker_name} marked tests if present when specifying in -m <markexpr>")
            for item in items:
                if marker_name in item.keywords:
                    item.add_marker(skipper)


@pytest.fixture
def package_root_dir():
    return os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def rng():
    import jax
    return jax.random.PRNGKey(0)


@pytest.fixture
def run_subprocess_blocking():
    def _run_subprocess(*cmd_and_args, env=None):
        pipes = subprocess.Popen(
            cmd_and_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        stdout, stderr = pipes.communicate()
        return stdout, stderr, pipes.returncode
    return _run_subprocess


@pytest.fixture
def run_subprocess_in_background():

    def block_until_text_found(process, block_until_seen: str):
        for line in iter(process.stdout.readline, b''):
            line = line.decode()
            logging.info(line.rstrip())
            if block_until_seen in line:
                break

    @contextmanager
    def _run_subprocess(*cmd_and_args, block_until_seen: str, env=None):
        ON_POSIX = 'posix' in sys.builtin_module_names
        process = subprocess.Popen(
            cmd_and_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            close_fds=ON_POSIX,
            env=env,
        )
        block_until_text_found(process, block_until_seen)
        yield
        process.terminate()
        process.wait()
    return _run_subprocess


# This doesn't work with modules using gin configs relying on the __main__ module
@pytest.fixture
def run_module(capfd):
    def _run_module(module: str, argv: List[str] | None = None):
        argv = [module] + argv if argv else [module]
        with patch('sys.argv', argv):
            runpy.run_module(module, run_name='__main__')
        stdout, stderr = capfd.readouterr()
        return stdout, stderr
    return _run_module


@dataclass
class WebdatasetMetadata:
    num_examples: int = 20 * 4
    batch_size: int = 4
    image_size: int = 224
    channels: int = 3
    seq_len: int = 77
    image_key: str = 'jpg'
    text_key: str = 'txt'
    class_key: str = 'cls'
    num_classes: int = 10
    path: str | None = None


@pytest.fixture(scope='session')
def dummy_wds_metadata(
    tmp_path_factory: pytest.TempPathFactory,
):
    # HACK(terry): There is a bug in webdataset/writer.py that imports PIL, but not the module under it so we are doing it here as a WAR
    #   https://github.com/webdataset/webdataset/issues/198
    import PIL.Image  # noqa: F401
    metadata = WebdatasetMetadata()
    out_tar = tmp_path_factory.mktemp('wds_test') / 'dataset.tar'
    out_tar_path = out_tar.as_posix()
    with wds.TarWriter(out_tar_path) as dst:
        for index in range(metadata.num_examples):
            dst.write({
                "__key__": f"sample{index:06d}",
                metadata.image_key: np.full((metadata.image_size, metadata.image_size, metadata.channels), fill_value=1.0/index if index else 0.0, dtype=np.float32),
                metadata.class_key: index % metadata.num_classes,
                metadata.text_key: f'A random image #{index}',
            })
    metadata.path = out_tar_path
    yield metadata
