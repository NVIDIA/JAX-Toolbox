from jax_nsys import (
    ensure_compiled_protos_are_importable,
    load_profiler_data,
)
import os
import pathlib
import pytest
import sys
import tempfile
import zipfile

helper_dir = os.path.join(os.path.dirname(__file__), "jax_nsys_test_helpers")
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)
from jax_nsys_test_helpers import nsys_jax


@pytest.fixture(scope="module")
def example_program():
    """
    Fixture that yields an extracted archive of the result of profiling
    example_program.py with nsys-jax.
    """
    tmpdir = tempfile.TemporaryDirectory()
    archive = nsys_jax(
        [sys.executable, os.path.join(os.path.dirname(__file__), "example_program.py")]
    )
    old_dir = os.getcwd()
    os.chdir(tmpdir.name)
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall()
    finally:
        os.chdir(old_dir)
    # Make sure the protobuf bindings can be imported, the generated .py will go into
    # a temporary directory that is not currently cleaned up. The bindings cannot be
    # un-imported from the test process, so there is a tacit assumption that in a given
    # test session there will only be one set of .proto files and it doesn't matter
    # which ones are picked up.
    ensure_compiled_protos_are_importable(prefix=pathlib.Path(tmpdir.name))
    return tmpdir


@pytest.fixture(scope="module")
def profiler_data(example_program):
    return load_profiler_data(pathlib.Path(example_program.name))


def test_comms(profiler_data):
    # example_program.py should contain no communication
    assert len(profiler_data.communication) == 0


def test_modules(profiler_data):
    assert sum(profiler_data.module["Name"] == "jit_distinctively_named_function") == 5
