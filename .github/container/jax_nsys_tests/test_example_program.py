from jax_nsys import (
    ensure_compiled_protos_are_importable,
    load_profiler_data,
)
import os
import pathlib
import pytest  # type: ignore
import sys
import tempfile
import zipfile

helper_dir = os.path.join(os.path.dirname(__file__), "jax_nsys_test_helpers")
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)
from jax_nsys_test_helpers import nsys_jax  # noqa: E402


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
    test_func_mask = profiler_data.module["Name"] == "jit_distinctively_named_function"
    assert sum(test_func_mask) == 5
    test_func_data = profiler_data.module[test_func_mask]
    assert test_func_data.index.names == ["ProgramId", "ProgramExecution", "Device"]
    # All executions should have the same program id
    program_ids = test_func_data.index.get_level_values("ProgramId")
    assert all(program_ids == program_ids[0])
    # All executions should be on device 0
    execution_devices = test_func_data.index.get_level_values("Device")
    assert all(execution_devices == 0)
    # Execution indices should count from 0..n-1
    execution_indices = test_func_data.index.get_level_values("ProgramExecution")
    assert all(execution_indices == range(len(test_func_data)))
