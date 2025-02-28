from nsys_jax import load_profiler_data
import os
import pathlib
import pytest  # type: ignore
import sys

helper_dir = os.path.join(os.path.dirname(__file__), "nsys_jax_test_helpers")
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)
from nsys_jax_test_helpers import nsys_jax_archive  # noqa: E402


@pytest.fixture(scope="module")
def profiler_data():
    """
    Fixture that yields the loaded result of profiling example_program.py with nsys-jax.
    """
    outdir = nsys_jax_archive(
        [sys.executable, os.path.join(os.path.dirname(__file__), "example_program.py")]
    )
    return load_profiler_data(pathlib.Path(outdir.name))


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
