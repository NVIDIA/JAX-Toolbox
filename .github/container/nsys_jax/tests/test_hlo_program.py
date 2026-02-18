from nsys_jax import load_profiler_data
import os
import pathlib
import pytest  # type: ignore
import shutil
import sys

helper_dir = os.path.join(os.path.dirname(__file__), "nsys_jax_test_helpers")
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)
from nsys_jax_test_helpers import nsys_jax_archive, nsys_version  # noqa: E402

num_repeats = 2
hlo_runner_main = shutil.which("hlo_runner_main")
nsys_version_tup = nsys_version()
pytestmark = [
    pytest.mark.skipif(
        hlo_runner_main is None, reason="HLO runner binary not available"
    ),
    pytest.mark.skipif(
        nsys_version_tup[:3] == (2026, 1, 1) or nsys_version_tup == (2026, 1, 2, 63),
        reason="nvbug/5910527",
    ),
]


@pytest.fixture(scope="module")
def profiler_data(tmp_path_factory):
    """
    Fixture that yields the loaded result of profiling example_program.py with nsys-jax.
    """
    outdir = nsys_jax_archive(
        [
            hlo_runner_main,
            f"--num_repeats={num_repeats}",
            os.path.join(os.path.dirname(__file__), "example.hlo"),
        ],
        out_dir=tmp_path_factory.mktemp("test_hlo_program"),
    )
    return load_profiler_data(pathlib.Path(outdir.name))


def test_comms(profiler_data):
    # example_program.py should contain no communication
    assert len(profiler_data.communication) == 0


def test_modules(profiler_data):
    # Find the executions of the main example.hlo. There may be other modules executed by the autotuner.
    example_mask = profiler_data.module["Name"] == "some_hlo_module"
    example_executions = sum(example_mask)
    assert example_executions % num_repeats == 0, (example_executions, num_repeats)
    num_devices = example_executions // num_repeats
    assert num_devices > 0, num_devices
    example_data = profiler_data.module[example_mask]
    # Check the data frame has the expected structure
    assert example_data.index.names == ["ProgramId", "ProgramExecution", "Device"]
    # All executions should have the same program id
    program_ids = example_data.index.get_level_values("ProgramId")
    assert all(program_ids == program_ids[0])
    # ProgramExecution should be [0, 0, 0, 1, 1, 1, ...] for 3 devices
    assert all(
        example_data.index.get_level_values("ProgramExecution")
        == [n for n in range(num_repeats) for _ in range(num_devices)]
    )
    # Device should be [0, 1, 2, 0, 1, 2, ...] for 3 devices
    assert all(
        example_data.index.get_level_values("Device")
        == list(range(num_devices)) * num_repeats
    )
