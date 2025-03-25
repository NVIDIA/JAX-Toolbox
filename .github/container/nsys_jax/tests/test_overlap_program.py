from nsys_jax import load_profiler_data
import os
import pathlib
import pytest  # type: ignore
import sys

helper_dir = os.path.join(os.path.dirname(__file__), "nsys_jax_test_helpers")
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)
from nsys_jax_test_helpers import nsys_jax_archive  # noqa: E402


overlap_program = os.path.join(os.path.dirname(__file__), "overlap_program.py")


@pytest.fixture(scope="module")
def profiler_data_full():
    """
    Fixture that yields the loaded result of profiling overlap_program.py with nsys-jax.
    """
    outdir = nsys_jax_archive([sys.executable, overlap_program])
    return load_profiler_data(pathlib.Path(outdir.name))


@pytest.fixture(scope="module")
def profiler_data_narrow():
    """
    Fixture that yields the loaded result of profiling overlap_program.py with nsys-jax.
    """
    outdir = nsys_jax_archive(
        [
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--",
            sys.executable,
            overlap_program,
        ]
    )
    return load_profiler_data(pathlib.Path(outdir.name))


input_combinations = [
    ("profiler_data_full", 4),  # 4 executions in the program
    ("profiler_data_narrow", 2),  # 2 executions after cudaProfilerStart
]

module_name = "jit_where_the_magic_happens"
num_devices = 2


def get_program_id(module_data):
    test_func_mask = module_data["Name"] == module_name
    program_ids = module_data[test_func_mask].index.get_level_values("ProgramId")
    # All executions should have the same program id
    assert all(program_ids == program_ids[0])
    return program_ids[0]


@pytest.mark.parametrize("fixture_name,num_executions", input_combinations)
def test_comms(fixture_name, num_executions, request):
    profiler_data = request.getfixturevalue(fixture_name)
    module_id = get_program_id(profiler_data.module)
    assert profiler_data.communication.index.names == [
        "ProgramId",
        "ProgramExecution",
        "Name",
        "ThunkExecution",
        "Device",
    ]
    test_comm_data = profiler_data.communication.loc[module_id, :]
    assert test_comm_data.index.names == profiler_data.communication.index.names[1:]
    # 1 communication operation per execution
    assert len(test_comm_data) == num_executions * num_devices
    # all the collectives should be 2-device
    assert (test_comm_data["CollectiveSize"] == 2).all()
    # it's possible this will turn out to be flaky, as it is difficult to
    # *guarantee* overlap, but assert that we managed to at least hide *some*
    # communication behind *some* compute, on at least one device
    assert (test_comm_data["ProjDurHiddenMs"] > 0.0).any()
    # executions of the same collective on the 2 devices should overlap
    gpu0_df = test_comm_data.loc[(slice(None), slice(None), slice(None), 0), :]
    gpu1_df = test_comm_data.loc[(slice(None), slice(None), slice(None), 1), :]
    gpu0_end = (
        gpu0_df["ProjStartMs"] + gpu0_df["ProjDurMs"] + gpu0_df["ProjDurHiddenMs"]
    )
    gpu1_end = (
        gpu1_df["ProjStartMs"] + gpu1_df["ProjDurMs"] + gpu1_df["ProjDurHiddenMs"]
    )
    assert (gpu0_df["ProjStartMs"].array < gpu1_end.array).all()
    assert (gpu1_df["ProjStartMs"].array < gpu0_end.array).all()


@pytest.mark.parametrize("fixture_name,num_executions", input_combinations)
def test_modules(fixture_name, num_executions, request):
    profiler_data = request.getfixturevalue(fixture_name)
    module_id = get_program_id(profiler_data.module)
    assert profiler_data.module.index.names == [
        "ProgramId",
        "ProgramExecution",
        "Device",
    ]
    test_module_data = profiler_data.module.loc[module_id, :]
    assert test_module_data.index.names == profiler_data.module.index.names[1:]
    assert len(test_module_data) == num_executions * num_devices
    # executions of the same module on the 2 devices should overlap
    gpu0_df = test_module_data.loc[(slice(None), 0), :]
    gpu1_df = test_module_data.loc[(slice(None), 1), :]
    gpu0_end = gpu0_df["ProjStartMs"] + gpu0_df["ProjDurMs"]
    gpu1_end = gpu1_df["ProjStartMs"] + gpu1_df["ProjDurMs"]
    assert (gpu0_df["ProjStartMs"].array < gpu1_end.array).all()
    assert (gpu1_df["ProjStartMs"].array < gpu0_end.array).all()
    # all module executions should have the same number of thunks
    assert len(set(test_module_data["NumThunks"])) == 1
    # 1 process, 2 device execution => global=local, process=slice=0
    assert (test_module_data["Process"] == 0).all()
    assert (test_module_data["Slice"] == 0).all()
    assert (
        test_module_data.index.get_level_values("Device")
        == test_module_data["LocalDevice"]
    ).all()
