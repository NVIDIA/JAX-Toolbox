import os
import pytest  # type: ignore
import sys

helper_dir = os.path.join(os.path.dirname(__file__), "nsys_jax_test_helpers")
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)
from nsys_jax_test_helpers import nsys_jax, nsys_jax_with_result  # noqa: E402

# This example program does different things after calling cudaProfilerStop
cuda_profiler_api = os.path.join(os.path.dirname(__file__), "cuda_profiler_api.py")

kill_args = {
    None: [],
    "none": ["--kill=none"],
    "sigterm": ["--kill", "sigterm"],
    "sigkill": ["--kill=sigkill"],
    "SIGINT": ["--kill", "2"],
}


@pytest.mark.parametrize("kill", kill_args.keys())
def test_program_that_is_killed_by_nsys(kill):
    """
    The default --capture-range-end=stop-shutdown behaviour of nsys profile causes nsys
    to kill the profiled process after cudaProfilerStop if
    --capture-range=cudaProfilerApi is used. Test that in this case, nsys-jax returns
    success (bug 5049736).
    """
    if kill == "none":
        pytest.skip("Expected to hang")
    output_zip = nsys_jax(
        kill_args[kill]
        + [
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop-shutdown",
            "--",
            sys.executable,
            cuda_profiler_api,
            "sleep",
        ]
    )
    assert os.path.isfile(output_zip.name)


def test_program_that_fails_after_cuda_profiler_stop():
    """
    With --capture-range=stop then nsys-jax should still propagate a failure code from
    the application.
    """
    output_zip, result = nsys_jax_with_result(
        [
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--",
            sys.executable,
            cuda_profiler_api,
            "exit42",
        ]
    )
    assert result.returncode == 42, result
    assert os.path.isfile(output_zip.name)


@pytest.mark.parametrize("kill", kill_args.keys())
def test_program_that_fails_after_cuda_profiler_stop_as_nsys_tries_to_kill_it(kill):
    """
    The racy case, where either nsys sends SIGTERM or the application exits with 42.
    Also cover the case where --capture-range-end is not passed explicitly.
    """
    output_zip, result = nsys_jax_with_result(
        kill_args[kill]
        + [
            "--capture-range=cudaProfilerApi",
            "--",
            sys.executable,
            cuda_profiler_api,
            "exit42",
        ]
    )
    # If nsys sends SIGTERM fast enough, the child process will exit due to that and
    # nsys-jax will return 0 (because that is the expected result). The application
    # might manage to return 42 before it is killed, which should be propagated by
    # nsys-jax because it is not expected from the --capture-range-end setting.
    print(f"nsys-jax returned code {result.returncode}")
    assert result.returncode in {0, 42}, result
    assert os.path.isfile(output_zip.name)
