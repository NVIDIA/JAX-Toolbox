from ctypes import byref, cdll, c_int, POINTER
import os
import pytest  # type: ignore
import subprocess
import sys
import tempfile

helper_dir = os.path.join(os.path.dirname(__file__), "nsys_jax_test_helpers")
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)
from nsys_jax_test_helpers import multi_process_nsys_jax, nsys_jax  # noqa: E402


def visible_device_count() -> int:
    """
    Query the number of local devices visible to this process.
    """
    libcudart = cdll.LoadLibrary("libcudart.so")
    cudaGetDeviceCount = libcudart.cudaGetDeviceCount
    cudaGetDeviceCount.argtypes = [POINTER(c_int)]
    cudaGetDeviceCount.restype = c_int
    count = c_int()
    assert cudaGetDeviceCount(byref(count)) == 0
    return count.value


def capture_args(collection):
    return {
        "full": [],
        "partial": [],
    }[collection]


def set_env(monkeypatch):
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
    # Disable CUDA graphs
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")


@pytest.mark.parametrize("collection", ["full", "partial"])
def test_jax_nccl_single_process(monkeypatch, tmp_path, collection):
    set_env(monkeypatch)
    nsys_jax(
        capture_args(collection)
        + [
            "--nsys-jax-analysis",
            "communication",
            "--nsys-jax-analysis",
            "summary",
            "--",
            "jax-nccl-test",
        ],
        out_dir=tmp_path,
    )


process_counts_to_test = []
device_count = visible_device_count()
if device_count >= 2:
    # Use two processes with GPUS_PER_NODE/2 GPUs per process in the hope that
    # this will flush out more bugs than process-per-node or process-per-GPU.
    process_counts_to_test.append(2)
if device_count >= 3:
    process_counts_to_test.append(device_count)


@pytest.mark.parametrize("process_count", process_counts_to_test)
@pytest.mark.parametrize("collection", ["full", "partial"])
def test_jax_nccl_multi_process(monkeypatch, tmp_path, process_count, collection):
    assert device_count % process_count == 0, (device_count, process_count)
    gpus_per_process = device_count // process_count
    set_env(monkeypatch)
    outputs = multi_process_nsys_jax(
        process_count,
        lambda rank: (
            capture_args(collection)
            + [
                "--",
                "jax-nccl-test",
                "--process-id",
                str(rank),
                "--process-count",
                str(process_count),
                "--coordinator-address",
                "127.0.0.1:12345",
                "--gpus-per-process",
                str(gpus_per_process),
                "--distributed",
            ]
        ),
        out_dir=tmp_path,
    )
    combined_output = tempfile.NamedTemporaryFile(suffix=".zip")
    subprocess.run(
        [
            "nsys-jax-combine",
            "--force-overwrite",
            "--output",
            combined_output.name,
            "--analysis",
            "communication",
            "--analysis",
            "summary",
        ]
        + [output.name for output in outputs],
        check=True,
    )
