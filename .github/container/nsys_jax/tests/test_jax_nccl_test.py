from ctypes import byref, cdll, c_int, POINTER
import itertools
import math
from nsys_jax import (
    apply_warmup_heuristics,
    ensure_compiled_protos_are_importable,
    load_profiler_data,
)
import os
import pathlib
import portpicker
import pytest  # type: ignore
import subprocess
import sys
import tempfile

helper_dir = os.path.join(os.path.dirname(__file__), "nsys_jax_test_helpers")
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)
from nsys_jax_test_helpers import extract, multi_process_nsys_jax  # noqa: E402


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


process_counts_to_test = []
device_count = visible_device_count()
if device_count >= 2:
    # Use two processes with GPUS_PER_NODE/2 GPUs per process in the hope that
    # this will flush out more bugs than process-per-node or process-per-GPU.
    process_counts_to_test.append(2)
if device_count >= 3:
    process_counts_to_test.append(device_count)

replica_counts_to_test = [1]
if device_count >= 4:
    # If we have at least 4 GPUs, also run with two groups of GPUS_PER_NODE/2 GPUs
    replica_counts_to_test.append(2)


MIN_SIZE_POWER = 4
MAX_SIZE_POWER = 30
NUM_ITERATIONS = 4


@pytest.fixture(
    scope="module",
    params=itertools.product(
        replica_counts_to_test, process_counts_to_test, ["full", "partial"]
    ),
)
def individual_results(tmp_path_factory, request):
    replica_count, process_count, collection = request.param
    assert device_count % process_count == 0, (device_count, process_count)
    gpus_per_process = device_count // process_count
    port = portpicker.pick_unused_port()

    def distributed_args(rank):
        if process_count == 1:
            return []
        return [
            "--process-id",
            str(rank),
            "--process-count",
            str(process_count),
            "--coordinator-address",
            f"127.0.0.1:{port}",
            "--gpus-per-process",
            str(gpus_per_process),
            "--distributed",
        ]

    capture_args = {
        "full": [],
        "partial": ["--capture-range=cudaProfilerApi", "--capture-range-end=stop"],
    }[collection]
    return multi_process_nsys_jax(
        process_count,
        lambda rank: (
            capture_args
            + [
                "--",
                "jax-nccl-test",
            ]
            + distributed_args(rank)
            + [
                "--replica-count",
                str(replica_count),
                "--num-iterations",
                str(NUM_ITERATIONS),
                "--min-size-power",
                str(MIN_SIZE_POWER),
                "--max-size-power",
                str(MAX_SIZE_POWER),
            ]
        ),
        out_dir=tmp_path_factory.mktemp("jax-nccl-multiprocess"),
    ), {"replica_count": replica_count}


@pytest.mark.parametrize("recipe", ["summary", "communication"])
def test_analysis_recipes(individual_results, recipe):
    """
    Test that the analysis recipes can swallow jax-nccl-test data.
    """
    individual_files, metadata = individual_results
    with tempfile.NamedTemporaryFile(suffix=".zip") as combined_output:
        subprocess.run(
            [
                "nsys-jax-combine",
                "--force-overwrite",
                "--output",
                combined_output.name,
                "--analysis",
                recipe,
            ]
            + [output.name for output in individual_files],
            check=True,
        )


@pytest.fixture(scope="module")
def combined_results(individual_results, tmp_path_factory):
    """
    Fixture that yields the merged result from nsys-jax-combine.
    """
    individual_files, metadata = individual_results
    combined_output = (
        tmp_path_factory.mktemp("jax-nccl-test-multiprocess-combined") / "combined.zip"
    )
    subprocess.run(
        ["nsys-jax-combine", "--output", combined_output]
        + [output.name for output in individual_files],
        check=True,
    )
    combined_dir = extract(combined_output)
    combined_output_path = pathlib.Path(combined_dir.name)
    # Make sure the protobuf bindings can be imported, the generated .py will go into
    # a temporary directory that is not currently cleaned up. The bindings cannot be
    # un-imported from the test process, so there is a tacit assumption that in a given
    # test session there will only be one set of .proto files and it doesn't matter
    # which ones are picked up.
    ensure_compiled_protos_are_importable(prefix=combined_output_path)
    return combined_dir, combined_output_path, metadata


@pytest.fixture(scope="module")
def combined_data(combined_results):
    _, combined_output_path, metadata = combined_results
    return load_profiler_data(combined_output_path), metadata


@pytest.fixture(scope="module")
def comms_data(combined_data):
    combined, metadata = combined_data
    _, steady_state = apply_warmup_heuristics(combined)
    stats = steady_state.communication.groupby(["Collective", "MessageSize"]).agg(
        {"CollectiveSize": ("min", "max", "count")}
    )
    # We should only see one collective size per instance of this test case
    assert (
        stats[("CollectiveSize", "min")] == stats[("CollectiveSize", "max")]
    ).all(), stats
    return stats, metadata["replica_count"]


@pytest.mark.parametrize(
    "collective",
    [
        "all-gather",
        "all-reduce",
        "collective-broadcast",
        "collective-permute",
        "reduce-scatter",
    ],
)
def test_multi_process(comms_data, collective):
    stats, replica_count = comms_data
    # The data type is implicitly 4 bytes. There will be a small all-gather from the
    # sync collective, but this should be below 2**MIN_SIZE_POWER.
    expected_sizes = [4 * 2**p for p in range(MIN_SIZE_POWER, MAX_SIZE_POWER + 1)]
    if collective == "reduce-scatter":
        # The convention is that the message size refers to the output size, which is a
        # factor `CollectiveSize` smaller than the input size that scans between
        # MIN_SIZE_POWER and MAX_SIZE_POWER. This means some of the values in
        # `expected_sizes` are not actually expected for reduce scatter.
        sizes_to_drop = int(math.log2(device_count // replica_count))
        expected_sizes = expected_sizes[:-sizes_to_drop]
    df = stats.loc[collective, :]
    # Did we see instances with all expected sizes?
    seen_message_sizes = df.index.get_level_values("MessageSize").unique()
    assert set(seen_message_sizes) >= set(expected_sizes), (
        seen_message_sizes,
        expected_sizes,
    )
    # What was the collective size corresponding to the expected message sizes?
    expected_message_size_df = df[
        df.index.get_level_values("MessageSize").isin(expected_sizes)
    ]
    collective_sizes = expected_message_size_df[("CollectiveSize", "min")].unique()
    assert len(collective_sizes) == 1, collective_sizes
    collective_size = next(iter(collective_sizes))
    if collective in {
        "all-gather",
        "all-reduce",
        "collective-broadcast",
        "reduce-scatter",
    }:
        assert collective_size == device_count // replica_count
    else:
        assert collective == "collective-permute", collective
        assert collective_size == 2
    # Did we see each size the expected number of times. We should see every collective
    # captured on every device. If the whole execution is profiled, there will be
    # NUM_ITERATIONS runs in total but 2 will be discarded by apply_warmup_heuristics,
    # if the cudaProfilerApi range is used then NUM_ITERATIONS-1 will be captured in the
    # profile and 1 will be discarded by apply_warmup_heuristics.
    collective_counts = expected_message_size_df[("CollectiveSize", "count")]
    assert (collective_counts == device_count * (NUM_ITERATIONS - 2)).all(), (
        collective_counts.unique()
    )
