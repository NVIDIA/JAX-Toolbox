from collections import defaultdict
import functools
import math
import numpy as np
import pandas as pd  # type: ignore
import pathlib
from typing import Any

from .protobuf import HloProto, _host_memory_space, xla_module_metadata
from .utils import make_child_mask, ProfilerData

pd.options.mode.copy_on_write = True


def align_profiler_data_timestamps(
    frames: ProfilerData,
) -> tuple[ProfilerData, dict[str, Any]]:
    """
    Given a ProfilerData dataclass, as returned by `load_profiler_data`, perform a time
    alignment across profiles collected in different processes. This is based on the
    end timestamps of collectives involving all devices that were profiled.

    Returns a tuple of:
      ProfilerData data class derived from the input, but with device-side timestamps
        corrected
      dictionary of information about the alignment process
    """
    # Error if the communication frame doesn't exist at all, but not if it is empty.
    # Calling this on a profile that does not contain any communication should
    # gracefully yield empty results.
    assert (
        frames.communication is not None
    ), "align_profiler_data_timestamps requires a communication frame"
    if not len(frames.communication):
        # Nothing to be done, return an empty result
        return frames, {}
    comm_df = frames.communication
    # Determine which collective size will be used for the alignment
    num_profiled_devices = len(comm_df.index.get_level_values("Device").unique())
    max_collective_size = comm_df["CollectiveSize"].max()
    if max_collective_size == 1:
        print(
            f"WARNING: cannot align {num_profiled_devices} devices because max collective size is 1"
        )
        return frames, {}
    assert (
        num_profiled_devices == max_collective_size
    ), f"Aligning {num_profiled_devices} using collectives of size {max_collective_size} is not implemented"
    # Find the collectives that will be used
    align_df = comm_df[comm_df["CollectiveSize"] == max_collective_size]
    # Calculate the collectives' end times
    end_times = (
        align_df["ProjStartMs"] + align_df["ProjDurMs"] + align_df["ProjDurHiddenMs"]
    )
    # For each collective, calculate the mean end time of each collective across devices
    mean_end_times = end_times.groupby(
        ["ProgramId", "ProgramExecution", "ThunkIndex"]
    ).agg("mean")
    # For each collective + device, calculate the delta of the end time from the mean
    end_time_skews = end_times - mean_end_times
    device_skews = end_time_skews.groupby("Device")
    median_device_skews = device_skews.agg("median")
    # Apply these corrections to the device-side timestamps
    for k in ["communication", "module", "thunk"]:
        df = getattr(frames, k)
        if df is None:
            continue
        df["ProjStartMs"] -= median_device_skews
        setattr(frames, k, df)
    return frames, {
        "collective_end_time_skews_ms": end_time_skews,
        "device_corrections": median_device_skews,
        "collective_size": max_collective_size,
    }


def apply_warmup_heuristics(frames: ProfilerData) -> tuple[ProfilerData, ProfilerData]:
    """
    Given a ProfilerData dataclass, as returned by `load_profiler_data`, use heuristics
    to split the profile data into initialisation and steady state running. The current
    approach is to assume everything is steady state if compilation was not profiled,
    and if compilation *was* profiled then label the 0th execution as initialisation
    and the 2nd and later ones as steady state operation, discarding one execution in
    between. If there is no communication in the profile, that one in between is not
    discarded.

    Returns a tuple of:
      ProfilerData dataclass, with only initialisation (and compile)
      ProfilerData dataclass, with only steady state (and no compile)
    """
    assert frames.compile is not None
    # Which program IDs did we witness compilation of?
    compilation_ids_seen = sorted(frames.compile["ProgramId"].unique())
    # Generally the first execution of a module will be slower, but we don't know for
    # sure if the profile being analysed included the whole runtime or was more
    # selective. As a heuristic, we can skip the first two executions of modules that
    # we saw the compilation of. The motivation for skipping the second executions is
    # that with a typical structure like:
    #
    # for n in range(n_iterations):
    #  preamble(n)
    #  step_function(n) # involves collectives/synchronisation
    #  postamble(n)
    #
    # then one-time costs (e.g. JIT compilation) of postamble(0) will affect when
    # step_function(1) is actually launched, whereas step_function(2) and later are
    # expected to launch closer to in lockstep across processes.
    init = ProfilerData(compile=frames.compile)
    steady = ProfilerData()
    steady_state_threshold = (
        1 if frames.communication is not None and len(frames.communication) else 0
    )
    for k in ["communication", "thunk", "module"]:
        df = getattr(frames, k)
        if df is None:
            continue
        compile_mask = df.index.get_level_values("ProgramId").isin(compilation_ids_seen)
        prog_exec_values = df.index.get_level_values("ProgramExecution")
        init_mask = compile_mask & (prog_exec_values == 0)
        steady_mask = ~compile_mask | (prog_exec_values > steady_state_threshold)
        if len(df) != 0 and not steady_mask.any():
            print(
                f"WARNING: heuristics could not identify steady-state execution in {k} frame, assuming EVERYTHING is steady-state. You may want to increase the number of profiled executions."
            )
            setattr(init, k, df[steady_mask])
            setattr(steady, k, df[~steady_mask])
        else:
            assert (
                steady_state_threshold == 0
                or (prog_exec_values[~init_mask & ~steady_mask] == 1).all()
            )
            setattr(init, k, df[init_mask])
            setattr(steady, k, df[steady_mask])
    return init, steady


def element_type_width(element_type: int) -> int:
    """
    Given an int representing an XLA PrimitiveType enum value, return the width of that
    type in bits.
    """
    from xla.xla_data_pb2 import PrimitiveType

    enum_name = PrimitiveType.Name(element_type)
    if enum_name == "PRED":
        # Based on
        # https://github.com/openxla/xla/blob/664a36a2b5e5be9179c5841830da56799b6dfe60/xla/service/gpu/runtime/nccl_api.cc#L116-L118
        return 8

    # There are several 8-bit floating point types of the form F8E{n}M{m}...
    if enum_name.startswith("F8E"):
        return 8

    # S32 is a 32-bit type and so on.
    for prefix in ["BF", "C", "F", "S", "U"]:
        if enum_name.startswith(prefix):
            return int(enum_name[len(prefix) :])
    raise Exception(f"Could not deduce size of {enum_name}")


def _collective_correction(kind: str, size: int) -> tuple[float, float]:
    """
    Calculate the correction factor from algorithm bandwidth to bus bandwidth, see:
    https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bus-bandwidth
    """
    match kind:
        # For AllGather the size in the bandwidth calculation is the total/output size
        # https://github.com/NVIDIA/nccl-tests/blob/c6afef0b6f76ffc55d4172d971be6cf5a08a73a4/doc/PERFORMANCE.md#allgather
        case "all-gather":
            return (size, (size - 1) / size)
        case "all-reduce":
            return (1, 2 * (size - 1) / size)
        case "all-to-all":
            # https://github.com/NVIDIA/nccl-tests/blob/a1efb427e764241bc43d2d91be875c9f55da03a5/src/alltoall.cu#L44
            return (1, (size - 1) / size)
        case "collective-broadcast":
            return (1, 1)
        case "collective-permute":
            return (1, 1)
        # For ReduceScatter the size in the bandwidth calculation is the total size
        # https://github.com/NVIDIA/nccl-tests/blob/c6afef0b6f76ffc55d4172d971be6cf5a08a73a4/doc/PERFORMANCE.md#reducescatter
        case "reduce-scatter":
            return (size, (size - 1) / size)
        case _:
            assert False, f"Unknown collective kind {kind}"


def _get_message_size(
    module_proto: HloProto, instruction: str
) -> tuple[int, str, int, float, float]:
    _, inst = module_proto.find_instruction(instruction)
    comm_inst = inst.communication_proto()
    assert (
        comm_inst.opcode
        in {
            "all-gather-start",
            "all-reduce-start",
            "all-to-all",
            "collective-broadcast",
            "collective-permute-start",
            "dynamic-slice",
            "dynamic-update-slice",
            "reduce-scatter",
        }
    ), f"{instruction}: message size calculation for {comm_inst.opcode} has not yet been validated"

    def _byte_size(inst) -> int:
        size_bits = math.prod(
            inst.shape.dimensions,
            start=element_type_width(inst.shape.element_type),
        )
        size_bytes, rem = divmod(size_bits, 8)
        assert rem == 0
        return size_bytes

    if comm_inst.opcode == "collective-permute-start":
        # See https://openxla.org/xla/operation_semantics#collectivepermute, which
        # generates pair-wise send+recv between devices
        collective_size = 2
    elif comm_inst.opcode in {"dynamic-slice", "dynamic-update-slice"}:
        # Label host-device transfers orchestrated by dynamic[-update]-slice as single
        # device collectives.
        collective_size = 1
        if comm_inst.opcode == "dynamic-update-slice":
            # For dynamic-update-slice the second operand is the one being copied
            _, src_inst = module_proto.find_instruction_by_id(comm_inst.operand_ids[1])
            transfer_size = _byte_size(src_inst.proto())
        else:
            # For dynamic-slice the return type size is the transfer size
            assert comm_inst.opcode == "dynamic-slice"
            _, src_inst = module_proto.find_instruction_by_id(comm_inst.operand_ids[0])
            transfer_size = _byte_size(comm_inst)
        dest_on_host = _host_memory_space(comm_inst)
        src_on_host = _host_memory_space(src_inst.proto())
        assert src_on_host != dest_on_host, (
            'dynamic[-update]-slice is only considered is only "communication" if it '
            "represents a host-device transfer"
        )
        return (
            transfer_size,
            "device-to-host" if dest_on_host else "host-to-device",
            1,  # collective size
            1.0,  # bw_correction
            1.0,  # bus_correction
        )
    else:
        # replica_groups is something like {{0,1},{4,5},{2,3},{6,7}}, if there are 8
        # devices that are doing pair-wise collectives
        try:
            replica_groups = comm_inst.collective_device_list.replica_groups
        except AttributeError:
            replica_groups = comm_inst.replica_groups
        if len(replica_groups) == 0:
            # perhaps we have the newer format
            iota_group_list = comm_inst.collective_device_list.iota_replica_group_list
            collective_size = iota_group_list.num_devices_per_group
        else:
            collective_sizes = set(len(group.replica_ids) for group in replica_groups)
            assert (
                len(collective_sizes) == 1
            ), f"Heterogeneous collective {comm_inst} could not be interpreted"
            collective_size = next(iter(collective_sizes))
    total_msg_size = 0
    for operand_id in comm_inst.operand_ids:
        _, operand = module_proto.find_instruction_by_id(operand_id)
        msg_size_bytes = _byte_size(operand.proto())
        if comm_inst.opcode == "reduce-scatter":
            # NCCL's convention is that the message size of a reduce-scatter is the size of output buffer:
            # https://github.com/NVIDIA/nccl/blob/ab2b89c4c339bd7f816fbc114a4b05d386b66290/src/collectives.cc#L122
            msg_size_bytes, rem = divmod(msg_size_bytes, collective_size)
            assert rem == 0
        total_msg_size += msg_size_bytes

    collective = comm_inst.opcode.removesuffix("-start")
    bw_correction, bus_correction = _collective_correction(collective, collective_size)
    return (total_msg_size, collective, collective_size, bw_correction, bus_correction)


@functools.cache
def get_message_size(
    program_id: int, instruction: str, prefix: pathlib.Path
) -> pd.Series:
    """
    Given the name of a collective instruction (e.g. all-gather-start.N), calculate the
    message size in bytes. See https://openxla.org/xla/operation_semantics#allgather,
    https://openxla.org/xla/operation_semantics#allreduce and so on for more explanation
    of the semantics. This implementation aims to follow the same conventions that NCCL
    uses in its NVTX payloads and tests.
    """
    return pd.Series(
        xla_module_metadata(program_id, prefix=prefix, policy="all").unique_result(
            lambda proto: _get_message_size(proto, instruction)
        ),
        index=[
            "MessageSize",
            "Collective",
            "CollectiveSize",
            "BandwidthCorrection",
            "BusBandwidthCorrection",
        ],
    )


def calculate_collective_metrics(
    thunk_df: pd.DataFrame, prefix: pathlib.Path
) -> pd.DataFrame:
    """
    Given a "thunk" data frame from `load_profiler_data`, produce a new data frame that
    contains one row per communication thunk and contains extra metrics such as the
    message size, algorithm bandwidth, bus bandwidth, and collective operation.
    """
    comm_df = thunk_df[thunk_df["Communication"]].drop(columns=["Communication"])
    if len(comm_df) == 0:
        return comm_df
    comm_df = pd.concat(
        [
            comm_df,
            comm_df.apply(
                lambda row: get_message_size(row.name[0], row.Name, prefix=prefix),
                axis=1,
            ),
        ],
        axis=1,
    )
    # Note that this is decimal GB not binary GiB; GB/s == B/ns == 1e-6 * B / ms
    comm_df["AlgorithmBandwidthGBPerSec"] = (
        1e-6
        * comm_df["BandwidthCorrection"]
        * comm_df["MessageSize"]
        / (comm_df["ProjDurMs"] + comm_df["ProjDurHiddenMs"])
    )
    comm_df["BusBandwidthGBPerSec"] = (
        comm_df["AlgorithmBandwidthGBPerSec"] * comm_df["BusBandwidthCorrection"]
    )
    return comm_df.drop(columns=["BandwidthCorrection", "BusBandwidthCorrection"])


def generate_compilation_statistics(compile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse a "compile" data frame returned by `load_profiler_data` and aggregate over
    different compilations in a parallel-compilation-aware way.

    Both autotuning and backend compilation can use thread pools in XLA, which needs to
    be handled explicitly -- naive handling would result in individual stages of the
    compilation summing to >100% of the wall clock compilation time.

    This utility identifies parallel regions of the compilation and attributes the time
    spent in each parallel region based on the activity of the worker threads.

    For example, in a parallel region that takes 20 seconds to execute, where on
    average the worker threads spent 25% of their time on task A and 75% on task B, 5
    seconds of task A and 15 seconds of task B will be accounted in the returned
    statistics.
    """
    # Aggregate compilation stats in here
    compile_time_ms: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2))
    for profile_name, profile_df in compile_df.groupby("ProfileName"):
        # Identify the main thread
        main_thread = profile_df.loc[compile_df["Name"] == "XlaCompile", "TID"].unique()
        assert len(main_thread) == 1
        main_thread = main_thread[0]

        # Identify the ranges in the main thread that represent parallel compilation, i.e.
        # ranges whose child ranges are in different threads.
        worker_parent_ids = profile_df.loc[
            profile_df["TID"].ne(main_thread), "ParentId"
        ].astype(np.int32)
        # These are the main-thread ranges that directly contain parallel workers
        launcher_mask = profile_df.loc[(profile_name, worker_parent_ids), "TID"].eq(
            main_thread
        )
        launcher_ids = launcher_mask[launcher_mask].index.unique()
        # Loop over the main-thread ranges that launched parallel work
        for launcher_row in profile_df.loc[launcher_ids, :].itertuples():
            assert launcher_row.TID == main_thread
            # Find all child ranges; some may still be in the main thread. Assume for now
            # that the sequence will just be something like:
            #   M(A) M(A) M(A) .. W1(B) W2(B) W3(B) W1(B) .. M(C) M(C)
            # i.e. the main thread M does some task (A), then workers W{1,2,3} do some task
            # (B) in parallel, then the main thread continues with another task (C),
            # without overlap between A, B and C. For simplicity, we assume that there is
            # only one parallel region B in a given parent range, but this restriction
            # could be relaxed if needed.
            child_df = profile_df[make_child_mask(profile_df, launcher_row.Index)]
            is_main = child_df["TID"] == launcher_row.TID
            child_ends = child_df["StartMs"] + child_df["DurMs"]
            # Assuming there's only one parallel region inside `launcher_row`
            parallel_start = child_df.loc[~is_main, "StartMs"].min()
            parallel_end = child_ends[~is_main].max()
            # Assert that there are no main-thread tasks during this period
            main_before = is_main & (child_ends < parallel_start)
            main_after = is_main & (child_df["StartMs"] > parallel_end)
            assert ((main_before | main_after) == is_main).all()
            # Aggregate statistics for how the worker threads spend their time and use that
            # distribution to divide up the [parallel_start, parallel_end] range of the overall
            # compilation time.
            parallel_dur = parallel_end - parallel_start
            total_worker_time = child_df.loc[~is_main, "DurNonChildMs"].sum()

            def attribute_parallel_time(row):
                compile_time_ms[row.Name] += (
                    parallel_dur * row.DurNonChildMs / total_worker_time,
                    parallel_dur * row.DurChildMs / total_worker_time,
                )

            child_df[~is_main].apply(attribute_parallel_time, axis="columns")
            # Easy to update these given the simplifying assumptions above; they are set to
            # np.nan when worker ranges are spliced in by `_load_nvtx_pushpop_trace`
            compile_df.loc[launcher_row.Index, "DurChildMs"] = (
                child_df.loc[is_main, "DurMs"].sum() + parallel_dur
            )
            compile_df.loc[launcher_row.Index, "DurNonChildMs"] = (
                launcher_row.DurMs - compile_df.loc[launcher_row.Index, "DurChildMs"]
            )

        # `compile_time_ms` now accounts for parallel compilation worker threads, but not
        # the work from the main thread. Add that too.
        for row in compile_df[compile_df["TID"] == main_thread].itertuples():
            compile_time_ms[row.Name] += (row.DurNonChildMs, row.DurChildMs)

    return pd.DataFrame.from_dict(
        compile_time_ms, columns=["DurNonChildMs", "DurChildMs"], orient="index"
    ).sort_values(by=["DurNonChildMs"], ascending=False)
