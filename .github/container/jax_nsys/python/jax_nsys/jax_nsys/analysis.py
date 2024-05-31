from collections import defaultdict
import functools
import math
import numpy as np
import pandas as pd  # type: ignore

from .protobuf import xla_module_metadata
from .utils import make_child_mask

pd.options.mode.copy_on_write = True


def element_type_in_bits(element_type: int) -> int:
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

    # S32 is a 32-bit type and so on.
    # FIXME: does not handle FP8 yet
    for prefix in ["BF", "C", "F", "S", "U"]:
        if enum_name.startswith(prefix):
            return int(enum_name[len(prefix) :])
    raise Exception(f"Could not deduce size of {enum_name}")


@functools.lru_cache
def get_message_size(program_id: int, instruction: str) -> int:
    """
    Given the name of a collective instruction (e.g. all-gather-start.N), calculate the
    message size in bytes. See https://openxla.org/xla/operation_semantics#allgather,
    https://openxla.org/xla/operation_semantics#allreduce and so on for more explanation
    of the semantics. This implementation aims to follow the same conventions that NCCL
    uses in its NVTX payloads and tests.
    """
    module_proto = xla_module_metadata(program_id)
    _, inst = module_proto.find_instruction(instruction)
    assert (
        inst.opcode
        in {
            "all-gather-start",
            "all-reduce-start",
            "collective-broadcast",
            "collective-permute-start",
            "reduce-scatter",
        }
    ), f"{instruction}: message size calculation for {inst.opcode} has not yet been validated"
    if inst.opcode == "collective-permute-start":
        # See https://openxla.org/xla/operation_semantics#collectivepermute, which
        # generates pair-wise send+recv between devices
        collective_size = 2
    else:
        # replica_groups is something like {{0,1},{4,5},{2,3},{6,7}}, if there are 8
        # devices that are doing pair-wise collectives
        collective_sizes = tuple(
            {len(group.replica_ids) for group in inst.replica_groups}
        )
        assert (
            len(collective_sizes) == 1
        ), f"Heterogeneous collective {inst.replica_groups} could not be interpreted"
        collective_size = collective_sizes[0]
    total_msg_size = 0
    for operand_id in inst.operand_ids:
        _, operand = module_proto.find_instruction_by_id(operand_id)
        msg_size_bits = math.prod(
            operand.shape.dimensions,
            start=element_type_in_bits(operand.shape.element_type),
        )
        if inst.opcode == "reduce-scatter":
            # NCCL's convention is that the message size of a reduce-scatter is the size of output buffer:
            # https://github.com/NVIDIA/nccl/blob/ab2b89c4c339bd7f816fbc114a4b05d386b66290/src/collectives.cc#L122
            assert msg_size_bits % collective_size == 0
            msg_size_bits //= collective_size
        assert msg_size_bits % 8 == 0
        total_msg_size += msg_size_bits // 8

    # Calculate the correction factor from algorithm bandwidth to bus bandwidth, see:
    # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bus-bandwidth
    collective = inst.opcode.removesuffix("-start")
    bw_correction, bus_correction = {
        # For AllGather the size in the bandwidth calculation is the total/output size
        # https://github.com/NVIDIA/nccl-tests/blob/c6afef0b6f76ffc55d4172d971be6cf5a08a73a4/doc/PERFORMANCE.md#allgather
        "all-gather": (collective_size, (collective_size - 1) / collective_size),
        "all-reduce": (1, 2 * (collective_size - 1) / collective_size),
        "collective-broadcast": (1, 1),
        "collective-permute": (1, 1),
        # For ReduceScatter the size in the bandwidth calculation is the total size
        # https://github.com/NVIDIA/nccl-tests/blob/c6afef0b6f76ffc55d4172d971be6cf5a08a73a4/doc/PERFORMANCE.md#reducescatter
        "reduce-scatter": (collective_size, (collective_size - 1) / collective_size),
    }[collective]
    return pd.Series(
        [total_msg_size, collective, collective_size, bw_correction, bus_correction],
        index=[
            "MessageSize",
            "Collective",
            "CollectiveSize",
            "BandwidthCorrection",
            "BusBandwidthCorrection",
        ],
    )


def calculate_collective_metrics(thunk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a "thunk" data frame from `load_profiler_data`, produce a new data frame that
    contains one row per communication thunk and contains extra metrics such as the
    message size, algorithm bandwidth, bus bandwidth, and collective operation.
    """
    comm_df = thunk_df[thunk_df["Communication"]].drop(columns=["Communication"])
    comm_df = pd.concat(
        [
            comm_df,
            comm_df.apply(
                lambda row: get_message_size(row.ProgramId, row.Name), axis=1
            ),
        ],
        axis=1,
    )
    # Note that this is decimal GB not binary GiB; GB/s == B/ns
    comm_df["AlgorithmBandwidthGBPerSec"] = (
        comm_df["BandwidthCorrection"]
        * comm_df["MessageSize"]
        / (comm_df["ProjDurNs"] + comm_df["ProjDurHiddenNs"])
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
    # Identify the main thread
    main_thread = compile_df.loc[compile_df["Name"] == "XlaCompile", "TID"].unique()
    assert len(main_thread) == 1
    main_thread = main_thread[0]

    # Aggregate compilation stats in here
    compile_time_ns: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2))

    # Identify the ranges in the main thread that represent parallel compilation, i.e.
    # ranges whose child ranges are in different threads.
    worker_parent_ids = compile_df.loc[
        compile_df["TID"].ne(main_thread), "ParentId"
    ].astype(np.int32)
    # These are the main-thread ranges that directly contain parallel workers
    launcher_mask = compile_df.loc[worker_parent_ids, "TID"].eq(main_thread)
    launcher_ids = launcher_mask[launcher_mask].index.unique()
    # Loop over the main-thread ranges that launched parallel work
    for launcher_row in compile_df.loc[launcher_ids, :].itertuples():
        assert launcher_row.TID == main_thread
        # Find all child ranges; some may still be in the main thread. Assume for now
        # that the sequence will just be something like:
        #   M(A) M(A) M(A) .. W1(B) W2(B) W3(B) W1(B) .. M(C) M(C)
        # i.e. the main thread M does some task (A), then workers W{1,2,3} do some task
        # (B) in parallel, then the main thread continues with another task (C),
        # without overlap between A, B and C. For simplicity, we assume that there is
        # only one parallel region B in a given parent range, but this restriction
        # could be relaxed if needed.
        child_df = compile_df[make_child_mask(compile_df, launcher_row.Index)]
        is_main = child_df["TID"] == launcher_row.TID
        child_ends = child_df["StartNs"] + child_df["DurNs"]
        # Assuming there's only one parallel region inside `launcher_row`
        parallel_start = child_df.loc[~is_main, "StartNs"].min()
        parallel_end = child_ends[~is_main].max()
        # Assert that there are no main-thread tasks during this period
        main_before = is_main & (child_ends < parallel_start)
        main_after = is_main & (child_df["StartNs"] > parallel_end)
        assert ((main_before | main_after) == is_main).all()
        # Aggregate statistics for how the worker threads spend their time and use that
        # distribution to divide up the [parallel_start, parallel_end] range of the overall
        # compilation time.
        parallel_dur = parallel_end - parallel_start
        total_worker_time = child_df.loc[~is_main, "DurNonChildNs"].sum()

        def attribute_parallel_time(row):
            compile_time_ns[row.Name] += (
                parallel_dur * row.DurNonChildNs / total_worker_time,
                parallel_dur * row.DurChildNs / total_worker_time,
            )

        child_df[~is_main].apply(attribute_parallel_time, axis="columns")
        # Easy to update these given the simplifying assumptions above; they are set to
        # np.nan when worker ranges are spliced in by `_load_nvtx_pushpop_trace`
        compile_df.loc[launcher_row.Index, "DurChildNs"] = (
            child_df.loc[is_main, "DurNs"].sum() + parallel_dur
        )
        compile_df.loc[launcher_row.Index, "DurNonChildNs"] = (
            launcher_row.DurNs - compile_df.loc[launcher_row.Index, "DurChildNs"]
        )

    # `compile_time_ns` now accounts for parallel compilation worker threads, but not
    # the work from the main thread. Add that too.
    for row in compile_df[compile_df["TID"] == main_thread].itertuples():
        compile_time_ns[row.Name] += (row.DurNonChildNs, row.DurChildNs)

    return pd.DataFrame.from_dict(
        compile_time_ns, columns=["DurNonChildNs", "DurChildNs"], orient="index"
    ).sort_values(by=["DurNonChildNs"], ascending=False)
