from collections import defaultdict
import numpy as np
import pandas as pd

from .utils import make_child_mask


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
    compile_time_ns = defaultdict(lambda: np.zeros(2))

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
