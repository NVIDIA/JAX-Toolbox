#!/usr/bin/env python
import argparse
import math
from nsys_jax import (
    apply_warmup_heuristics,
    ensure_compiled_protos_are_importable,
    generate_compilation_statistics,
    load_profiler_data,
    remove_autotuning_detail,
)
import pathlib
from prettytable import PrettyTable
from uncertainties import ufloat  # type: ignore


def main():
    parser = argparse.ArgumentParser(
        description="Print summary statistics from an nsys-jax report"
    )
    parser.add_argument("prefix", type=pathlib.Path)
    args = parser.parse_args()

    # Make sure that the .proto files under protos/ have been compiled to .py, and
    # that those generated .py files are importable.
    ensure_compiled_protos_are_importable(prefix=args.prefix)
    # Load the profiler data
    all_data = load_profiler_data(args.prefix)
    # Remove some detail from the autotuner
    all_data = remove_autotuning_detail(all_data)
    # Partition the profile data into initialisation and steady-state running
    init, steady_state = apply_warmup_heuristics(all_data)
    # Get high-level statistics about the modules that were profiled
    assert steady_state.module is not None
    module_stats = (
        steady_state.module.groupby("ProgramId")
        .agg(
            {
                "Name": ("first", "count"),
                "NumThunks": ("mean", "std"),
                "ProjDurMs": ("sum", "mean", "std"),
            }
        )
        .sort_values(("ProjDurMs", "sum"), ascending=False)
    )
    module_stats[("ProjDurMs", "percent")] = (
        100
        * module_stats[("ProjDurMs", "sum")]
        / module_stats[("ProjDurMs", "sum")].sum()
    )

    have_comms = steady_state.communication is not None and len(
        steady_state.communication
    )
    if have_comms:
        # Calculate the time spent waiting in collectives for each module.
        # Min/max over devices within individual communication thunk executions
        min_max_device_times = (
            steady_state.communication["ProjDurMs"]
            .groupby(["ProgramId", "ProgramExecution", "Name", "ThunkExecution"])
            .agg(("min", "max"))
        )
        # Define wait time as max-min *exposed* communication thunk times
        thunk_wait_times = min_max_device_times["max"] - min_max_device_times["min"]
        # Sum over thunks within each module
        module_wait_times = thunk_wait_times.groupby(
            ["ProgramId", "ProgramExecution"]
        ).agg("sum")
        # Stats over different executions of the module
        wait_averages = module_wait_times.groupby("ProgramId").agg(("mean", "std"))
        module_stats[("WaitMs", "mean")] = wait_averages["mean"]
        module_stats[("WaitMs", "std")] = wait_averages["std"]
        module_stats[("WaitMs", "percent")] = (
            100 * wait_averages["mean"] / module_stats[("ProjDurMs", "mean")]
        )

    def dump(fname, df):
        with open(fname + ".json", "w") as ofile:
            df.to_json(ofile, orient="split")

    dump("module-stats", module_stats)
    print(" === MODULE EXECUTION SUMMARY ===")
    fields = {
        "ID": lambda _, v: str(v),
        "Name": lambda _, v: v,
        "#execs": lambda _, v: str(v),
        "Thunks": lambda _, v: f"{v:S}" if v.s else f"{v.n:.0f}",
        "Duration [ms]": lambda _, v: f"{v:S}",
        "Duration [%]": lambda _, v: f"{v:.3f}",
    }
    if have_comms:
        fields["Wait time [ms]"] = lambda _, v: "---" if math.isnan(v.n) else f"{v:S}"
        fields["Wait time [%]"] = lambda _, v: "---" if math.isnan(v) else f"{v:.3f}"
    table = PrettyTable(align="r", custom_format=fields, field_names=fields.keys())
    for id, row in module_stats.iterrows():
        table.add_row(
            [
                id,
                row[("Name", "first")],
                row[("Name", "count")],
                ufloat(row[("NumThunks", "mean")], row[("NumThunks", "std")]),
                ufloat(row[("ProjDurMs", "mean")], row[("ProjDurMs", "std")]),
                row[("ProjDurMs", "percent")],
            ]
            + (
                [
                    ufloat(row[("WaitMs", "mean")], row[("WaitMs", "std")]),
                    row[("WaitMs", "percent")],
                ]
                if have_comms
                else []
            )
        )
    print(table)

    compilation_stats = generate_compilation_statistics(init.compile)
    if len(compilation_stats):
        total_compile_time = compilation_stats["DurNonChildMs"].sum()
        compilation_stats["DurNonChildPercent"] = (
            100 * compilation_stats["DurNonChildMs"] / total_compile_time
        )
        # Dump before dropping
        dump("compilation-ranges", compilation_stats)
        compilation_stats = compilation_stats.drop(columns=["DurChildMs"])
        top_n = 10
        top_n_ranges = compilation_stats.iloc[:top_n]

        # All XlaPass ranges combined into a single XlaPasses range, XlaPassPipeline ranges ignored
        def remove_xlapass_xlapasspipeline_detail(name):
            if name.startswith("XlaPass:#"):
                return "XlaPasses"
            elif name.startswith("XlaPassPipeline:#"):
                return "XlaPassPipelines"
            else:
                return name

        no_pass_detail = (
            compilation_stats.groupby(remove_xlapass_xlapasspipeline_detail)
            .agg("sum")
            .sort_values("DurNonChildMs", ascending=False)
        )
        dump("compilation-high-level", no_pass_detail)
        # Top few passes, with the percentages re-scaled to be relative to XlaPasses above
        pass_df = compilation_stats[
            compilation_stats.index.to_series().str.startswith("XlaPass:#")
        ]
        pass_df["DurNonChildPercent"] = (
            100 * pass_df["DurNonChildMs"] / pass_df["DurNonChildMs"].sum()
        )
        dump("compilation-passes", pass_df)
        print(f" === COMPILATION TIME -- TOP {top_n} RANGES ===\n{top_n_ranges}")
        print(f" === COMPILATION TIME -- NO PASS DETAIL ===\n{no_pass_detail}")
        print(
            f" === COMPILATION TIME -- TOP {top_n} XLA PASSES ===\n{pass_df.iloc[:top_n]}"
        )


if __name__ == "__main__":
    main()
