#!/usr/bin/env python
import argparse
from nsys_jax import (
    apply_warmup_heuristics,
    ensure_compiled_protos_are_importable,
    generate_compilation_statistics,
    load_profiler_data,
    remove_autotuning_detail,
)
import pathlib


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

    def dump(fname, df):
        with open(fname + ".json", "w") as ofile:
            df.to_json(ofile, orient="split")

    dump("module-stats", module_stats)
    print(f" === MODULE EXECUTION SUMMARY ===\n{module_stats}")

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
