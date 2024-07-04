#!/usr/bin/env python
import argparse
from jax_nsys import (
    apply_warmup_heuristics,
    ensure_compiled_protos_are_importable,
    generate_compilation_statistics,
    load_profiler_data,
)
import pathlib
from pprint import pprint

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
# Ignore autotuning executions with ProgramId < 0
all_data.module = all_data.module.loc[0:]
all_data.thunk = all_data.thunk.loc[0:]
# Partition the profile data into initialisation and steady-state running
init, steady_state = apply_warmup_heuristics(all_data)
# Get high-level statistics about the modules that were profiled
module_stats = (
    steady_state.module.groupby("ProgramId")
    .agg(
        {
            "Name": ("first", "count"),
            "ProjDurMs": ("sum", "std"),
            "NumThunks": ("mean", "std"),
        }
    )
    .sort_values(("ProjDurMs", "sum"), ascending=False)
)
module_stats["ProjDurPercent"] = (
    100 * module_stats[("ProjDurMs", "sum")] / module_stats[("ProjDurMs", "sum")].sum()
)
print(module_stats)

print(generate_compilation_statistics(init.compile))
