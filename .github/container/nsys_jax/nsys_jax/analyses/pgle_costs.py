#!/usr/bin/env python
import argparse
from nsys_jax import (
    apply_warmup_heuristics,
    ensure_compiled_protos_are_importable,
    load_profiler_data,
    xla_module_metadata,
)
import pathlib


def write_pbtxt(outdir: pathlib.Path, series_ms, hlo_module):
    mod_proto = hlo_module.proto().hlo_module
    fingerprint = mod_proto.frontend_attributes.map["fingerprint_before_lhs"]
    outdir.mkdir(exist_ok=True)
    fp_fname = f"{fingerprint}.pbtxt"
    null_names = 0
    with open(outdir / fp_fname, "w") as ofile:
        for name, cost_ms in series_ms.items():
            comp, inst = hlo_module.find_instruction(name)
            scheduling_name = inst.proto().metadata.scheduling_name
            null_names += len(scheduling_name) == 0
            ofile.write(
                f'costs {{ name: "{scheduling_name}" cost_us: {cost_ms * 1000:.1f} }}\n'
            )
    if null_names:
        print(f"Got {null_names} empty scheduling names")


def min_compute_max_comm(grouped):
    tmp_df = grouped.agg({"ProjDurFullMs": ("min", "max"), "Communication": "first"})
    result = tmp_df[("ProjDurFullMs", "max")].where(
        tmp_df[("Communication", "first")],
        tmp_df[("ProjDurFullMs", "min")],
    )
    result.name = "ProjDurFullMs"  # not max anymore
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Write something for --xla_gpu_pgle_profile_file_or_directory_path"
    )
    parser.add_argument("prefix", type=pathlib.Path)
    args = parser.parse_args()

    # Make sure that the .proto files under protos/ have been compiled to .py, and
    # that those generated .py files are importable.
    ensure_compiled_protos_are_importable(prefix=args.prefix)
    # Load the profiler data
    all_data = load_profiler_data(args.prefix)
    # Partition the profile data into initialisation and steady-state running
    _, steady_state = apply_warmup_heuristics(all_data)
    assert steady_state.module is not None
    module_ranking = (
        steady_state.module.loc[:, ("Name", "ProjDurMs")]
        .groupby("ProgramId")
        .agg({"Name": "first", "ProjDurMs": "sum"})
        .sort_values(ascending=False, by="ProjDurMs")
    )
    for row in module_ranking.itertuples():
        print(f"Processing module {row.Name} ({row.Index})")
        try:
            hlo_module = xla_module_metadata(row.Index, prefix=args.prefix)
        except Exception as e:
            print(f"Skipping due to: {e}")
            continue
        # Thunks in this module
        thunk_df = steady_state.thunk.loc[row.Index]
        # Re-add the overlapped part of communication thunks
        thunk_df["ProjDurFullMs"] = thunk_df["ProjDurMs"] + thunk_df["ProjDurHiddenMs"]
        # Slowest-seen communication kernel time, fastest-seen compute kernel time.
        write_pbtxt(
            pathlib.Path("./maxcomm_mincompute"),
            min_compute_max_comm(thunk_df.groupby("Name")),
            hlo_module,
        )


if __name__ == "__main__":
    main()
