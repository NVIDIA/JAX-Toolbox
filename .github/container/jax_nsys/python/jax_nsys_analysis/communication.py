#!/usr/bin/env python
import argparse
from collections import defaultdict
from jax_nsys import (
    align_profiler_data_timestamps,
    apply_warmup_heuristics,
    ensure_compiled_protos_are_importable,
    load_profiler_data,
)
from math import sqrt
import pathlib
from uncertainties import ufloat  # type: ignore


def main():
    parser = argparse.ArgumentParser(
        description="Summarise communication in an nsys-jax report"
    )
    parser.add_argument("prefix", type=pathlib.Path)
    args = parser.parse_args()
    # Make sure that the .proto files under protos/ have been compiled to .py, and
    # that those generated .py files are importable.
    ensure_compiled_protos_are_importable(prefix=args.prefix)
    # Load the profiler data; the compilation part is needed for the warmup heuristics
    all_data = load_profiler_data(args.prefix, frames={"communication", "compile"})
    # Align timestamps
    all_data, alignment_metadata = align_profiler_data_timestamps(all_data)
    # TODO: make this pretty
    # print(alignment_metadata)
    # Partition the profile data into initialisation and steady-state running
    _, steady_state = apply_warmup_heuristics(all_data)
    assert len(steady_state.communication), (
        "Communication summary was requested but no steady-state communication was "
        "identified."
    )
    collective_types = set()
    summary_data = defaultdict(dict)
    for (collective, message_size), df in steady_state.communication.groupby(
        ["Collective", "MessageSize"]
    ):
        collective_types.add(collective)
        # This grouped data frame will have a row for each device that is participating
        # in this instance of the collective.
        devices = df.groupby(["ProgramId", "ProgramExecution", "ThunkIndex"])
        # Take the fastest device bandwidth. Rationale: the slower devices appear
        # slower because they spend some time waiting for the last device, and then all
        # devices complete the collective at the same time. The fastest device is
        # therefore the last one to join the collective and its bandwidth estimate does
        # not contain a wait time component. The .mean() is over the different
        # (ProgramId, ProgramExecution, ThunkIndex) values.
        bandwidth = devices["BusBandwidthGBPerSec"].agg("max")
        summary_data[message_size][collective] = ufloat(
            bandwidth.mean(), bandwidth.std() / sqrt(len(bandwidth))
        )
    collective_types = sorted(collective_types)
    collective_widths = {
        collective: max(
            len(collective),
            max(
                len(f"{data[collective]:S}")
                for data in summary_data.values()
                if collective in data
            ),
        )
        for collective in collective_types
    }
    size_heading = "Size [B]"
    size_width = max(len(size_heading), max(len(f"{s:,}") for s in summary_data.keys()))
    print(f"{'':<{size_width}} | Bus bandwidth [GB/s]")
    print(
        " | ".join(
            [f"{size_heading:<{size_width}}"]
            + [f"{coll:<{collective_widths[coll]}}" for coll in collective_types]
        )
    )

    def format_message_size(message_size):
        return f"{message_size:<{size_width},}"

    def format_bandwidth(data, collective):
        width = collective_widths[collective]
        if collective not in data:
            return "-" * width
        return f"{data[collective]:>{width}S}"

    for message_size in sorted(summary_data.keys()):
        data = summary_data[message_size]
        print(
            " | ".join(
                [format_message_size(message_size)]
                + [
                    format_bandwidth(data, collective)
                    for collective in collective_types
                ]
            )
        )


if __name__ == "__main__":
    main()
