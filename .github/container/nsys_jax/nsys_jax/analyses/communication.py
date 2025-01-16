#!/usr/bin/env python
import argparse
import csv
from collections import defaultdict

from nsys_jax import (
    align_profiler_data_timestamps,
    apply_warmup_heuristics,
    ensure_compiled_protos_are_importable,
    load_profiler_data,
)
from math import sqrt
from prettytable import PrettyTable
import pathlib
from uncertainties import ufloat  # type: ignore


def process_communication_data(steady_state):
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
    return sorted(collective_types), summary_data


def print_bandwidth_table(collective_types, summary_data):
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


def process_hidden_ms_to_total_ms(steady_state):
    if steady_state.communication["ProjDurHiddenMs"].sum() == 0:
        return None, None

    collective_types = set()
    summary_data = defaultdict(dict)
    for collective, df in steady_state.communication.groupby(["Collective"]):
        collective_types.add(collective)
        mean_dur_hidden_ms_to_total_ms = (
            df["ProjDurHiddenMs"] / (df["ProjDurMs"] + df["ProjDurHiddenMs"])
        ).mean()
        summary_data[collective] = mean_dur_hidden_ms_to_total_ms
    return collective_types, summary_data


def print_hidden_ms_to_total_ms_table(
    collective_types, summary_data, overall_hidden_ms_to_total_ms
):
    table = PrettyTable()
    table.field_names = ["Collective", "Mean HiddenToTotalMs"]

    for collective in collective_types:
        mean_value = summary_data[collective]
        table.add_row([collective[0], mean_value])

    print(table)
    print("Overall HiddenMs to TotalMs:", overall_hidden_ms_to_total_ms)


def calculate_overall_hidden_ms_to_total_ms(steady_state):
    if steady_state.communication["ProjDurHiddenMs"].sum() == 0:
        return None

    overall_hidden_ms_to_total_ms = (
        steady_state.communication["ProjDurHiddenMs"].sum()
        / (
            steady_state.communication["ProjDurMs"]
            + steady_state.communication["ProjDurHiddenMs"]
        ).sum()
    )
    return overall_hidden_ms_to_total_ms


def write_to_csv(
    collective_types,
    bandwidth_summary,
    hidden_to_total_summary,
    overall_hidden_ms_to_total_ms,
    output_file,
):
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write bandwidth table
        writer.writerow(["Bandwidth Table"])
        writer.writerow(["Size [B]"] + list(collective_types))
        for message_size in sorted(bandwidth_summary.keys()):
            row = [message_size]
            for collective in collective_types:
                if collective in bandwidth_summary[message_size]:
                    row.append(f"{bandwidth_summary[message_size][collective]:S}")
                else:
                    row.append("-")
            writer.writerow(row)

        writer.writerow([])  # Empty row for separation

        # Write hidden to total table if data is available
        if hidden_to_total_summary is not None:
            writer.writerow(["HiddenMs to TotalMs Table"])
            writer.writerow(["Collective", "Mean HiddenToTotalMs"])
            for collective in hidden_to_total_summary:
                writer.writerow([collective[0], hidden_to_total_summary[collective]])

            writer.writerow([])  # Empty row for separation

            if overall_hidden_ms_to_total_ms is not None:
                writer.writerow(
                    ["Overall HiddenMs to TotalMs", overall_hidden_ms_to_total_ms]
                )


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

    collective_types, bandwidth_summary = process_communication_data(steady_state)
    print_bandwidth_table(collective_types, bandwidth_summary)

    hidden_to_total_collective_types, hidden_to_total_summary = (
        process_hidden_ms_to_total_ms(steady_state)
    )
    if hidden_to_total_summary is not None:
        overall_hidden_ms_to_total_ms = calculate_overall_hidden_ms_to_total_ms(
            steady_state
        )
        print_hidden_ms_to_total_ms_table(
            hidden_to_total_collective_types,
            hidden_to_total_summary,
            overall_hidden_ms_to_total_ms,
        )

    # Write all tables to a single CSV file
    write_to_csv(
        collective_types,
        bandwidth_summary,
        hidden_to_total_summary,
        overall_hidden_ms_to_total_ms,
        "communication_summary.csv",
    )


if __name__ == "__main__":
    main()
