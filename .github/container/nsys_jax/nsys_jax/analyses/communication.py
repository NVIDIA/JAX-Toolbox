#!/usr/bin/env python
import argparse
import csv
import pathlib
from collections import defaultdict
from math import sqrt

from nsys_jax import (
    align_profiler_data_timestamps,
    apply_warmup_heuristics,
    ensure_compiled_protos_are_importable,
    load_profiler_data,
)
from prettytable import PrettyTable
from uncertainties import ufloat  # type: ignore



def process_communication_data(steady_state):
    """
    Process communication data from a steady state, to compute bandwith summaries.

    Args:
        steady_state: A steady state data frame.

    Return:
        A tuple of (collective_types, summary_data), where:
            collective_types (List[str]): sorted list of collective operation types
            summary_data (Dict[int, Dict[str, ufloat]]): Dictionary wiht summaries for bandwith data
    """
    collective_types = set()
    summary_data = defaultdict(dict)

    communication_grouped_by = steady_state.communication.groupby(
        ["Collective", "MessageSize"]
    )

    for (collective, message_size), df in communication_grouped_by:
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
        bandwidth_of_fastest_device = devices["BusBandwidthGBPerSec"].agg("max")
        mean_bandwidth = bandwidth_of_fastest_device.mean()
        stderr_bandwidth = bandwidth_of_fastest_device.std() / sqrt(
            len(bandwidth_of_fastest_device)
        )

        summary_data[message_size][collective] = ufloat(
            mean_bandwidth, stderr_bandwidth
        )

    return sorted(collective_types), summary_data


def print_bandwidth_table(collective_types, summary_data):
    """
    This function prints a table for summarizing the bandwidth for each collective operation

    Args:
        collective_types (List[str]): sorted list of collective operation types
        summary_data (Dict[int, Dict[str, ufloat]]): Dictionary wiht summaries for bandwith data
    """

    def format_message_size(message_size):
        """
        Function to format the message size
        """
        return f"{message_size:<{size_width},}"

    def format_bandwidth(data, collective):
        """
        Function to format the bandwidth
        """
        width = collective_widths[collective]
        if collective not in data:
            return "-" * width
        return f"{data[collective]:>{width}S}"

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

    header_log = f"{'':<{size_width}} | Bus bandwidth [GB/s]"
    print(header_log)
    log_specs = " | ".join(
        [f"{size_heading:<{size_width}}"]
        + [f"{coll:<{collective_widths[coll]}}" for coll in collective_types]
    )
    print(log_specs)

    for message_size in sorted(summary_data.keys()):
        data = summary_data[message_size]
        log_row = " | ".join(
            [format_message_size(message_size)]
            + [format_bandwidth(data, collective) for collective in collective_types]
        )
        print(log_row)


def process_hidden_ms_to_total_ms(steady_state):
    """
    Function to compute the fraction of communication time that is hidden behind computations.

    Args:
        steady_state: The steady state data

    Returns:
        collective_types (Set[str]): set of collective operation types
        summary_data (Dict[str, float]): dictionary with mean hidden-to-total milliseconds ratio
    """
    if steady_state.communication["ProjDurHiddenMs"].sum() == 0:
        return None, None

    collective_types = set()
    summary_data = defaultdict(dict)
    grouped_data = steady_state.communication.groupby(["Collective"])

    for collective, df in grouped_data:
        collective_types.add(collective)
        total_ms = df["ProjDurMs"] + df["ProjDurHiddenMs"]
        mean_dur_hidden_ms_to_total_ms = (df["ProjDurHiddenMs"] / total_ms).mean()
        summary_data[collective] = mean_dur_hidden_ms_to_total_ms

    return collective_types, summary_data


def print_hidden_ms_to_total_ms_table(
    collective_types, summary_data, overall_hidden_ms_to_total_ms
):
    """
    Print the hidden ms to total ms

    Args:
        collective_types (Set[str]): set of collective operation types
        summary_data (Dict[str, float]): mean hidden-to-total milliseconds ratio
        overall_ratio (float): overall hidden-to-total milliseconds ratio
    """
    table = PrettyTable()
    table.field_names = ["Collective", "Mean HiddenToTotalMs"]

    for collective in collective_types:
        mean_value = summary_data[collective]
        table.add_row([collective[0], mean_value])

    print(table)
    if overall_hidden_ms_to_total_ms is not None:
        print(
            f"Overall HiddenMs to TotalMs: {overall_hidden_ms_to_total_ms:.4f}"
        )


def calculate_overall_hidden_ms_to_total_ms(steady_state):
    """
    Function to calculate the overall hidden milliseconds to total milliseconds

    Args:
      steady_state: the steady-state data extracted from the profiler

    Returns:
      overall_hidden_ms_to_total_ms (float): overall hidden milliseconds to total milliseconds ratio
    """
    total_hidden_ms = steady_state.communication["ProjDurHiddenMs"].sum()
    if total_hidden_ms == 0:
        return None

    total_ms = (
        steady_state.communication["ProjDurMs"]
        + steady_state.communication["ProjDurHiddenMs"]
    ).sum()

    overall_hidden_ms_to_total_ms = total_hidden_ms / total_ms

    return overall_hidden_ms_to_total_ms


def write_to_csv(
    collective_types,
    bandwidth_summary,
    hidden_to_total_summary,
    overall_hidden_ms_to_total_ms,
    output_file,
):
    """
    Function to write the summaries to a csv file

    Args:
        collective_types (List[str]): list of collective operation types
        bandwidth_summary (Dict[int, Dict[str, ufloat]]): bandwidth summary data
        hidden_to_total_summary (Dict[str, float]): hidden-to-total milliseconds ratio summary
        overall_hidden_ms_to_total_ms (float): overall hidden-to-total milliseconds ratio
        output_file (str): output CSV file path

    """
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
    """
    Main entry point to process the nsys-jax report and generate communication summaries
    """
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
    print(f"Alignment metadata: {alignment_metadata}")
    # Partition the profile data into initialisation and steady-state running
    _, steady_state = apply_warmup_heuristics(all_data)

    if len(steady_state.communication) == 0:
        print(
            "Communication summary was requested but no steady-state communication was identified."
        )
        return

    collective_types, bandwidth_summary = process_communication_data(steady_state)
    print_bandwidth_table(collective_types, bandwidth_summary)

    hidden_to_total_collective_types, hidden_to_total_summary = (
        process_hidden_ms_to_total_ms(steady_state)
    )

    # initailise overall_hidden_ms_to_total_ms
    overall_hidden_ms_to_total_ms = None

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
