import json
import logging
import pathlib
import typing
from .logic import TestResult


def add_summary_record(
    output_prefix: pathlib.Path,
    section: str,
    record: typing.Union[typing.Dict[str, typing.Any], TestResult],
    scalar=False,
):
    """
    Add a record to the output JSON file. This is intended to provide a useful record
    even in case of a fatal error.

    Args:
        output_prefix (pathlib.Path): The prefix for the output directory.
        section (str): The section of the summary to which the record belongs.
        record (dict or TestResult): The record to be added, either as a dictionary or
            as a TestResult object.
        scalar (bool): If True, the record is a scalar value; if False, it is a list of
            records. Defaults to False.

    Returns:
        None
    """
    summary_filename = output_prefix / "summary.json"
    try:
        with open(summary_filename, "r") as ifile:
            data = json.load(ifile)
    except FileNotFoundError:
        data = {}
    if scalar:
        if section in data:
            logging.warning(f"Overwriting summary data in section {section}")
        data[section] = record
    else:
        if section not in data:
            data[section] = []
        data[section].append(record)

    with open(summary_filename, "w") as ofile:
        json.dump(data, ofile)


def create_output_symlinks(
    output_prefix: pathlib.Path,
    last_known_good: typing.Optional[TestResult],
    first_known_bad: typing.Optional[TestResult],
):
    """
    Create symlinks to the last-good and first-bad output directories.
    versions.

    Args:
        output_prefix (pathlib.Path): The prefix for the output directory.
        last_known_good (TestResult): The last known good test result.
        first_known_bad (TestResult): The first known bad test result.

    Returns:
        None
    """

    def symlink(result: typing.Optional[TestResult], symlink_name: str) -> None:
        if result is None:
            return
        symlink_path = (output_prefix / symlink_name).resolve()
        assert not symlink_path.exists(), symlink_path
        assert symlink_path.parent == result.host_output_directory.parent, (
            symlink_path,
            result.host_output_directory,
        )
        symlink_path.symlink_to(result.host_output_directory)

    symlink(last_known_good, "last-known-good")
    symlink(first_known_bad, "first-known-bad")
