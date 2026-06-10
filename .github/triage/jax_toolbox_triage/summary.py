import json
import logging
import pathlib
import typing
from .logic import (
    TestExecutionOutcome,
    TestResult,
    version_cache_key,
)

SummaryCacheKey = typing.Tuple[str, typing.Any]
CONTAINER_CACHE_SECTION = "container"
VERSION_CACHE_SECTION = "versions"
VERSION_RECORD_METADATA = {
    "build_time",
    "container",
    "output_directory",
    "result",
    "test_repetition",
    "test_time",
}


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
    return data


def load_summary(output_prefix: pathlib.Path) -> typing.Dict[str, typing.Any]:
    """
    Load the JSON summary for a previous or current triage run.
    """
    with open(output_prefix / "summary.json", "r") as ifile:
        return json.load(ifile)


def _record_output_directory(
    output_prefix: pathlib.Path, record: typing.Dict[str, typing.Any]
) -> pathlib.Path:
    out_dir = pathlib.Path(record["output_directory"])
    if out_dir.exists() or not out_dir.is_absolute():
        return out_dir
    copied_dir = output_prefix / out_dir.name
    if copied_dir.exists():
        return copied_dir.resolve()
    return out_dir


def result_cache_from_summary(
    output_prefix: pathlib.Path,
    summary: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> typing.Dict[SummaryCacheKey, TestResult]:
    """
    Reconstruct completed build/test results from summary.json.

    The summary file is treated as the transaction log. Output directories that exist
    without a corresponding summary record are ignored by construction.
    """
    if summary is None:
        summary = load_summary(output_prefix)
    cache = {}
    for record in summary.get("container", []):
        if not isinstance(record, dict):
            continue
        if not {"container", "result", "output_directory"} <= record.keys():
            logging.warning("Ignoring incomplete restart container record: %s", record)
            continue
        result = (
            TestExecutionOutcome.TEST_SUCCESS
            if record["result"]
            else TestExecutionOutcome.TEST_FAILURE
        )
        cache[(CONTAINER_CACHE_SECTION, record["container"])] = TestResult(
            build_stdouterr=None,
            host_output_directory=_record_output_directory(output_prefix, record),
            result=result,
            stdouterr=None,
        )

    for record in summary.get("versions", []):
        if not isinstance(record, dict):
            continue
        if "result" not in record or "output_directory" not in record:
            logging.warning("Ignoring incomplete restart summary record: %s", record)
            continue
        versions = {
            key: value
            for key, value in record.items()
            if key not in VERSION_RECORD_METADATA
        }
        if not versions:
            logging.warning(
                "Ignoring restart summary record that is missing package keys: %s",
                record,
            )
            continue
        repetition = int(record.get("test_repetition", 0))
        key = version_cache_key(versions, repetition=repetition)
        result_name = record["result"].rsplit(".", 1)[-1]
        cache[(VERSION_CACHE_SECTION, key)] = TestResult(
            build_stdouterr=None,
            host_output_directory=_record_output_directory(output_prefix, record),
            result=TestExecutionOutcome[result_name],
            stdouterr=None,
        )
    return cache


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
