import json
import logging
import pathlib
import typing
from .logic import (
    FlatVersionDict,
    TestExecutionOutcome,
    TestResult,
    version_cache_key,
)


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


def _parse_result(value) -> TestExecutionOutcome:
    if isinstance(value, TestExecutionOutcome):
        return value
    if isinstance(value, bool):
        return (
            TestExecutionOutcome.TEST_SUCCESS
            if value
            else TestExecutionOutcome.TEST_FAILURE
        )
    if isinstance(value, str):
        name = value.rsplit(".", 1)[-1]
        return TestExecutionOutcome[name]
    raise ValueError(f"Cannot parse test result from {value!r}")


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


def version_result_cache_from_summary(
    output_prefix: pathlib.Path,
    packages: typing.Iterable[str],
    summary: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> typing.Dict[FlatVersionDict, TestResult]:
    """
    Reconstruct completed version-level build/test results from summary.json.

    The summary file is treated as the transaction log. Output directories that exist
    without a corresponding summary record are ignored by construction.
    """
    if summary is None:
        summary = load_summary(output_prefix)
    packages = set(packages)
    cache = {}
    for record in summary.get("versions", []):
        if not isinstance(record, dict):
            continue
        if not packages <= record.keys():
            logging.warning(
                "Ignoring restart summary record that is missing package keys: %s",
                sorted(packages - record.keys()),
            )
            continue
        if "result" not in record or "output_directory" not in record:
            logging.warning("Ignoring incomplete restart summary record: %s", record)
            continue
        versions = {package: record[package] for package in packages}
        repetition = int(record.get("test_repetition", 0))
        key = version_cache_key(versions, repetition=repetition)
        cache[key] = TestResult(
            build_stdouterr=None,
            host_output_directory=_record_output_directory(output_prefix, record),
            result=_parse_result(record["result"]),
            stdouterr=None,
        )
    return cache


def container_result_cache_from_summary(
    output_prefix: pathlib.Path,
    summary: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> typing.Dict[str, TestResult]:
    """
    Reconstruct completed container-level test results from summary.json.
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
        cache[record["container"]] = TestResult(
            build_stdouterr=None,
            host_output_directory=_record_output_directory(output_prefix, record),
            result=_parse_result(record["result"]),
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
        symlink_path = output_prefix / symlink_name
        if symlink_path.exists() or symlink_path.is_symlink():
            assert symlink_path.resolve() == result.host_output_directory.resolve(), (
                symlink_path,
                result.host_output_directory,
            )
            return
        absolute_symlink_path = symlink_path.resolve()
        assert absolute_symlink_path.parent == result.host_output_directory.parent, (
            absolute_symlink_path,
            result.host_output_directory,
        )
        absolute_symlink_path.symlink_to(result.host_output_directory)

    symlink(last_known_good, "last-known-good")
    symlink(first_known_bad, "first-known-bad")
