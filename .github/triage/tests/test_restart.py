import collections
import datetime
import json
import logging

import pytest

from jax_toolbox_triage.args import parse_args
from jax_toolbox_triage.logic import (
    TestExecutionOutcome,
    version_search,
)
from jax_toolbox_triage.summary import (
    CONTAINER_CACHE_SECTION,
    result_cache_from_summary,
    VERSION_CACHE_SECTION,
)
from jax_toolbox_triage.triage_tool import TriageTool


start_date = datetime.datetime(2026, 1, 1)


def make_commits():
    return collections.OrderedDict(
        [
            ("xla", [("xla-good", start_date)]),
            (
                "jax",
                [
                    ("jax-good", start_date),
                    ("jax-bad", start_date + datetime.timedelta(days=1)),
                ],
            ),
        ]
    )


def test_restart_folder_requires_summary_json(tmp_path):
    with pytest.raises(Exception, match="summary.json"):
        parse_args(
            [
                "--restart-folder",
                str(tmp_path),
                "--container-runtime=local",
                "--passing-versions",
                "jax:jax-good,xla:xla-good",
                "--failing-versions",
                "jax:jax-bad,xla:xla-good",
                "test-command",
            ]
        )


def test_restart_uses_restart_folder_as_output_prefix(tmp_path):
    (tmp_path / "summary.json").write_text("{}")

    args = parse_args(
        [
            "--restart-folder",
            str(tmp_path),
            "--container-runtime=local",
            "--passing-versions",
            "jax:jax-good,xla:xla-good",
            "--failing-versions",
            "jax:jax-bad,xla:xla-good",
            "test-command",
        ]
    )

    assert args.output_prefix == tmp_path.resolve()


def test_version_search_reuses_preloaded_restart_cache(tmp_path):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad"
    good_dir.mkdir()
    bad_dir.mkdir()
    summary = {
        "versions": [
            {
                "container": "container-url",
                "xla": "xla-good",
                "jax": "jax-good",
                "output_directory": str(good_dir),
                "result": "TestExecutionOutcome.TEST_SUCCESS",
            },
            {
                "container": "container-url",
                "xla": "xla-good",
                "jax": "jax-bad",
                "output_directory": str(bad_dir),
                "result": "TestExecutionOutcome.TEST_FAILURE",
            },
        ]
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    summary_cache = result_cache_from_summary(tmp_path, make_commits().keys())
    result_cache = {
        key: result
        for (section, key), result in summary_cache.items()
        if section == VERSION_CACHE_SECTION
    }

    def build_and_test(**kwargs):
        raise AssertionError(f"Unexpected build/test during restart: {kwargs}")

    result, last_known_good, first_known_bad = version_search(
        versions=make_commits(),
        build_and_test=build_and_test,
        logger=logging.getLogger("triage-restart-test"),
        skip_precondition_checks=False,
        confirmation_iterations=0,
        result_cache=result_cache,
    )

    assert result == {
        "jax_bad": "jax-bad",
        "jax_good": "jax-good",
        "xla_ref": "xla-good",
    }
    assert last_known_good.result == TestExecutionOutcome.TEST_SUCCESS
    assert first_known_bad.result == TestExecutionOutcome.TEST_FAILURE


def test_summary_cache_loads_container_records(tmp_path):
    out_dir = tmp_path / "container-output"
    out_dir.mkdir()
    summary = {
        "container": [
            {
                "container": "container-url",
                "output_directory": str(out_dir),
                "result": True,
            }
        ]
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))

    summary_cache = result_cache_from_summary(tmp_path)

    cached_result = summary_cache[(CONTAINER_CACHE_SECTION, "container-url")]
    assert cached_result.result == TestExecutionOutcome.TEST_SUCCESS
    assert cached_result.host_output_directory == out_dir


def test_restart_suffixed_stale_output_directory(tmp_path):
    (tmp_path / "summary.json").write_text("{}")
    args = parse_args(
        [
            "--restart-folder",
            str(tmp_path),
            "--container-runtime=local",
            "--passing-versions",
            "jax:jax-good,xla:xla-good",
            "--failing-versions",
            "jax:jax-bad,xla:xla-good",
            "test-command",
        ]
    )
    tool = TriageTool(args, logging.getLogger("triage-restart-test"))

    stale = tool._test_output_directory("local", {"jax": "jax-good"})
    retry = tool._test_output_directory("local", {"jax": "jax-good"})

    assert stale.name in retry.name
    assert retry.name.endswith("-restart-1")


def test_triage_tool_loads_restart_cache(tmp_path):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad"
    good_dir.mkdir()
    bad_dir.mkdir()
    summary = {
        "versions": [
            {
                "container": "local",
                "xla": "xla-good",
                "jax": "jax-good",
                "output_directory": str(good_dir),
                "result": "TestExecutionOutcome.TEST_SUCCESS",
            },
            {
                "container": "local",
                "xla": "xla-good",
                "jax": "jax-bad",
                "output_directory": str(bad_dir),
                "result": "TestExecutionOutcome.TEST_FAILURE",
            },
        ]
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    args = parse_args(
        [
            "--restart-folder",
            str(tmp_path),
            "--container-runtime=local",
            "--passing-versions",
            "jax:jax-good,xla:xla-good",
            "--failing-versions",
            "jax:jax-bad,xla:xla-good",
            "--confirmation-iterations=0",
            "test-command",
        ]
    )
    tool = TriageTool(args, logging.getLogger("triage-restart-test"))
    tool.bisection_url = "local"
    tool._gather_histories = lambda worker, passing, failing: make_commits()
    tool._check_installation_scripts = lambda worker: set()

    def build_and_test(**kwargs):
        raise AssertionError(f"Unexpected build/test during restart: {kwargs}")

    tool._build_and_test = build_and_test

    result = tool.run_version_bisection(
        {"jax": "jax-good", "xla": "xla-good"},
        {"jax": "jax-bad", "xla": "xla-good"},
    )

    assert result["result"]["jax_good"] == "jax-good"
    assert result["result"]["jax_bad"] == "jax-bad"
