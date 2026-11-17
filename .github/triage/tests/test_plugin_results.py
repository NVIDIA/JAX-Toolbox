import json
import logging
import subprocess
from types import SimpleNamespace

from jax_toolbox_triage.logic import TestExecutionOutcome
from jax_toolbox_triage.triage_tool import TriageTool


def make_tool():
    tool = object.__new__(TriageTool)
    tool.args = SimpleNamespace(container_runtime="plugin", metric_name=None)
    tool.logger = logging.getLogger("plugin-result-tests")
    return tool


def test_plugin_can_report_missing_exit_code(tmp_path):
    with open(tmp_path / "metrics.json", "w") as metrics_file:
        json.dump({}, metrics_file)

    result = make_tool()._run_test(
        lambda: (
            subprocess.CompletedProcess([], 1, stdout="submission failed"),
            tmp_path,
            "example.invalid/container",
        )
    )

    assert result.result == TestExecutionOutcome.TEST_YIELDED_RESULTS
    assert result.metrics == {}


def test_plugin_without_metrics_reports_test_error(tmp_path):
    result = make_tool()._run_test(
        lambda: (
            subprocess.CompletedProcess([], 1, stdout="workload failed"),
            tmp_path,
            "example.invalid/container",
        )
    )

    assert result.result == TestExecutionOutcome.TEST_ERROR
    assert result.metrics == {}
