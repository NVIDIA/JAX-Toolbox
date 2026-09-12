import collections
import datetime
import logging
import pathlib
from types import SimpleNamespace

import pytest

from jax_toolbox_triage.logic import (
    _EXIT_CODE_METRIC,
    ExitCodeClassifier,
    TestExecutionOutcome,
    TestResult,
)
from jax_toolbox_triage.triage_tool import TriageTool, _CommitCandidate

START = datetime.datetime(2026, 1, 1)
DAY = datetime.timedelta(days=1)


def make_histories():
    return collections.OrderedDict(
        [
            (
                "xla",
                [
                    ("X0", START),
                    ("X1", START + 2 * DAY),
                    ("X2", START + 4 * DAY),
                    ("X3", START + 6 * DAY),
                ],
            ),
            (
                "jax",
                [
                    ("J0", START),
                    ("J1", START + DAY),
                    ("J2", START + 3 * DAY),
                    ("J3", START + 5 * DAY),
                ],
            ),
            (
                "flax",
                [
                    ("F0", START),
                    ("F1", START + 3 * DAY),
                    ("F2", START + 5 * DAY),
                ],
            ),
        ]
    )


def make_test_result(passes):
    return TestResult(
        build_stdouterr="",
        host_output_directory=pathlib.Path("output"),
        result=TestExecutionOutcome.TEST_YIELDED_RESULTS,
        stdouterr="",
        time=0.0,
        metrics={_EXIT_CODE_METRIC: 0 if passes else 1},
    )


def make_tool(predicate):
    tool = object.__new__(TriageTool)
    tool.args = SimpleNamespace(confirmation_iterations=0)
    tool.logger = logging.getLogger("commit-candidate-tests")
    observed_versions = []

    def build_and_test(*, versions, **kwargs):
        observed_versions.append(versions.copy())
        return make_test_result(predicate(versions))

    tool._build_and_test = build_and_test
    return tool, observed_versions


@pytest.mark.parametrize(
    "references,expected_xla",
    [
        ({}, "X2"),
        ({"xla": "X1"}, "X1"),
    ],
)
def test_candidate_uses_contemporary_or_explicit_references(references, expected_xla):
    tool, observed_versions = make_tool(lambda versions: versions["jax"] == "J1")
    candidate = _CommitCandidate(
        package="jax",
        commit="J2",
        parent="J1",
        date=START + 3 * DAY,
        references=references,
    )

    candidate_result, _ = tool._check_candidate_commit(
        candidate=candidate,
        package_versions=make_histories(),
        result_cache={},
        classifier=ExitCodeClassifier(),
    )

    result, _, _ = candidate_result
    assert result == {
        "jax_bad": "J2",
        "jax_good": "J1",
        "xla_ref": expected_xla,
        "flax_ref": "F1",
    }
    assert {versions["xla"] for versions in observed_versions} == {expected_xla}
    assert {versions["flax"] for versions in observed_versions} == {"F1"}


def test_passing_candidate_narrows_every_package_history():
    tool, _ = make_tool(lambda versions: True)
    candidate = _CommitCandidate(
        package="jax",
        commit="J1",
        parent="J0",
        date=START + DAY,
        references={},
    )

    candidate_result, narrowed = tool._check_candidate_commit(
        candidate=candidate,
        package_versions=make_histories(),
        result_cache={},
        classifier=ExitCodeClassifier(),
    )

    assert candidate_result is None
    assert [version for version, _ in narrowed["jax"]] == ["J1", "J2", "J3"]
    assert [version for version, _ in narrowed["xla"]] == ["X1", "X2", "X3"]
    assert [version for version, _ in narrowed["flax"]] == ["F1", "F2"]


def test_failing_parent_narrows_every_package_history():
    tool, _ = make_tool(lambda versions: False)
    candidate = _CommitCandidate(
        package="jax",
        commit="J2",
        parent="J1",
        date=START + 3 * DAY,
        references={"xla": "X1"},
    )

    candidate_result, narrowed = tool._check_candidate_commit(
        candidate=candidate,
        package_versions=make_histories(),
        result_cache={},
        classifier=ExitCodeClassifier(),
    )

    assert candidate_result is None
    assert [version for version, _ in narrowed["jax"]] == ["J0", "J1"]
    assert [version for version, _ in narrowed["xla"]] == ["X0", "X1"]
    assert [version for version, _ in narrowed["flax"]] == ["F0", "F1"]


def test_candidate_outside_range_can_still_be_confirmed():
    tool, observed_versions = make_tool(lambda versions: versions["jax"] == "J1")
    candidate = _CommitCandidate(
        package="jax",
        commit="J-outside",
        parent="J1",
        date=START + 3 * DAY,
        references={},
    )

    candidate_result, _ = tool._check_candidate_commit(
        candidate=candidate,
        package_versions=make_histories(),
        result_cache={},
        classifier=ExitCodeClassifier(),
    )

    result, _, _ = candidate_result
    assert result["jax_bad"] == "J-outside"
    assert result["jax_good"] == "J1"
    assert result["xla_ref"] == "X2"
    assert {versions["xla"] for versions in observed_versions} == {"X2"}


def test_wrong_candidate_outside_range_does_not_narrow_histories():
    tool, _ = make_tool(lambda versions: True)
    histories = make_histories()
    candidate = _CommitCandidate(
        package="jax",
        commit="J-outside",
        parent="J1",
        date=START + 3 * DAY,
        references={},
    )

    candidate_result, remaining = tool._check_candidate_commit(
        candidate=candidate,
        package_versions=histories,
        result_cache={},
        classifier=ExitCodeClassifier(),
    )

    assert candidate_result is None
    assert remaining is histories
