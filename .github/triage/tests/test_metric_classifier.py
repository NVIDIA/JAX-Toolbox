from collections import defaultdict
from functools import partial
import pathlib
import pytest
import random

from jax_toolbox_triage.logic import (
    ClassifiedTestOutcome,
    TestExecutionOutcome,
    TestResult,
)
from jax_toolbox_triage.metric_classifier import MetricClassifier

_PASS_MU = 1
_FAIL_MU = 0
_METRIC_NAME = "metric"
_MAX_RETRIES = 10
_SAMPLES_TO_TEST = 1000
_OUTCOME_MAP = {
    True: ClassifiedTestOutcome.PASS,
    False: ClassifiedTestOutcome.FAIL,
}


def wrap(metric_value: float) -> TestResult:
    return TestResult(
        build_stdouterr=None,
        host_output_directory=pathlib.Path(),
        metrics={_METRIC_NAME: metric_value},
        result=TestExecutionOutcome.TEST_YIELDED_RESULTS,
        stdouterr=None,
        time=0.0,
    )


@pytest.mark.parametrize("random_seed", range(4))
# How many labelled values to seed the classifier with.
@pytest.mark.parametrize("seed_values", [2, 4])
# Noise level in the good measurements.
@pytest.mark.parametrize("pass_sigma", [1e-4, 0.1, 0.2])
# Noise level in the bad measurements.
@pytest.mark.parametrize("fail_sigma", [1e-4, 0.1, 0.2])
# Effectively whether the culprit is early or late in the search range. This is
# symmetric, so no need to cover 0.75 and 1.0
@pytest.mark.parametrize("pass_probability", [0.0, 0.25, 0.5])
# Whether the pass/fail distributions should be assumed to have the same width
@pytest.mark.parametrize("common_variance", [True, False])
# How tolerant of failures to be
@pytest.mark.parametrize("threshold", [0.954, 0.997])
def test_random_data(
    random_seed,
    seed_values,
    pass_sigma,
    fail_sigma,
    pass_probability,
    common_variance,
    threshold,
):
    if pass_sigma != fail_sigma and common_variance:
        pytest.skip(
            "Expect failures when assuming common variance if the true variances "
            "differ too much."
        )
    rng = random.Random(random_seed)
    draw_pass = partial(rng.gauss, mu=_PASS_MU, sigma=pass_sigma)
    draw_fail = partial(rng.gauss, mu=_FAIL_MU, sigma=fail_sigma)
    classifier = MetricClassifier(
        metric_name=_METRIC_NAME,
        threshold=threshold,
        common_variance=common_variance,
        passing_values=[draw_pass() for _ in range(seed_values)],
        failing_values=[draw_fail() for _ in range(seed_values)],
    )
    test_stats = defaultdict(int)
    for _ in range(_SAMPLES_TO_TEST):
        is_pass = rng.random() <= pass_probability
        values = []
        for _ in range(_MAX_RETRIES):
            values.append(wrap(draw_pass() if is_pass else draw_fail()))
            outcome = classifier(values)
            if outcome != ClassifiedTestOutcome.AMBIGUOUS:
                break
        for row in classifier.text_summary():
            print(row)
        if outcome == ClassifiedTestOutcome.AMBIGUOUS:
            test_stats["ambiguous"] += 1
        elif outcome == ClassifiedTestOutcome.ERROR:
            test_stats["errors"] += 1
        elif outcome == _OUTCOME_MAP[is_pass]:
            test_stats["correct"] += 1
        else:
            assert outcome == _OUTCOME_MAP[not is_pass], outcome
            test_stats["incorrect"] += 1
    assert sum(test_stats.values()) == _SAMPLES_TO_TEST
    assert test_stats["errors"] == 0, test_stats
    hit_rate = test_stats["correct"] / (
        test_stats["incorrect"] + test_stats["correct"] + test_stats["ambiguous"]
    )
    # The misclassification rate should not significantly exceed the
    # configured threshold. If the pass/fail populations are very distinct, it's hard
    # to get wrong and the hit rate will be ~100% regardless of the threshold.
    assert hit_rate > threshold * 0.9, (hit_rate, threshold)


def test_progress_reporting():
    rng = random.Random(0)
    draw_pass = partial(rng.gauss, mu=_PASS_MU, sigma=0.1)
    draw_fail = partial(rng.gauss, mu=_FAIL_MU, sigma=0.15)
    seed_values = 2
    classifier = MetricClassifier(
        metric_name=_METRIC_NAME,
        threshold=0.954,
        common_variance=False,
        passing_values=[draw_pass() for _ in range(seed_values)],
        failing_values=[draw_fail() for _ in range(seed_values)],
    )
    initial_summary = classifier.text_summary(columns=60, rows=2)
    initial_reference = [
        "x=[-0.12, 1.12] pass=0.98+/-0.14 (n=2) fail=-0.02+/-0.10 (n=2)",
        "█       █                                                   ",
        "█       █                                                   ",
        "\\FFF|FFFF/                                                  ",
        "                                               █          █ ",
        "                                               █          █ ",
        "                                              \\PPPPPP|PPPPP/",
    ]
    assert initial_summary == initial_reference
    for _ in range(100):
        is_pass = rng.random() <= 0.25
        values = []
        for _ in range(_MAX_RETRIES):
            values.append(wrap(draw_pass() if is_pass else draw_fail()))
            outcome = classifier(values)
            if outcome != ClassifiedTestOutcome.AMBIGUOUS:
                break
    final_summary = classifier.text_summary(columns=60, rows=2)
    final_reference = [
        "x=[-0.35, 1.24] pass=1.00+/-0.13 (n=26) fail=-0.01+/-0.15 (n=81)",
        "        ▅  ▅    █                                           ",
        "▂▄▅ ▄▂▂▇██▇█▇██▇█▄▄▂▅ ▄▂ ▄ ▂                                ",
        "      \\FFFFF|FFFFF/                                         ",
        "                                                            ",
        "                                          ▄  ▄▄▇▂ ▅▂▂▄▄▄ ▄▂▂",
        "                                             \\PPPP|PPPP/    ",
    ]
    assert final_summary == final_reference
