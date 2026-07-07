import math
import numpy as np
import typing
from uncertainties import ufloat

from .logic import (
    ClassifiedTestOutcome,
    ExecutionClassifier,
    TestExecutionOutcome,
    TestResult,
)


class MetricClassifier(ExecutionClassifier):
    def __init__(
        self,
        *,
        metric_name: str,
        passing_values: typing.List[float],
        failing_values: typing.List[float],
        threshold: float = 0.997,
        common_variance: bool = True,
    ):
        """
        The classifier models the metric as two Gaussian distributions (pass/fail),
        optionally constrained to have the same variance as one another. The parameters
        of the two distributions are seeded by the values passed to this constructor
        for each population, and are optionally updated with additional labelled values
        using `add_data_with_label`. If new values can be assigned to one of the two
        populations with confidence greater than `threshold`, they are also used to
        update the estimated parameters of the distributions.
        """
        self._metric_name = metric_name
        self._threshold = threshold
        self._log_threshold_odds = math.log(threshold / (1 - threshold))
        self._pass = self._fit(passing_values)
        self._fail = self._fit(failing_values)
        # Make sure neither of the widths is excessively low w.r.t. the spread between
        # the seed mean values of the two populations.
        self._min_b = 1e-4 * ((self._pass["m"] - self._fail["m"]) ** 2)
        self._pass["b"] = max(self._min_b, self._pass["b"])
        self._fail["b"] = max(self._min_b, self._fail["b"])
        self._shared = {}
        if common_variance:
            self._shared["a"] = self._pass.pop("a") + self._fail.pop("a")
            self._shared["b"] = self._pass.pop("b") + self._fail.pop("b")
        self._pass_history = passing_values.copy()
        self._fail_history = failing_values.copy()

    def _fit(self, xs):
        n = len(xs)
        m = sum(xs) / n
        ss = sum((x - m) ** 2 for x in xs)
        return {
            "m": m,
            "k": n,
            "a": n / 2,
            "b": 0.5 * ss,
        }

    def _scale(self, s) -> float:
        a = s.get("a", self._shared.get("a"))
        b = s.get("b", self._shared.get("b"))
        assert a is not None and b is not None
        return math.sqrt(b * (s["k"] + 1) / (a * s["k"]))

    def _log_evidence(self, s, xs) -> float:
        n = len(xs)
        xb = sum(xs) / n
        ss = sum((x - xb) ** 2 for x in xs)

        a_input = s.get("a", self._shared.get("a"))
        b_input = s.get("b", self._shared.get("b"))

        k = s["k"] + n
        a = a_input + n / 2
        b = b_input + 0.5 * ss + s["k"] * n * (xb - s["m"]) ** 2 / (2 * k)
        return (
            math.lgamma(a)
            - math.lgamma(a_input)
            + 0.5 * (math.log(s["k"]) - math.log(k))
            + a_input * math.log(b_input)
            - a * math.log(b)
            - n / 2 * math.log(math.pi)
        )

    def _log_ratio(self, xs) -> float:
        return self._log_evidence(self._pass, xs) - self._log_evidence(self._fail, xs)

    def _update(self, s, xs):
        n = len(xs)
        xb = sum(xs) / n
        k = s["k"] + n

        def _add_to(key, v):
            if key in s:
                s[key] += v
            else:
                self._shared[key] += v

        _add_to(
            "b",
            0.5 * sum((x - xb) ** 2 for x in xs)
            + s["k"] * n * (xb - s["m"]) ** 2 / (2 * k),
        )
        s["m"] = (s["k"] * s["m"] + n * xb) / k
        s["k"] = k
        _add_to("a", n / 2)

    def __call__(
        self, test_results: typing.Sequence[TestResult]
    ) -> ClassifiedTestOutcome:
        assert len(test_results)

        if any(
            r.result != TestExecutionOutcome.TEST_YIELDED_RESULTS for r in test_results
        ):
            return ClassifiedTestOutcome.ERROR

        # FIXME: what if some of `test_results` have been seen before?
        xs = [r.metrics[self._metric_name] for r in test_results]
        log_ratio = self._log_ratio(xs)
        if log_ratio > self._log_threshold_odds:
            self._pass_history += xs
            self._update(self._pass, xs)
            return ClassifiedTestOutcome.PASS
        elif log_ratio < -self._log_threshold_odds:
            self._fail_history += xs
            self._update(self._fail, xs)
            return ClassifiedTestOutcome.FAIL
        return ClassifiedTestOutcome.AMBIGUOUS

    def add_data_with_label(
        self, results: typing.Sequence[TestResult], label: ClassifiedTestOutcome
    ):
        xs = [r.metrics[self._metric_name] for r in results]
        if label == ClassifiedTestOutcome.PASS:
            self._pass_history += xs
            self._update(self._pass, xs)
        else:
            assert label == ClassifiedTestOutcome.FAIL, label
            self._fail_history += xs
            self._update(self._fail, xs)

    def text_summary(self, *, columns: int = 120, rows: int = 3) -> typing.List[str]:
        fail_mean, pass_mean = self._fail["m"], self._pass["m"]
        fail_width, pass_width = self._scale(self._fail), self._scale(self._pass)
        fail_lo, fail_hi = fail_mean - fail_width, fail_mean + fail_width
        pass_lo, pass_hi = pass_mean - pass_width, pass_mean + pass_width
        min_x = min(min(self._fail_history), min(self._pass_history), fail_lo, pass_lo)
        max_x = max(max(self._fail_history), max(self._pass_history), fail_hi, pass_hi)

        def _hist(data, *, edges=False):
            data, bin_data = np.histogram(data, bins=columns, range=(min_x, max_x))
            return (data, bin_data) if edges else data

        fail_hist, bin_edges = _hist(self._fail_history, edges=True)
        pass_hist = _hist(self._pass_history)
        max_bin_value = int(max(fail_hist.max(), pass_hist.max()))
        sep = ""  # narrow space tempting, but not with monospace font
        output = []

        def _print_hist(hist):
            dots = {
                0: " ",
                1: "\N{LOWER ONE EIGHTH BLOCK}",
                2: "\N{LOWER ONE QUARTER BLOCK}",
                3: "\N{LOWER THREE EIGHTHS BLOCK}",
                4: "\N{LOWER HALF BLOCK}",
                5: "\N{LOWER FIVE EIGHTHS BLOCK}",
                6: "\N{LOWER THREE QUARTERS BLOCK}",
                7: "\N{LOWER SEVEN EIGHTHS BLOCK}",
                8: "\N{FULL BLOCK}",
            }
            max_dots = max(dots.keys())
            for row in range(rows)[::-1]:
                chars = []
                for col in range(columns):
                    # ceil to avoid rounding non-zero bins to zero
                    height = int(math.ceil(rows * max_dots * hist[col] / max_bin_value))
                    chars.append(dots[min(max_dots, max(0, height - max_dots * row))])
                assert len(chars) == columns
                output.append(sep.join(chars))

        low_edges, high_edges = bin_edges[:-1], bin_edges[1:]

        def _print_label(*, low, mean, high, char):
            row = np.asarray([" "] * columns)
            row[(low_edges > low) & (high_edges < high)] = char
            row[_hist([low]).astype(bool)] = "\\"
            row[_hist([high]).astype(bool)] = "/"
            row[_hist([mean]).astype(bool)] = "|"
            output.append(sep.join(row))

        pass_uf = ufloat(pass_mean, pass_width)
        fail_uf = ufloat(fail_mean, fail_width)
        output.append(
            f"x=[{min_x:.2f}, {max_x:.2f}] "
            f"pass={pass_uf} (n={len(self._pass_history)}) "
            f"fail={fail_uf} (n={len(self._fail_history)})"
        )
        _print_hist(fail_hist)
        _print_label(low=fail_lo, high=fail_hi, mean=fail_mean, char="F")
        _print_hist(pass_hist)
        _print_label(low=pass_lo, high=pass_hi, mean=pass_mean, char="P")
        return output
