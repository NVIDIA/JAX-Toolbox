import datetime
import itertools
import logging
import pytest
import random
from jax_toolbox_triage.logic import commit_search, container_search, TestResult


def wrap(b):
    return b, "", ""


@pytest.fixture
def logger():
    logger = logging.getLogger("triage-tests")
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.mark.parametrize(
    "dummy_test,expected",
    [
        (
            lambda jax_commit, xla_commit: wrap(jax_commit == "oJ"),
            {"xla_ref": "oX", "jax_bad": "mJ", "jax_good": "oJ"},
        ),
        (
            lambda jax_commit, xla_commit: wrap(jax_commit != "nJ"),
            {"xla_ref": "oX", "jax_bad": "nJ", "jax_good": "mJ"},
        ),
        (
            lambda jax_commit, xla_commit: wrap(xla_commit == "oX"),
            {"jax_ref": "nJ", "xla_bad": "nX", "xla_good": "oX"},
        ),
    ],
)
def test_commit_search_explicit(logger, dummy_test, expected):
    """
    Test the three possibilities in the hardcoded sequence below, where the container
    level search yielded that (oJ, oX) passed and (nX, nJ) failed. mJ, nJ or nX could
    be the culprit.
    """
    jax_commits = [("oJ", 1), ("mJ", 2), ("nJ", 3)]
    xla_commits = [("oX", 1), ("nX", 3)]
    algorithm_result = commit_search(
        build_and_test=dummy_test,
        jax_commits=jax_commits,
        logger=logger,
        skip_precondition_checks=False,
        xla_commits=xla_commits,
    )
    assert algorithm_result == expected


start_date = datetime.datetime(2024, 10, 1)
step_size = datetime.timedelta(days=1)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("extra_commits", [0, 2, 7, 100])
def test_commit_search(logger, extra_commits, seed):
    """
    Generate random sequences of JAX/XLA commits and test that the commit-level search
    algorithm yields the expected results.

    The minimal set of commits generated is (good, bad) for one component and (ref) for
    the other, where the test passes for (good, ref) and fails for (bad, ref).

    Around `extra_commits` extra commits will be added across the two components around
    these.
    """
    rng = random.Random(seed)

    def random_hash():
        return hex(int(rng.uniform(1e10, 9e10)))[2:]

    def random_delay():
        return rng.randint(1, 10) * step_size

    # Randomise whether JAX or XLA is newer at the start of the range
    commits = {
        "jax": [(random_hash(), start_date)],
        "xla": [(random_hash(), start_date + rng.randint(-2, +2) * step_size)],
    }

    def append_random_commits(n):
        for _ in range(n):
            output = rng.choice(list(commits.values()))
            output.append((random_hash(), output[-1][1] + random_delay()))

    # Noise
    append_random_commits(extra_commits // 2)

    # Inject the bad commit
    culprit, innocent = rng.choice([("jax", "xla"), ("xla", "jax")])
    good_commit, good_date = commits[culprit][-1]
    bad_commit, bad_date = random_hash(), good_date + random_delay()
    assert good_date < bad_date
    commits[culprit].append((bad_commit, bad_date))

    # Noise
    append_random_commits(extra_commits // 2)

    def dummy_test(*, jax_commit, xla_commit):
        jax_date = {sha: dt for sha, dt in commits["jax"]}[jax_commit]
        xla_date = {sha: dt for sha, dt in commits["xla"]}[xla_commit]
        return wrap(xla_date < bad_date if culprit == "xla" else jax_date < bad_date)

    algorithm_result = commit_search(
        build_and_test=dummy_test,
        jax_commits=commits["jax"],
        logger=logger,
        skip_precondition_checks=False,
        xla_commits=commits["xla"],
    )
    # Do not check the reference commit, it's a bit underspecified quite what it means,
    # other than that the dummy_test assertions below should pass
    innocent_ref = algorithm_result.pop(f"{innocent}_ref")
    assert {
        f"{culprit}_bad": bad_commit,
        f"{culprit}_good": good_commit,
    } == algorithm_result
    if culprit == "jax":
        assert not dummy_test(jax_commit=bad_commit, xla_commit=innocent_ref)[0]
        assert dummy_test(jax_commit=good_commit, xla_commit=innocent_ref)[0]
    else:
        assert not dummy_test(jax_commit=innocent_ref, xla_commit=bad_commit)[0]
        assert dummy_test(jax_commit=innocent_ref, xla_commit=good_commit)[0]


def other(project):
    return "xla" if project == "jax" else "jax"


def create_commits(num_commits):
    """
    Generate commits for test_commit_search_exhaustive.
    """

    def fake_hash():
        fake_hash.n += 1
        return str(fake_hash.n)

    fake_hash.n = 0
    for first_project in ["jax", "xla"]:
        for commit_types in itertools.product(range(3), repeat=num_commits - 1):
            commits = [(first_project, fake_hash(), start_date)]
            # Cannot have all commits from the same project
            if sum(commit_types) == 0:
                continue
            for commit_type in commit_types:
                prev_project, _, prev_date = commits[-1]
                if commit_type == 0:  # same
                    commits.append((prev_project, fake_hash(), prev_date + step_size))
                elif commit_type == 1:  # diff
                    commits.append(
                        (other(prev_project), fake_hash(), prev_date + step_size)
                    )
                else:
                    assert commit_type == 2  # diff-concurrent
                    commits.append((other(prev_project), fake_hash(), prev_date))
            assert len(commits) == num_commits

            # The commits for a each project must have increasing timestamps
            def increasing(project):
                project_dates = list(
                    map(lambda t: t[2], filter(lambda t: t[0] == project, commits))
                )
                return all(x < y for x, y in zip(project_dates, project_dates[1:]))

            if not increasing("jax") or not increasing("xla"):
                continue

            for bad_commit_index in range(
                1,  # bad commit cannot be the first one
                num_commits,
            ):
                bad_project, _, _ = commits[bad_commit_index]
                # there must be a good commit before the last one
                if not any(
                    project == bad_project
                    for project, _, _ in commits[:bad_commit_index]
                ):
                    continue
                yield bad_commit_index, commits


@pytest.mark.parametrize("commits", create_commits(5))
def test_commit_search_exhaustive(logger, commits):
    """
    Exhaustive search over combinations of a small number of commits
    """
    bad_index, merged_commits = commits
    bad_project, bad_commit, bad_date = merged_commits[bad_index]
    good_project = other(bad_project)
    split_commits = {
        p: [(commit, date) for proj, commit, date in merged_commits if proj == p]
        for p in ("jax", "xla")
    }
    good_commit, _ = list(
        filter(lambda t: t[1] < bad_date, split_commits[bad_project])
    )[-1]
    # in this test, there are no commit collisions
    dates = {commit: date for _, commit, date in merged_commits}
    assert all(len(v) for v in split_commits.values())
    assert len(split_commits[bad_project]) >= 2

    def dummy_test(*, jax_commit, xla_commit):
        return wrap(
            dates[jax_commit if bad_project == "jax" else xla_commit] < bad_date
        )

    algorithm_result = commit_search(
        build_and_test=dummy_test,
        jax_commits=split_commits["jax"],
        logger=logger,
        skip_precondition_checks=False,
        xla_commits=split_commits["xla"],
    )
    # Do not check the reference commit, it's a bit underspecified quite what it means.
    assert algorithm_result[f"{bad_project}_bad"] == bad_commit
    assert algorithm_result[f"{bad_project}_good"] == good_commit
    # Do check that the reference commit gives the expected results
    assert not dummy_test(
        **{
            f"{bad_project}_commit": bad_commit,
            f"{good_project}_commit": algorithm_result[f"{good_project}_ref"],
        }
    )[0]
    assert dummy_test(
        **{
            f"{bad_project}_commit": good_commit,
            f"{good_project}_commit": algorithm_result[f"{good_project}_ref"],
        }
    )[0]


@pytest.mark.parametrize(
    "jax_commits,xla_commits",
    [
        ([], [("", start_date)]),
        ([("", start_date)], []),
        ([("", start_date)], [("", start_date)]),
    ],
)
def test_commit_search_no_commits(logger, jax_commits, xla_commits):
    with pytest.raises(Exception, match="Not enough commits"):
        commit_search(
            build_and_test=lambda jax_commit, xla_commit: None,
            jax_commits=jax_commits,
            logger=logger,
            skip_precondition_checks=False,
            xla_commits=xla_commits,
        )


@pytest.mark.parametrize("value", [True, False])
def test_commit_search_static_test_function(logger, value):
    with pytest.raises(Exception, match="Could not reproduce"):
        commit_search(
            build_and_test=lambda jax_commit, xla_commit: wrap(value),
            jax_commits=[("", start_date), ("", start_date + step_size)],
            xla_commits=[("", start_date), ("", start_date + step_size)],
            logger=logger,
            skip_precondition_checks=False,
        )


far_future = datetime.date(year=2100, month=1, day=1)
further_future = datetime.date(year=2100, month=1, day=12)
assert far_future > datetime.date.today()
assert further_future > far_future
good_date = datetime.date(year=2100, month=1, day=1)
bad_date = datetime.date(year=2100, month=1, day=12)


@pytest.mark.parametrize(
    "start_date,end_date,dates_that_exist,match_string",
    [
        # Explicit start_date is later than explicit end date
        (
            further_future,
            far_future,
            {far_future, further_future},
            "2100-01-12 must be before 2100-01-01",
        ),
        # Good order, but both invalid
        (far_future, further_future, {}, "is not a valid container"),
        # Good order, one invalid
        (far_future, further_future, {far_future}, "is not a valid container"),
        (far_future, further_future, {further_future}, "is not a valid container"),
        # Valid end_date, but there are no valid earlier ones to be found
        (None, far_future, {far_future}, "Could not find a valid nightly before"),
        # Start from today, nothing valid to be found
        (None, None, {}, "Could not find a valid container from"),
        # Valid start, default end will not work
        (far_future, None, {far_future}, "Could not find a valid container from"),
    ],
)
def test_container_search_limits(
    logger, start_date, end_date, dates_that_exist, match_string
):
    """
    Test for failure if an invalid date is explicitly passed.
    """
    with pytest.raises(Exception, match=match_string):
        container_search(
            container_exists=lambda dt: dt in dates_that_exist,
            container_passes=lambda dt: TestResult(result=False),
            start_date=start_date,
            end_date=end_date,
            logger=logger,
            skip_precondition_checks=False,
            threshold_days=1,
        )


@pytest.mark.parametrize(
    "start_date,end_date,dates_that_pass,match_string",
    [
        # Test never passes
        pytest.param(
            far_future,
            further_future,
            {},
            "Could not find a passing nightly before",
            marks=pytest.mark.xfail(
                reason="No cutoff implemented if all dates exist but none pass"
            ),
        ),
        # Test passes at the end of the range but not the start
        (
            far_future,
            further_future,
            {further_future},
            "Could not reproduce failure in",
        ),
        # Test passes at both ends of the range
        (
            far_future,
            further_future,
            {far_future, further_future},
            "Could not reproduce failure in",
        ),
    ],
)
def test_container_search_checks(
    logger, start_date, end_date, dates_that_pass, match_string
):
    """
    Test for failure if start/end dates are given that do not meet the preconditions.
    """
    with pytest.raises(Exception, match=match_string):
        container_search(
            container_exists=lambda dt: True,
            container_passes=lambda dt: TestResult(result=dt in dates_that_pass),
            start_date=start_date,
            end_date=end_date,
            logger=logger,
            skip_precondition_checks=False,
            threshold_days=1,
        )


@pytest.mark.parametrize("start_date", [None, datetime.date(year=2024, month=1, day=1)])
@pytest.mark.parametrize(
    "days_of_failure", [1, 2, 17, 19, 32, 64, 71, 113, 128, 256, 359]
)
@pytest.mark.parametrize("threshold_days", [1, 4, 15])
def test_container_search(logger, start_date, days_of_failure, threshold_days):
    end_date = datetime.date(year=2024, month=12, day=31)
    one_day = datetime.timedelta(days=1)
    threshold_date = end_date - days_of_failure * one_day
    assert start_date is None or threshold_date >= start_date
    good_date, bad_date = container_search(
        container_exists=lambda dt: True,
        container_passes=lambda dt: TestResult(result=dt < threshold_date),
        start_date=start_date,
        end_date=end_date,
        logger=logger,
        skip_precondition_checks=False,
        threshold_days=threshold_days,
    )
    assert bad_date != good_date
    assert bad_date - good_date <= datetime.timedelta(days=threshold_days)
    assert good_date < threshold_date
    assert bad_date >= threshold_date
