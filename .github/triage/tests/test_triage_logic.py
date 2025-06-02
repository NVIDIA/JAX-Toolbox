import collections
import datetime
import itertools
import logging
import pytest
import random
from jax_toolbox_triage.logic import commit_search, container_search, TestResult


def wrap(b, commits={}):
    return TestResult(
        host_output_directory="-".join(map("-".join, sorted(commits.items()))),
        result=b,
        stdouterr="",
    )


@pytest.fixture
def logger():
    logger = logging.getLogger("triage-tests")
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    return logger


def make_commits(jax, xla, flax=None):
    # BEWARE: the order is meaningful for the triage algorithm
    return collections.OrderedDict(
        (("xla", xla), ("jax", jax)) + (() if flax is None else (("flax", flax),))
    )


@pytest.mark.parametrize(
    "dummy_test,expected,last_known_good_dir,first_known_bad_dir",
    [
        (
            lambda commits: wrap(commits["jax"] == "oJ", commits),
            {"flax_ref": "nF", "xla_ref": "oX", "jax_bad": "mJ", "jax_good": "oJ"},
            "flax-nF-jax-oJ-xla-oX",
            "flax-nF-jax-mJ-xla-oX",
        ),
        (
            lambda commits: wrap(commits["jax"] != "nJ", commits),
            {"flax_ref": "nF", "xla_ref": "oX", "jax_bad": "nJ", "jax_good": "mJ"},
            "flax-nF-jax-mJ-xla-oX",
            "flax-nF-jax-nJ-xla-oX",
        ),
        (
            lambda commits: wrap(commits["xla"] == "oX", commits),
            {"flax_ref": "nF", "jax_ref": "nJ", "xla_bad": "nX", "xla_good": "oX"},
            "flax-nF-jax-nJ-xla-oX",
            "flax-nF-jax-nJ-xla-nX",
        ),
        (
            lambda commits: wrap(commits["flax"] == "oF", commits),
            {"flax_bad": "nF", "flax_good": "oF", "xla_ref": "oX", "jax_ref": "oJ"},
            "flax-oF-jax-oJ-xla-oX",
            "flax-nF-jax-oJ-xla-oX",
        ),
    ],
)
def test_commit_search_explicit(
    logger, dummy_test, expected, last_known_good_dir, first_known_bad_dir
):
    """
    Test the four possibilities in the hardcoded sequence below, where the container
    level search yielded that (oJ, oX, oF) passed and (nX, nJ, nF) failed. mJ, nJ, nX
    or nF could be the culprit.
    """
    commits = make_commits(
        jax=[("oJ", 1), ("mJ", 2), ("nJ", 3)],
        xla=[("oX", 1), ("nX", 3)],
        flax=[("oF", 1), ("nF", 3)],
    )
    algorithm_result, last_known_good, first_known_bad = commit_search(
        build_and_test=dummy_test,
        commits=commits,
        logger=logger,
        skip_precondition_checks=False,
    )
    assert algorithm_result == expected
    assert last_known_good.result
    assert not first_known_bad.result
    assert last_known_good.host_output_directory == last_known_good_dir
    assert first_known_bad.host_output_directory == first_known_bad_dir


start_date = datetime.datetime(2024, 10, 1)
step_size = datetime.timedelta(days=1)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("packages", [2, 3, 100])
@pytest.mark.parametrize("extra_commits", [0, 2, 7, 100])
def test_commit_search(logger, extra_commits, packages, seed):
    """
    Generate random sequences of commits for `packages` different packages and test
    that the commit-level search algorithm yields the expected results.

    The minimal set of commits generated is (good, bad) for the buggy package and
    (ref_i) for each other package, where the test passes for (good, ref_1, ...) and
    fails for (bad, ref_1, ...).

    Around `extra_commits` extra commits will be added across the 2+ packages.
    """
    rng = random.Random(seed)

    def random_hash():
        return hex(int(rng.uniform(1e10, 9e10)))[2:]

    def random_delay():
        return rng.randint(1, 10) * step_size

    commits = collections.OrderedDict()
    package_names = [f"pkg{i}" for i in range(packages)]
    for package in package_names:
        # Add the minimal good/start-of-range commit
        commits[package] = [
            (random_hash(), start_date + rng.randint(-2, +2) * step_size)
        ]

    def append_random_commits(n):
        for _ in range(n):
            output = rng.choice(list(commits.values()))
            output.append((random_hash(), output[-1][1] + random_delay()))

    # Noise
    append_random_commits(extra_commits // 2)

    # Inject the bad commit
    culprit = rng.choice(package_names)
    good_commit, good_date = commits[culprit][-1]
    bad_commit, bad_date = random_hash(), good_date + random_delay()
    assert good_date < bad_date
    commits[culprit].append((bad_commit, bad_date))

    # Noise
    append_random_commits(extra_commits // 2)

    culprit_dates = {sha: dt for sha, dt in commits[culprit]}
    assert len(culprit_dates) == len(commits[culprit])

    def dummy_test(*, commits):
        return wrap(culprit_dates[commits[culprit]] < bad_date)

    algorithm_result, _, _ = commit_search(
        build_and_test=dummy_test,
        commits=commits,
        logger=logger,
        skip_precondition_checks=False,
    )
    # Do not check the reference commits, it's a bit underspecified quite what they
    # mean, other than that the dummy_test assertions below should pass
    commits = {
        package: algorithm_result.pop(f"{package}_ref")
        for package in package_names
        if package != culprit
    }
    assert {
        f"{culprit}_bad": bad_commit,
        f"{culprit}_good": good_commit,
    } == algorithm_result
    commits[culprit] = algorithm_result[f"{culprit}_good"]
    assert dummy_test(commits=commits).result
    commits[culprit] = algorithm_result[f"{culprit}_bad"]
    assert not dummy_test(commits=commits).result


def other(project):
    return "xla" if project == "jax" else "jax"


def create_commits(num_commits, num_projects):
    """
    Generate commits for test_commit_search_exhaustive.
    """
    assert num_commits > num_projects, (
        "Need at least a good commit for each project plus one bad commit for a culprit project"
    )

    def fake_hash():
        fake_hash.n += 1
        return str(fake_hash.n)

    fake_hash.n = 0
    for first_project in range(num_projects):
        # commit_types is a list of instructions for how to create the next commit from
        # the current one. There are 1 + 2 * (num_projects - 1) values:
        # - same project, increment time (0)
        # - different project, increment time (odd#, project offset = (code-1)//2)
        # - different project, same time (even#, project offset = (code-1)//2)
        # these are coded as integers
        for commit_types in itertools.product(
            range(1 + 2 * (num_projects - 1)), repeat=num_commits - 1
        ):
            commits = [(first_project, fake_hash(), start_date)]
            # Cannot have all commits from the same project
            if sum(commit_types) == 0:
                continue
            for commit_type in commit_types:
                prev_project, _, prev_date = commits[-1]
                if commit_type == 0:  # same
                    commits.append((prev_project, fake_hash(), prev_date + step_size))
                else:
                    project_offset = (commit_type - 1) // 2
                    new_project = (prev_project + project_offset) % num_projects
                    if commit_type % 2 == 1:  # diff
                        commits.append(
                            (new_project, fake_hash(), prev_date + step_size)
                        )
                    else:
                        assert commit_type % 2 == 0  # diff-concurrent
                        commits.append((new_project, fake_hash(), prev_date))
            assert len(commits) == num_commits

            # The commits for a each project must have increasing timestamps
            def increasing(project):
                project_dates = list(
                    map(lambda t: t[2], filter(lambda t: t[0] == project, commits))
                )
                return all(x < y for x, y in zip(project_dates, project_dates[1:]))

            if not all(map(increasing, range(num_projects))):
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


@pytest.mark.parametrize(
    "commits",
    itertools.chain(
        create_commits(num_commits=5, num_projects=2),
        create_commits(num_commits=5, num_projects=3),
    ),
)
def test_commit_search_exhaustive(logger, commits):
    """
    Exhaustive search over combinations of a small number of commits
    """
    # Note that the "project" here is just an integer. That should be OK.
    bad_index, merged_commits = commits
    bad_project, bad_commit, bad_date = merged_commits[bad_index]
    # create_commits internally generates all the different combinations of which of
    # the `num_projects` is "first", so by making sure the "algorithm order" is sorted
    # we should cover all the possibilities
    all_projects = sorted({proj for proj, _, _ in merged_commits})
    split_commits = collections.OrderedDict()
    for p in all_projects:
        split_commits[p] = [
            (commit, date) for proj, commit, date in merged_commits if proj == p
        ]
    good_commit, _ = list(
        filter(lambda t: t[1] < bad_date, split_commits[bad_project])
    )[-1]
    # in this test, there are no commit collisions
    dates = {commit: date for _, commit, date in merged_commits}
    assert all(len(v) for v in split_commits.values())
    assert len(split_commits[bad_project]) >= 2

    def dummy_test(*, commits):
        return wrap(dates[commits[bad_project]] < bad_date)

    algorithm_result, _, _ = commit_search(
        build_and_test=dummy_test,
        commits=split_commits,
        logger=logger,
        skip_precondition_checks=False,
    )
    # Do not check the reference commit, it's a bit underspecified quite what it means.
    assert algorithm_result[f"{bad_project}_bad"] == bad_commit
    assert algorithm_result[f"{bad_project}_good"] == good_commit
    # Do check that the reference commit gives the expected results
    commits = {
        proj: algorithm_result[f"{proj}_ref"]
        for proj in all_projects
        if proj != bad_project
    }
    commits[bad_project] = bad_commit
    assert not dummy_test(commits=commits).result
    commits[bad_project] = good_commit
    assert dummy_test(commits=commits).result


@pytest.mark.parametrize(
    "commits",
    [
        make_commits(jax=[("", start_date)], xla=[]),
        make_commits(jax=[("", start_date)], flax=[], xla=[]),
        make_commits(xla=[("", start_date)], jax=[]),
        make_commits(xla=[("", start_date)], flax=[], jax=[]),
        make_commits(jax=[("", start_date)], xla=[("", start_date)]),
        make_commits(jax=[("", start_date)], xla=[("", start_date)], flax=[]),
        make_commits(
            jax=[("", start_date)], xla=[("", start_date)], flax=[("", start_date)]
        ),
    ],
)
def test_commit_search_no_commits(logger, commits):
    with pytest.raises(Exception, match="Not enough commits"):
        commit_search(
            build_and_test=lambda commits: None,
            commits=commits,
            logger=logger,
            skip_precondition_checks=False,
        )


@pytest.mark.parametrize("value", [True, False])
def test_commit_search_static_test_function(logger, value):
    with pytest.raises(Exception, match="Could not reproduce"):
        commit_search(
            build_and_test=lambda commits: wrap(value),
            commits=make_commits(
                jax=[("", start_date), ("", start_date + step_size)],
                xla=[("", start_date), ("", start_date + step_size)],
            ),
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
            container_passes=lambda dt: wrap(False),
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
            container_passes=lambda dt: wrap(dt in dates_that_pass),
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
        container_passes=lambda dt: wrap(dt < threshold_date),
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
