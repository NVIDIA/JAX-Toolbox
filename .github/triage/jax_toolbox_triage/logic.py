import collections
from dataclasses import dataclass
import datetime
import functools
import itertools
import logging
import typing

from .utils import console_log_level


@dataclass
class TestResult:
    """
    Hold the pass/fail result and the interleaved stdout/stderr of a test run
    """

    __test__ = False  # stop pytest gathering this
    result: bool
    stdouterr: str


def as_datetime(date: datetime.date) -> datetime.datetime:
    return datetime.datetime.combine(date, datetime.time())


def adjust_date(
    date: datetime.datetime,
    logger: logging.Logger,
    container_exists: typing.Callable[[datetime.date], bool],
    before: typing.Optional[datetime.date] = None,
    after: typing.Optional[datetime.date] = None,
    max_steps: int = 100,
) -> typing.Optional[datetime.date]:
    """
    Given a datetime that may have non-zero hour/minute/second/... parts, and where
    container_url(date.date()) might be a container that does not exist due to job
    failure, return a similar date where container_url(new_date) does exist, or None if
    no such container can be found.

    Arguments:
    date: date to adjust
    before: the returned date will be before this [optional]
    after: the returned date will be after this [optional]
    max_steps: maximum number of days away from the start date to venture
    """
    round_up = date.time() > datetime.time(12)
    down, up = (date.date(), -1), (date.date() + datetime.timedelta(days=1), +1)
    options = [up, down] if round_up else [down, up]
    n = 0
    while n < max_steps:
        plausible_directions = 0
        for start, direction in options:
            candidate = start + n * direction * datetime.timedelta(days=1)
            if (before is None or candidate < before) and (
                after is None or candidate > after
            ):
                plausible_directions += 1
                if container_exists(candidate):
                    if date.date() != candidate:
                        logger.debug(f"Adjusted {date} to {candidate}")
                    return candidate
                else:
                    logger.debug(f"{candidate} does not exist")
        n += 1
        if plausible_directions == 0:
            logger.info(
                f"Could not adjust {date} given before={before} and after={after}"
            )
            return None
    logger.info(f"Could not find an adjusted {date} within {max_steps} steps")
    return None


def container_search(
    *,
    container_exists: typing.Callable[[datetime.date], bool],
    container_passes: typing.Callable[[datetime.date], TestResult],
    start_date: typing.Optional[datetime.date],
    end_date: typing.Optional[datetime.date],
    logger: logging.Logger,
    skip_precondition_checks: bool,
    threshold_days: int,
):
    adjust = functools.partial(
        adjust_date, logger=logger, container_exists=container_exists
    )
    # Figure out the end date of the search
    if end_date is not None:
        # --end-date was passed
        if not container_exists(end_date):
            raise Exception(f"--end-date={end_date} is not a valid container")
        skip_end_date_check = skip_precondition_checks
    else:
        # Default to the most recent container
        now = datetime.datetime.now()
        end_date = adjust(now)
        if end_date is None:
            raise Exception(f"Could not find a valid container from {now}")
        skip_end_date_check = False

    # Check preconditions; the test is supposed to fail on the end date.
    if skip_end_date_check:
        logger.info(f"Skipping check for end-of-range failure in {end_date}")
    else:
        logger.info(f"Checking end-of-range failure in {end_date}")
        # Print context for the IMPORTANT .info(...) to the console
        with console_log_level(logger, logging.DEBUG):
            test_end_date = container_passes(end_date)
        if test_end_date.result:
            raise Exception(f"Could not reproduce failure in {end_date}")
        logger.info(
            "IMPORTANT: you should check that the test output above shows the "
            f"*expected* failure of your test case in the {end_date} container. It is "
            "very easy to accidentally provide a test case that fails for the wrong "
            "reason, which will not triage the correct issue!"
        )

    # Start the coarse, container-level, search for a starting point to the bisection range
    earliest_failure = end_date
    if start_date is None:
        # Start from the day before the end date.
        search_date = adjust(
            as_datetime(end_date) - datetime.timedelta(days=1), before=end_date
        )
        if search_date is None:
            raise Exception(f"Could not find a valid nightly before {end_date}")
        logger.info(
            f"Starting coarse search with {search_date} based on end_date={end_date}"
        )
        # We just found a starting value, we need to actually check if the test passes or
        # fails on it.
        skip_first_phase = False
    else:
        # If a start value seed was given, use it.
        if start_date >= end_date:
            raise Exception(f"{start_date} must be before {end_date}")
        if not container_exists(start_date):
            raise Exception(f"--start-date={start_date} is not a valid container")
        search_date = start_date
        assert search_date is not None  # for mypy
        # If --skip-precondition-checks and --start-date are both passed, we assume that
        # the test passed on the given --start-date and the first phase of the search can
        # be skipped
        skip_first_phase = skip_precondition_checks
        if not skip_first_phase:
            logger.info(
                f"Starting coarse search with {search_date} based on --start-date"
            )

    if skip_first_phase:
        logger.info(f"Skipping check that the test passes on start_date={start_date}")
    else:
        # While condition prints an info message
        while not container_passes(search_date).result:
            # Test failed on `search_date`, go further into the past
            earliest_failure = search_date
            new_search_date = adjust(
                as_datetime(end_date) - 2 * (end_date - search_date),
                before=search_date,
            )
            if new_search_date is None:
                raise Exception(
                    f"Could not find a passing nightly before {search_date}"
                )
            search_date = new_search_date

    # Continue the container-level search, refining the range until it meets the criterion
    # set by args.threshold_days. The test passed at range_start and not at range_end.
    range_start, range_end = search_date, earliest_failure
    logger.info(
        f"Coarse container-level search yielded [{range_start}, {range_end}]..."
    )
    while range_end - range_start > datetime.timedelta(days=threshold_days):
        range_mid = adjust(
            as_datetime(range_start) + 0.5 * (range_end - range_start),
            before=range_end,
            after=range_start,
        )
        if range_mid is None:
            # It wasn't possible to refine further.
            break
        result = container_passes(range_mid).result
        if result:
            range_start = range_mid
        else:
            range_end = range_mid
        logger.info(f"Refined container-level range to [{range_start}, {range_end}]")
    return range_start, range_end


class BuildAndTest(typing.Protocol):
    def __call__(self, *, commits: dict[str, str]) -> TestResult:
        """
        Given an [unordered] set of {package_name: package_commit_sha}, build
        those package versions and return the test result.
        """
        ...


def _not_first(d):
    return itertools.islice(d.items(), 1, None)


def commit_search(
    *,
    commits: collections.OrderedDict[
        str, typing.Sequence[typing.Tuple[str, datetime.datetime]]
    ],
    build_and_test: BuildAndTest,
    logger: logging.Logger,
    skip_precondition_checks: bool,
):
    """
    Bisect a failure back to a single commit.

    Arguments:
    commits: *ordered* dictionary of commit sequences for different software
        packages, e.g. commits["jax"][0] is (hash, date) of the passing JAX
        commit. The ordering of packages has implications for precisely how
        the triage proceeds.
    build_and_test: callable that tests if a given vector of commits passes
    logger: instance to log output to
    skip_precondition_checks: if True, some tests that should pass/fail by
        construction are skipped
    """
    assert all(len(commit_list) for commit_list in commits.values()), (
        "Not enough commits: need at least one commit for each package",
        commits,
    )
    assert sum(map(len, commits.values())) > len(commits), (
        "Not enough commits: need multiple commits for at least one package",
        commits,
    )

    if skip_precondition_checks:
        logger.info("Skipping check that 'good' commits reproduce success")
    else:
        # Verify that we can build successfully and that the test succeeds as expected.
        logger.info("Verifying test success using 'good' commits")
        passing_commits = {
            package: commit_list[0][0] for package, commit_list in commits.items()
        }
        check_pass = build_and_test(commits=passing_commits)
        if check_pass.result:
            logger.info("Verified test passes using 'good' commits")
        else:
            logger.fatal("Could not reproduce success with 'good' commits")
            logger.fatal(check_pass.stdouterr)
            raise Exception("Could not reproduce success with 'good' commits")

    if skip_precondition_checks:
        logger.info("Skipping check that 'bad' commits reproduce failure")
    else:
        # Verify we can build successfully and that the test fails as expected.
        logger.info("Verifying test failure using 'bad' commits")
        failing_commits = {
            package: commit_list[-1][0] for package, commit_list in commits.items()
        }
        # Temporarily print DEBUG output to the console, so the IMPORTANT info message
        # below is actionable without checking the debug logfile.
        with console_log_level(logger, logging.DEBUG):
            check_fail = build_and_test(commits=failing_commits)
        if not check_fail.result:
            logger.info(
                "Verified test failure using 'bad' commits. IMPORTANT: you should check "
                "that the test output above shows the *expected* failure of your test "
                "case. It is very easy to accidentally provide a test case that fails "
                "for the wrong reason, which will not triage the correct issue!"
            )
        else:
            logger.fatal("Could not reproduce failure with 'bad' commits")
            logger.fatal(check_fail.stdouterr)
            raise Exception("Could not reproduce failure with 'bad' commits")

    # Make sure that the primary package (zeroth entry in `commits`) has
    # multiple commits. If it doesn't, we can permute it to the end straight
    # away as there is nothing to be done. We already asserted above that at
    # least one package has multiple commits.
    while len(next(iter(commits.values()))) == 1:
        commits.move_to_end(next(iter(commits.keys())))

    # Finally, start bisecting. The iteration order of `commits` defines the
    # algorithm: we start bisecting using the first package (e.g. XLA), and
    # take the oldest commits of the other packages that are newer than the
    # first package.
    primary, _ = next(iter(commits.items()))
    while len(commits[primary]) > 2:
        middle = len(commits[primary]) // 2
        bisect_commits = {}
        bisect_commits[primary], primary_date = commits[primary][middle]
        log_msg = f"Chose from {len(commits[primary])} remaining {primary} commits"
        # Find the oldest commits of the other packages that are newer, or the last commit
        indices = {primary: middle}
        for secondary, commit_list in _not_first(commits):
            for index, (commit, date) in enumerate(commit_list):
                if date >= primary_date:
                    break
            indices[secondary] = index
            bisect_commits[secondary] = commit
            log_msg += f", {len(commit_list)} remaining {secondary} commits"
        logger.info(log_msg)
        bisect_result = build_and_test(commits=bisect_commits).result

        if bisect_result:
            # Test passed, continue searching in the second half
            for package, index in indices.items():
                commits[package] = commits[package][index:]
        else:
            # Test failed, continue searching in the first half
            for package, index in indices.items():
                commits[package] = commits[package][: index + 1]

    # Primary bisection converged, meaning that there are two remaining
    # commits there, but possibly more of the other packages:
    #     pass        fail
    # PRI  pX -------- pZ
    # SEC  sX -- sY -- sZ
    # TER  tX --- tY - tZ
    # ...
    # If (pX, sZ, tZ, ...) passes, triage has converged: pZ is the culprit and
    # (sZ, tZ, ...) gives the reference commits of the other projects.
    #
    # Otherwise, if it fails, pZ is innocent and we can continue triaging
    # with the old primary package always fixed to pX.
    assert len(commits[primary]) == 2, commits
    blame_commits = {
        primary: commits[primary][0][0],  # pX
    }
    for secondary, commit_list in _not_first(commits):
        blame_commits[secondary] = commit_list[-1][0]  # sZ, tZ, ...
    logger.info(
        f"Two {primary} commits remain, checking if {commits[primary][-1][0]} is the "
        "culprit"
    )
    blame = build_and_test(commits=blame_commits)
    if blame.result:
        # Test passed with {pX, sZ, tZ, ...} but was known to fail with
        # {pZ, sZ, tZ, ...}. Therefore pZ is the culprit commit.
        (good_commit, _), (bad_commit, _) = commits[primary]
        log_str = f"Bisected failure to {primary} {bad_commit} with"
        for secondary, secondary_commit in _not_first(blame_commits):
            log_str += f" {secondary} {secondary_commit}"
        logger.info(log_str)
        return {
            f"{primary}_bad": bad_commit,
            f"{primary}_good": good_commit,
        } | {
            f"{secondary}_ref": secondary_commit
            for secondary, secondary_commit in _not_first(blame_commits)
        }
    else:
        # Test failed with both {pX, sZ, tZ, ...} and {pZ, sZ, tZ, ...}, so
        # we can fix the primary package to pX and recurse with the old
        # secondary package (s) as the new primary, and the old primary
        # package (p) moved to the end.
        commits[primary] = [commits.pop(primary)[0]]
        return commit_search(
            build_and_test=build_and_test,
            commits=commits,
            logger=logger,
            skip_precondition_checks=True,
        )
