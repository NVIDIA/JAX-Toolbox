from dataclasses import dataclass
import datetime
import functools
import logging
import typing


@dataclass
class TestResult:
    """
    Hold the result/stdout/stderr of a test execution
    """

    __test__ = False  # stop pytest gathering this
    result: bool
    stdout: typing.Optional[str] = None
    stderr: typing.Optional[str] = None


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
        test_end_date = container_passes(end_date)
        logger.info(f"stdout: {test_end_date.stdout}")
        logger.info(f"stderr: {test_end_date.stderr}")
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
    def __call__(
        self, *, jax_commit: str, xla_commit: str
    ) -> typing.Tuple[bool, str, str]: ...


def commit_search(
    *,
    jax_commits: typing.Sequence[typing.Tuple[str, datetime.datetime]],
    xla_commits: typing.Sequence[typing.Tuple[str, datetime.datetime]],
    build_and_test: BuildAndTest,
    logger: logging.Logger,
    skip_precondition_checks: bool,
):
    """
    build_and_test: test the given commits in the container that originally shipped with end_{jax,xla}_commit.
    """
    if (
        len(jax_commits) == 0
        or len(xla_commits) == 0
        or len(jax_commits) + len(xla_commits) < 3
    ):
        raise Exception("Not enough commits passed")
    start_jax_commit = jax_commits[0][0]
    start_xla_commit = xla_commits[0][0]
    end_jax_commit = jax_commits[-1][0]
    end_xla_commit = xla_commits[-1][0]
    if skip_precondition_checks:
        logger.info("Skipping check that vanilla rebuild + test reproduces failure")
    else:
        # Verify we can build successfully and that the test fails as expected. These
        # commits are the ones already checked out in the container, but specifying
        # them explicitly is good for the summary JSON.
        logger.info("Building in the range-ending container...")
        range_end_result, stdout, stderr = build_and_test(
            jax_commit=end_jax_commit, xla_commit=end_xla_commit
        )
        if not range_end_result:
            logger.info("Verified test failure after vanilla rebuild")
        else:
            logger.fatal("Vanilla rebuild did not reproduce test failure")
            logger.fatal(stdout)
            logger.fatal(stderr)
            raise Exception(
                "Could not reproduce failure after rebuild in 'bad' container"
            )

    # Verify that we can build the commit at the start of the range and reproduce the
    # test success there in the end-of-range container.
    range_start_result, stdout, stderr = build_and_test(
        jax_commit=start_jax_commit, xla_commit=start_xla_commit
    )
    if range_start_result:
        logger.info(
            "Test passed after rebuilding commits from start container in end container"
        )
    else:
        logger.fatal(
            "Test failed after rebuilding commits from start container in end container"
        )
        logger.fatal(stdout)
        logger.fatal(stderr)
        raise Exception(
            "Could not reproduce success with 'good' commits in 'bad' container"
        )

    # Finally, start bisecting. This is XLA-centric; JAX is moved too but is secondary.
    while len(xla_commits) > 2:
        middle = len(xla_commits) // 2
        xla_hash, xla_date = xla_commits[middle]
        # Find the oldest JAX commit that is newer than this
        for jax_index, (jax_hash, jax_date) in enumerate(jax_commits):
            if jax_date >= xla_date:
                break
        bisect_result, _, _ = build_and_test(jax_commit=jax_hash, xla_commit=xla_hash)
        if bisect_result:
            # Test passed, continue searching in the second half
            xla_commits = xla_commits[middle:]
            jax_commits = jax_commits[jax_index:]
        else:
            # Test failed, continue searching in the first half
            xla_commits = xla_commits[: middle + 1]
            jax_commits = jax_commits[: jax_index + 1]

    # XLA bisection converged. xla_commits has two entries. jax_commits may be a little
    # longer, if it was more active than XLA at the relevant time. For example, here
    # xla_commits is {oX, nX} and jax_commits is {oJ, mJ, nJ}, and the test passes with
    # {oX, oJ} and fails with {nX, nJ}. Naming: o=old, m=medium, n=new, X=XLA, J=JAX.
    #     pass        fail
    # XLA: oX -------- nX
    # JAX: oJ -- mJ -- nJ
    #
    # To figure out whether to blame XLA or JAX, we now test {oX, nJ}.
    old_xla_hash = xla_commits[0][0]
    new_jax_hash = jax_commits[-1][0]
    blame_result, _, _ = build_and_test(
        jax_commit=new_jax_hash, xla_commit=old_xla_hash
    )
    if blame_result:
        # Test passed with {oX, nJ} but was known to fail with {nX, nJ}. Therefore, XLA
        # commit nX is responsible and JAX is innocent.
        results = (old_xla_hash, xla_commits[1][0])
        logger.info(
            "Bisected failure to XLA {}..{} with JAX {}".format(*results, new_jax_hash)
        )
        return {
            "jax_ref": new_jax_hash,
            "xla_bad": xla_commits[1][0],
            "xla_good": old_xla_hash,
        }
    else:
        # Test failed with {oX, nJ} but was known to pass with {oX, oJ}, so JAX is
        # responsible and we should bisect between oJ (pass) and nJ (fail). This yields
        # a single JAX commit to blame, either mJ or nJ in the example above.
        while len(jax_commits) > 2:
            middle = len(jax_commits) // 2
            jax_hash, _ = jax_commits[middle]
            bisect_result, _, _ = build_and_test(
                jax_commit=jax_hash, xla_commit=old_xla_hash
            )
            if bisect_result:
                # Test passsed, continue searching in second half
                jax_commits = jax_commits[middle:]
            else:
                # Test failed, continue searching in the first half
                jax_commits = jax_commits[: middle + 1]
        results = (jax_commits[0][0], jax_commits[1][0])
        logger.info(
            "Bisected failure to JAX {}..{} with XLA {}".format(*results, old_xla_hash)
        )
        return {
            "jax_bad": jax_commits[1][0],
            "jax_good": jax_commits[0][0],
            "xla_ref": old_xla_hash,
        }
