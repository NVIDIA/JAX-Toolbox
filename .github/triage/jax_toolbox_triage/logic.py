from dataclasses import dataclass
import datetime
import functools
import itertools
import logging
import pathlib
import typing


@dataclass
class TestResult:
    """
    Hold the pass/fail result and the interleaved stdout/stderr of a test run
    """

    __test__ = False  # stop pytest gathering this
    host_output_directory: pathlib.Path
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


class DateTester(typing.Protocol):
    def __call__(
        self, date: datetime.date, *, test_output_log_level: int = logging.DEBUG
    ) -> TestResult:
        """
        Given a date, return the test result as obtained from the nightly container on
        that date.
        """
        ...


def container_search(
    *,
    container_exists: typing.Callable[[datetime.date], bool],
    container_passes: DateTester,
    start_date: typing.Optional[datetime.date],
    end_date: typing.Optional[datetime.date],
    logger: logging.Logger,
    skip_precondition_checks: bool,
    threshold_days: int,
) -> typing.Tuple[datetime.date, datetime.date]:
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
        logger.info(f"Searching for a valid container date close to {now}")
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
        test_end_date = container_passes(end_date, test_output_log_level=logging.INFO)
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
        self,
        *,
        versions: typing.Dict[str, str],
        test_output_log_level: int = logging.DEBUG,
    ) -> TestResult:
        """
        Given an [unordered] set of {package_name: package_version}, build
        those package versions and return the test result.
        """
        ...


T = typing.TypeVar("T")
U = typing.TypeVar("U")
FlatVersionDict = typing.Tuple[typing.Tuple[str, str], ...]


def _first(xs: typing.Iterable[T]) -> T:
    return next(iter(xs))


def _not_first(d: typing.Dict[T, U]) -> typing.Iterable[typing.Tuple[T, U]]:
    return itertools.islice(d.items(), 1, None)


def _version_search(
    *,
    versions: typing.OrderedDict[
        str, typing.Sequence[typing.Tuple[str, datetime.datetime]]
    ],
    build_and_test: BuildAndTest,
    logger: logging.Logger,
    skip_precondition_checks: bool,
    result_cache: typing.Dict[FlatVersionDict, TestResult],
) -> typing.Tuple[typing.Dict[str, str], TestResult, typing.Optional[TestResult]]:
    assert all(len(version_list) for version_list in versions.values()), (
        "Not enough versions: need at least one version for each package",
        versions,
    )
    assert sum(map(len, versions.values())) > len(versions), (
        "Not enough versions: need multiple versions for at least one package",
        versions,
    )

    def _cache_key(
        versions: typing.Dict[str, str],
    ) -> FlatVersionDict:
        return tuple(sorted(versions.items()))

    if skip_precondition_checks:
        logger.info("Skipping check that 'good' versions reproduce success")
    else:
        # Verify that we can build successfully and that the test succeeds as expected.
        logger.info("Verifying test success using 'good' versions")
        passing_versions = {
            package: version_list[0][0] for package, version_list in versions.items()
        }
        check_pass = build_and_test(versions=passing_versions)
        assert _cache_key(passing_versions) not in result_cache
        result_cache[_cache_key(passing_versions)] = check_pass
        if check_pass.result:
            logger.info("Verified test passes using 'good' versions")
        else:
            logger.fatal("Could not reproduce success with 'good' versions")
            logger.fatal(check_pass.stdouterr)
            raise Exception("Could not reproduce success with 'good' versions")

    if skip_precondition_checks:
        logger.info("Skipping check that 'bad' versions reproduce failure")
    else:
        # Verify we can build successfully and that the test fails as expected.
        logger.info("Verifying test failure using 'bad' versions")
        failing_versions = {
            package: version_list[-1][0] for package, version_list in versions.items()
        }
        check_fail = build_and_test(
            versions=failing_versions, test_output_log_level=logging.INFO
        )
        assert _cache_key(failing_versions) not in result_cache
        result_cache[_cache_key(failing_versions)] = check_fail
        if not check_fail.result:
            logger.info(
                "Verified test failure using 'bad' versions. IMPORTANT: you should "
                "check that the test output above shows the *expected* failure of your "
                "test case. It is very easy to accidentally provide a test case that "
                "fails for the wrong reason, which will not triage the correct issue!"
            )
        else:
            logger.fatal("Could not reproduce failure with 'bad' versions")
            logger.fatal(check_fail.stdouterr)
            raise Exception("Could not reproduce failure with 'bad' versions")

    # Make sure that the primary package (zeroth entry in `versions`) has
    # multiple versions. If it doesn't, we can permute it to the end straight
    # away as there is nothing to be done. We already asserted above that at
    # least one package has multiple versions.
    while len(_first(versions.values())) == 1:
        versions.move_to_end(_first(versions.keys()))

    # Finally, start bisecting. The iteration order of `versions` defines the
    # algorithm: we start bisecting using the first package (e.g. XLA), and
    # take the oldest versions of the other packages that are newer than the
    # first package.
    primary = _first(versions.keys())
    while len(versions[primary]) > 2:
        middle = len(versions[primary]) // 2
        bisect_versions = {}
        bisect_versions[primary], primary_date = versions[primary][middle]
        log_msg = f"Chose from {len(versions[primary])} remaining {primary} versions"
        log_msg += f" [{versions[primary][0][0]}, {versions[primary][-1][0]}]"
        # Find the oldest versions of the other packages that are newer, or the last version
        indices = {primary: middle}
        for secondary, version_list in _not_first(versions):
            for index, (version, date) in enumerate(version_list):
                if date >= primary_date:
                    break
            indices[secondary] = index
            bisect_versions[secondary] = version
            if len(version_list) > 1:
                log_msg += f", {len(version_list)} remaining {secondary} versions"
                log_msg += (
                    f" [{versions[secondary][0][0]}, {versions[secondary][-1][0]}]"
                )
        logger.info(log_msg)
        bisect_result = build_and_test(versions=bisect_versions)
        assert _cache_key(bisect_versions) not in result_cache
        result_cache[_cache_key(bisect_versions)] = bisect_result

        if bisect_result.result:
            # Test passed, continue searching in the second half
            for package, index in indices.items():
                versions[package] = versions[package][index:]
        else:
            # Test failed, continue searching in the first half
            for package, index in indices.items():
                versions[package] = versions[package][: index + 1]

    # Primary bisection converged, meaning that there are two remaining
    # versions there, but possibly more of the other packages:
    #     pass        fail
    # PRI  pX -------- pZ
    # SEC  sX -- sY -- sZ
    # TER  tX --- tY - tZ
    # ...
    # If (pX, sZ, tZ, ...) passes, triage has converged: pZ is the culprit and
    # (sZ, tZ, ...) gives the reference versions of the other projects.
    #
    # Otherwise, if it fails, pZ is innocent and we can continue triaging
    # with the old primary package always fixed to pX.
    assert len(versions[primary]) == 2, versions
    blame_versions = {
        primary: versions[primary][0][0],  # pX
    }
    for secondary, version_list in _not_first(versions):
        blame_versions[secondary] = version_list[-1][0]  # sZ, tZ, ...
    logger.info(
        f"Two {primary} versions remain, checking if {versions[primary][-1][0]} is the "
        "culprit"
    )
    # It's possible that this combination has already been tested at this point
    blame = result_cache.get(_cache_key(blame_versions))
    if blame is None:
        blame = build_and_test(versions=blame_versions)
        result_cache[_cache_key(blame_versions)] = blame
    if blame.result:
        # Test passed with {pX, sZ, tZ, ...} but was known to fail with
        # {pZ, sZ, tZ, ...}. Therefore pZ is the culprit version.
        (good_version, _), (bad_version, _) = versions[primary]
        log_str = f"Bisected failure to {primary} {bad_version} with"
        for secondary, secondary_version in _not_first(blame_versions):
            log_str += f" {secondary} {secondary_version}"
        logger.info(log_str)
        ret = {
            f"{primary}_bad": bad_version,
            f"{primary}_good": good_version,
        }
        first_known_bad = {primary: bad_version}
        for secondary, secondary_version in _not_first(blame_versions):
            first_known_bad[secondary] = secondary_version
            ret[f"{secondary}_ref"] = secondary_version
        # `blame` represents the last-known-good test result, first-known-bad was seen
        # earlier, or possibly not at all e.g. if `skip_precondition_checks` is True
        # and first-known-bad was the end of the search range.
        first_known_bad_result = result_cache.get(_cache_key(first_known_bad))
        if first_known_bad_result is None:
            if skip_precondition_checks:
                logger.info(
                    "Did not find a cached result for the first-known-bad "
                    f"configuration {first_known_bad}, this is probably due to "
                    "--skip-precondition-checks having been passed."
                )
            else:
                logger.error(
                    "Did not find a cached result for the first-known-bad "
                    f"configuration {first_known_bad}, this is unexpected!"
                )
        return ret, blame, first_known_bad_result
    else:
        # Test failed with both {pX, sZ, tZ, ...} and {pZ, sZ, tZ, ...}, so
        # we can fix the primary package to pX and recurse with the old
        # secondary package (s) as the new primary, and the old primary
        # package (p) moved to the end.
        versions[primary] = [versions.pop(primary)[0]]
        return _version_search(
            build_and_test=build_and_test,
            versions=versions,
            logger=logger,
            skip_precondition_checks=True,
            result_cache=result_cache,
        )


def version_search(
    *,
    versions: typing.OrderedDict[
        str, typing.Sequence[typing.Tuple[str, datetime.datetime]]
    ],
    build_and_test: BuildAndTest,
    logger: logging.Logger,
    skip_precondition_checks: bool,
) -> typing.Tuple[
    typing.Dict[str, str],
    TestResult,
    typing.Optional[TestResult],
]:
    """
    Bisect a failure back to a single version of a single component.

    Arguments:
    versions: *ordered* dictionary of version sequences for different software
        packages, e.g. versions["jax"][0] is (commit_hash, date) of the passing JAX
        version. The ordering of packages has implications for precisely how
        the triage proceeds.
    build_and_test: callable that tests if a given vector of versions passes
    logger: instance to log output to
    skip_precondition_checks: if True, some tests that should pass/fail by
        construction are skipped

    Returns a 3-tuple of (summary_dict, last_known_good, first_known_bad),
    where the last element can be None if skip_precondition_checks=True. The
    last two elements' .result fields will always be, respectively, True and
    False, but the other fields can be used to obtain stdout+stderr and
    output files from those test invocations.
    """
    return _version_search(
        versions=versions,
        build_and_test=build_and_test,
        logger=logger,
        skip_precondition_checks=skip_precondition_checks,
        result_cache={},
    )
