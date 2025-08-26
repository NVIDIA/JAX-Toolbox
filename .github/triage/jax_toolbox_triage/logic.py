from dataclasses import dataclass
import datetime
from enum import auto, Enum
import functools
import itertools
import logging
import pathlib
import typing


class TestExecutionOutcome(Enum):
    """
    Enumerate the possible outcomes of a build + test run. This might be extended in
    future to allow test cases to return more nuanced results, e.g. TEST_DOES_NOT_EXIST
    to avoid searching back before the introduction of a test case.
    """

    __test__ = False  # stop pytest gathering this
    BUILD_FAILURE = auto()
    TEST_FAILURE = auto()
    TEST_SUCCESS = auto()


@dataclass
class TestResult:
    """
    Hold the pass/fail result and the interleaved stdout/stderr of a test run.

    `build_stdouterr` can be None if there was no build, e.g. running the test case in
    an existing container.
    `stdouterr` can be None if the build did not succeed, so the test was not run.
    """

    __test__ = False  # stop pytest gathering this
    build_stdouterr: typing.Optional[str]
    host_output_directory: pathlib.Path
    result: TestExecutionOutcome
    stdouterr: typing.Optional[str]


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
        if test_end_date.result != TestExecutionOutcome.TEST_FAILURE:
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
        while container_passes(search_date).result != TestExecutionOutcome.TEST_SUCCESS:
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
        if container_passes(range_mid).result == TestExecutionOutcome.TEST_SUCCESS:
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


def _get_versions(
    *,
    logger: logging.Logger,
    primary: str,
    primary_index: typing.Optional[int] = None,
    versions: typing.OrderedDict[
        str, typing.Sequence[typing.Tuple[str, datetime.datetime]]
    ],
) -> typing.Tuple[typing.Dict[str, str], typing.Dict[str, int]]:
    if primary_index is None:
        primary_index = len(versions[primary]) // 2
        log_msg = f"Chose from {len(versions[primary])} remaining {primary} versions"
        log_msg += f" [{versions[primary][0][0]}, {versions[primary][-1][0]}]"
    else:
        log_msg = None
    bisect_versions = {}
    bisect_versions[primary], primary_date = versions[primary][primary_index]
    # Find the oldest versions of the other packages that are newer, or the last version
    indices = {primary: primary_index}
    for secondary, version_list in _not_first(versions):
        for index, (version, date) in enumerate(version_list):
            if date >= primary_date:
                break
        indices[secondary] = index
        bisect_versions[secondary] = version
        if log_msg is not None and len(version_list) > 1:
            log_msg += f", {len(version_list)} remaining {secondary} versions"
            log_msg += f" [{versions[secondary][0][0]}, {versions[secondary][-1][0]}]"
    if log_msg is not None:
        logger.info(log_msg)
    return bisect_versions, indices


def _earliest_versions(
    versions: typing.OrderedDict[
        str, typing.Sequence[typing.Tuple[str, datetime.datetime]]
    ],
) -> typing.Dict[str, str]:
    return {package: version_list[0][0] for package, version_list in versions.items()}


def _latest_versions(
    versions: typing.OrderedDict[
        str, typing.Sequence[typing.Tuple[str, datetime.datetime]]
    ],
) -> typing.Dict[str, str]:
    return {package: version_list[-1][0] for package, version_list in versions.items()}


def _strip_build_failures(versions: typing.Dict[str, str]) -> typing.Dict[str, str]:
    def remove_build_failures(ver):
        ver_bits = ver.split(",")
        if len(ver_bits) == 1:
            return ver
        elif ver_bits[1][0] == "[" and ver_bits[-1][-1] == "]":
            # v1,[v2,v3] -> v1
            return ver_bits[0]
        else:
            # [v1,v2],v3 -> v3
            assert ver_bits[0][0] == "[" and ver_bits[-2][-1] == "]", ver_bits
            return ver_bits[-1]

    return {k: remove_build_failures(v) for k, v in versions.items()}


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
        return tuple(sorted(_strip_build_failures(versions).items()))

    if skip_precondition_checks:
        logger.info("Skipping check that 'good' versions reproduce success")
    else:
        # Verify that we can build successfully and that the test succeeds as expected.
        logger.info("Verifying test success using 'good' versions")
        passing_versions = _earliest_versions(versions)
        check_pass = build_and_test(versions=passing_versions)
        assert _cache_key(passing_versions) not in result_cache
        result_cache[_cache_key(passing_versions)] = check_pass
        if check_pass.result == TestExecutionOutcome.TEST_SUCCESS:
            logger.info("Verified test passes using 'good' versions")
        else:
            err = f"Could not reproduce success with 'good' versions ({check_pass.result})"
            logger.fatal(err)
            logger.fatal(check_pass.stdouterr)
            raise Exception(err)

    if skip_precondition_checks:
        logger.info("Skipping check that 'bad' versions reproduce failure")
    else:
        # Verify we can build successfully and that the test fails as expected.
        logger.info("Verifying test failure using 'bad' versions")
        failing_versions = _latest_versions(versions)
        check_fail = build_and_test(
            versions=failing_versions, test_output_log_level=logging.INFO
        )
        assert _cache_key(failing_versions) not in result_cache
        result_cache[_cache_key(failing_versions)] = check_fail
        if check_fail.result == TestExecutionOutcome.TEST_FAILURE:
            logger.info(
                "Verified test failure using 'bad' versions. IMPORTANT: you should "
                "check that the test output above shows the *expected* failure of your "
                "test case. It is very easy to accidentally provide a test case that "
                "fails for the wrong reason, which will not triage the correct issue!"
            )
        else:
            err = (
                f"Could not reproduce failure with 'bad' versions ({check_fail.result})"
            )
            logger.fatal(err)
            logger.fatal(check_fail.stdouterr)
            raise Exception(err)

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
    get_versions = functools.partial(_get_versions, logger=logger, primary=primary)

    def build_cached(bisect_versions):
        cache_key = _cache_key(bisect_versions)
        bisect_result = result_cache.get(cache_key)
        if bisect_result is not None:
            return bisect_result
        bisect_result = build_and_test(versions=_strip_build_failures(bisect_versions))
        result_cache[cache_key] = bisect_result
        return bisect_result

    def find_successful_build(versions):
        # Try to find a set of versions where the build does not fail, starting with
        # the set of versions that implement in a binary search for the actual test failure
        bisect_versions, indices = get_versions(versions=versions)
        bisect_result = build_cached(bisect_versions)
        if bisect_result.result != TestExecutionOutcome.BUILD_FAILURE:
            return bisect_result, bisect_versions
        logger.info(
            f"Encountered build failure using index={indices[primary]} of "
            f"the remaining {primary} versions"
        )
        # Versions based on the middle of the remaining `primary` commits led to a
        # build failure (n), but the endpoints build OK (y).
        # y -- ? -- n -- ? -- y
        # |         |         |
        # start   middle     end
        #
        # We may know about other build failures (n values) in the range due to
        # previous iterations.
        #       cached
        #         |
        # y???nnnnnnnnn???????????y
        # |           |           |
        # start     middle       end
        #
        # And the hope of this logic is that we will find either an early version where
        # the build succeeds and the test fails (so we can throw away the range with
        # build failures), or a late version where the build succeeds and the test
        # succeeds so we can do the same. The somewhat arbitrary logic here is to take
        # the shorter y??????n run and refine it in the hope one of the ? is a y.
        assert (
            result_cache.get(_cache_key(_earliest_versions(versions))).result
            == TestExecutionOutcome.TEST_SUCCESS
        )
        build_statuses = [(True, {p: 0 for p in versions})]
        for n in range(1, len(versions[primary]) - 1):
            versions_n, indices_n = get_versions(primary_index=n, versions=versions)
            result_n = result_cache.get(_cache_key(versions_n))
            assert (
                result_n is None
                or result_n.result == TestExecutionOutcome.BUILD_FAILURE
            )
            build_statuses.append((None if result_n is None else False, indices_n))
        assert (
            result_cache.get(_cache_key(_latest_versions(versions))).result
            == TestExecutionOutcome.TEST_FAILURE
        )
        build_statuses.append((True, {p: len(vs) - 1 for p, vs in versions.items()}))
        assert len(build_statuses) == len(versions[primary])
        # build_statuses is something like [True, None, False, None, True]; identify
        # the ranges [b, None, ..., not b] i.e [0, 2] and [2, 4] in the example, and do
        # binary searches in them
        start, start_v = 0, True
        ranges = []
        for n, (v, _) in enumerate(build_statuses):
            if v == start_v:
                start = n
            elif v is not None and v != start_v:
                if n - start > 1:
                    # Need some None values in betwen
                    ranges.append((start, n))
                start, start_v = n, v
        assert len(ranges) <= 2
        # Start with the narrower range
        ranges.sort(key=lambda v: v[1] - v[0])
        for start, end in ranges:
            start_status, start_indices = build_statuses[start]
            end_status, end_indices = build_statuses[end]
            assert start_status is not None
            assert end_status is not None
            assert start_status != end_status
            assert all(v is None for v, _ in build_statuses[start + 1 : end - 1])
            range_versions = {
                package: versions[package][start_index : end_indices[package] + 1]
                for package, start_index in start_indices.items()
            }
            while len(range_versions[primary]) > 2:
                bisect_versions, indices = get_versions(versions=range_versions)
                bisect_result = build_cached(bisect_versions)
                if bisect_result.result != TestExecutionOutcome.BUILD_FAILURE:
                    return bisect_result, bisect_versions
                logger.info(
                    "Encountered another build failure when refining "
                    f"{len(range_versions[primary])} {primary} versions in "
                    f"{len(ranges)} candidate range(s)"
                )

                def _slice(index):
                    # range was (Y??n if start_status else n??Y) and a ? turned out to be an n
                    return (
                        slice(None, index + 1) if start_status else slice(index, None)
                    )

                range_versions = {
                    package: range_versions[package][_slice(index)]
                    for package, index in indices.items()
                }
        return bisect_result, bisect_versions

    while len(versions[primary]) > 2:
        bisect_result, bisect_versions = find_successful_build(versions=versions)

        # reconstruct the indices in `versions` of the versions in `bisect_versions`
        def _index(pkg, ver):
            for n, (v, _) in enumerate(versions[pkg]):
                if v == ver:
                    return n
            assert False

        indices = {pkg: _index(pkg, ver) for pkg, ver in bisect_versions.items()}
        if bisect_result.result == TestExecutionOutcome.TEST_SUCCESS:
            # Test passed, continue searching in the second half
            for package, index in indices.items():
                versions[package] = versions[package][index:]
        elif bisect_result.result == TestExecutionOutcome.TEST_FAILURE:
            # Test failed, continue searching in the first half
            for package, index in indices.items():
                versions[package] = versions[package][: index + 1]
        else:
            assert bisect_result.result == TestExecutionOutcome.BUILD_FAILURE, (
                bisect_result
            )
            # Did not succeed in finding a version of `primary` that builds. This does
            # not quite mean that all versions fail, as the algorithm will not try all
            # versions in ranges with failures at both ends
            #
            #       might
            #          \
            # Y n n n Y n n n n n n n Y (build passes Y/n)
            # |     |     |           |
            # start Q1  middle       end
            #
            # as given middle=fail, and Q1=fail the points between Q1 and middle will
            # not be checked.
            n_primary = len(versions[primary])
            logger.info(n_primary)
            build_fail_commits = []
            for n in range(1, n_primary - 1):
                versions_n, _ = get_versions(primary_index=n, versions=versions)
                # Should have been a build failure if tested.
                result_n = result_cache.get(_cache_key(versions_n))
                if result_n is not None:
                    assert result_n.result == TestExecutionOutcome.BUILD_FAILURE, (
                        result_n
                    )
                build_fail_commits.append(versions_n[primary])
            logger.warning(
                f"Could not triage {primary} to a single version due to build "
                f"failures, adding {n_primary - 2} build failure version(s) to both "
                "last-known-good and first-known-bad versions to represent this lack "
                "of signal. The tool has not positively verified that *all* of these "
                "commits actually lead to build failures."
            )
            test_pass_commit, test_pass_date = versions[primary][0]
            test_fail_commit, test_fail_date = versions[primary][-1]
            test_pass_range = f"{test_pass_commit},[{','.join(build_fail_commits)}]"
            test_fail_range = f"[{','.join(build_fail_commits)}],{test_fail_commit}"
            versions[primary] = [
                (test_pass_range, test_pass_date),
                (test_fail_range, test_fail_date),
            ]

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
    (good_version, _), (bad_version, _) = versions[primary]
    blame_versions = _latest_versions(versions)  # [pZ,] sZ, tZ, ...
    blame_versions[primary] = good_version  # pX
    logger.info(
        f"Two {primary} versions remain, checking if {versions[primary][-1][0]} is the "
        "culprit"
    )
    # It's possible that this combination has already been tested at this point
    blame = build_cached(blame_versions)
    if blame.result == TestExecutionOutcome.TEST_SUCCESS:
        # Test passed with {pX, sZ, tZ, ...} but was known to fail with
        # {pZ, sZ, tZ, ...}. Therefore pZ is the culprit version.
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
        assert blame.result == TestExecutionOutcome.TEST_FAILURE, blame.result
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
