import collections
import datetime
import functools
import hashlib
import itertools
import json
import logging
import pathlib
import subprocess
import time
import typing

from .args import compulsory_software, optional_software, parse_args
from .container import Container
from .docker import DockerContainer
from .logic import commit_search, container_search, TestResult
from .pyxis import PyxisContainer
from .utils import (
    container_url as container_url_base,
    get_logger,
    prepare_bazel_cache_mounts,
)


def get_commit(
    container: Container, logger: logging.Logger, repo: str
) -> typing.Tuple[str, str]:
    """
    Get the commit of the given repository that was used in the given nightly container

    Arguments:
    container: running container to extract the commit from
    repo: repository, must be jax or xla
    """
    # Older containers used /opt/jax-source etc.
    results = []
    for suffix in ["", "-source"]:
        dirname = f"/opt/{repo}{suffix}"
        result = container.exec(
            ["git", "rev-parse", "HEAD"],
            policy="once",
            stderr="separate",
            workdir=dirname,
        )
        stderr = result.stderr.strip()
        if len(stderr):
            logger.debug(f"stderr: {stderr}")
        results.append(result)
        if result.returncode == 0:
            commit = result.stdout.strip()
            if len(commit) == 40:
                return commit, dirname
    raise Exception(
        f"Could not extract commit of {repo} from {container}: {' '.join(map(str, results))}"
    )


def get_commits_and_dirs(
    worker: Container, logger: logging.Logger
) -> typing.Tuple[typing.Dict[str, str], typing.Dict[str, str]]:
    package_commits, dirs = {}, {}
    for package in compulsory_software:
        package_commits[package], dirs[package] = get_commit(worker, logger, package)
    for package in optional_software:
        try:
            package_commits[package], dirs[package] = get_commit(
                worker, logger, package
            )
        except Exception:
            pass
    return package_commits, dirs


def main() -> None:
    args = parse_args()
    bazel_cache_mounts = prepare_bazel_cache_mounts(args.bazel_cache)
    logger = get_logger(args.output_prefix)
    logger.info(
        "Verbose output, including stdout/err of triage commands, will be written to "
        f"{(args.output_prefix / 'debug.log').resolve()}"
    )

    def test_output_directory(
        url: str, commits: typing.Dict[str, str] = {}
    ) -> pathlib.Path:
        # Construct an output directory name, on the host, for output files written by
        # the test case.
        hash_chars = 8
        urlhash = f"container-{hashlib.sha1(url.encode()).hexdigest()[:hash_chars]}"
        out_dirname = "-".join(
            itertools.chain(
                [urlhash], map(lambda t: f"{t[0]}-{t[1][:hash_chars]}", commits.items())
            )
        )
        out_dir = args.output_prefix / out_dirname
        assert not out_dir.exists(), (
            f"{out_dir} should not already exist, maybe you are re-using {args.output_prefix}?"
        )
        out_dir.mkdir(mode=0o755)
        return out_dir.resolve()

    container_url = functools.partial(container_url_base, container=args.container)

    def Container(
        url, test_output_host_directory: typing.Optional[pathlib.Path] = None
    ):
        Imp = DockerContainer if args.container_runtime == "docker" else PyxisContainer
        mounts = bazel_cache_mounts + args.container_mount
        if test_output_host_directory is not None:
            # This can be used to save useful output from the test case (e.g. HLOs)
            mounts.append((test_output_host_directory, "/triage-tool-output"))
        return Imp(url, logger=logger, mounts=mounts)

    def get_commits(
        container_url: typing.Optional[str],
        explicit_commits: typing.Optional[typing.Dict[str, str]],
    ) -> typing.Tuple[typing.Dict[str, str], typing.Optional[typing.Dict[str, str]]]:
        """
        Get the list of commit hashes for different packages inside the given
        container, or return the explicitly passed set of hashes if they are
        given.

        If both arguments are non-None, the commits read from inside the given
        container will be overwritten by explicit passed entries.

        If is an error for both of the arguments to be None.
        """
        if explicit_commits is None:
            assert container_url is not None
            with Container(container_url) as worker:
                return get_commits_and_dirs(worker, logger)
        else:
            if container_url is None:
                return explicit_commits, None
            with Container(container_url) as worker:
                commits, package_dirs = get_commits_and_dirs(worker, logger)
            commits.update(explicit_commits)
            return commits, package_dirs

    def add_summary_record(
        section: str,
        record: typing.Mapping[str, typing.Union[bool, float, str]],
        scalar: bool = False,
    ):
        """
        Add a record to the output JSON file. This is intended to provide a useful record
        even in case of a fatal error.
        """
        summary_filename = args.output_prefix / "summary.json"
        try:
            with open(summary_filename, "r") as ifile:
                data = json.load(ifile)
        except FileNotFoundError:
            data = {}
        if scalar:
            if section in data:
                logging.warning(f"Overwriting summary data in section {section}")
            data[section] = record
        else:
            if section not in data:
                data[section] = []
            data[section].append(record)
        with open(summary_filename, "w") as ofile:
            json.dump(data, ofile)

    def check_container(date: datetime.date) -> TestResult:
        """
        See if the test passes in the given dated container.
        """
        before = time.monotonic()
        out_dir = test_output_directory(container_url(date))
        with Container(
            container_url(date), test_output_host_directory=out_dir
        ) as worker:
            commits, _ = get_commits_and_dirs(worker, logger)
            # This will stream interleaved stdout/stderr into the logger
            result = worker.exec(args.test_command)
            test_time = time.monotonic() - before
            test_pass = result.returncode == 0
            logger.info(
                f"Ran test case in {worker} in {test_time:.1f}s, pass={test_pass}"
            )
        add_summary_record(
            "container",
            {
                "container": container_url(date),
                "output_directory": out_dir.as_posix(),
                "result": test_pass,
                "test_time": test_time,
            }
            | commits,
        )
        return TestResult(
            host_output_directory=out_dir, result=test_pass, stdouterr=result.stdout
        )

    if args.passing_container is None and args.failing_container is None:
        # Search through the published containers, narrowing down to a pair of dates with
        # the property that the test passed on `range_start` and fails on `range_end`.
        range_start, range_end = container_search(
            container_exists=lambda date: Container(container_url(date)).exists(),
            container_passes=check_container,
            start_date=args.start_date,
            end_date=args.end_date,
            logger=logger,
            skip_precondition_checks=args.skip_precondition_checks,
            threshold_days=args.threshold_days,
        )
        passing_url = container_url(range_start)
        failing_url = container_url(range_end)
    else:
        # Skip the container-level search because at lease one explicit end point was
        # given
        passing_url = args.passing_container
        failing_url = args.failing_container

    # Get the git commits in the endpoint containers, or the explicitly passed commits
    passing_commits, passing_package_dirs = get_commits(
        passing_url, args.passing_commits
    )
    failing_commits, failing_package_dirs = get_commits(
        failing_url, args.failing_commits
    )

    # We should have commit hashes for all the same software packages at both
    # ends of the range, one way or another. TODO: this could be relaxed.
    assert passing_commits.keys() == failing_commits.keys(), (
        passing_commits,
        failing_commits,
    )

    # Choose a container to do the commit-level bisection in; use directory
    # names that match it.
    if failing_url is not None:
        bisection_url = failing_url
        package_dirs = failing_package_dirs
    else:
        assert passing_url is not None
        bisection_url = passing_url
        package_dirs = passing_package_dirs
    assert package_dirs is not None

    # Get the full lists of JAX/XLA commits and dates
    def get_commit_history(worker, start, end, dir):
        # In particular the end commit might not already be known if the older,
        # passing, container is being used for triage.
        commits_known = worker.exec(
            [
                "sh",
                "-c",
                f"git cat-file commit {start} && git cat-file commit {end}",
            ],
            policy="once_per_container",
            workdir=dir,
        )
        if commits_known.returncode != 0:
            worker.check_exec(
                ["git", "fetch"], policy="once_per_container", workdir=dir
            )
        result = worker.check_exec(
            [
                "git",
                "log",
                "--first-parent",
                "--reverse",
                "--format=%H %cI",
                f"{start}^..{end}",
            ],
            policy="once",
            stderr=subprocess.PIPE,
            workdir=dir,
        )
        logger.debug(f"stderr: {result.stderr.strip()}")
        data = []
        for line in result.stdout.splitlines():
            commit, date = line.split()
            date = datetime.datetime.fromisoformat(date).astimezone(
                datetime.timezone.utc
            )
            data.append((commit, date))
        return data

    # Fire up the container that will be used for the commit-level search and use it to
    # extract the relevant history of the repositories that will be triaged.
    with Container(bisection_url) as worker:
        packages = passing_commits.keys()
        log_str = "Bisecting"
        for package in packages:
            log_str += (
                f" {package} [{passing_commits[package]}, {failing_commits[package]}]"
            )
        log_str += f" using {worker}"
        logger.info(log_str)
        # Get lists of (commit_hash, commit_date) pairs
        package_commits = collections.OrderedDict()
        for package in packages:
            package_commits[package] = get_commit_history(
                worker,
                passing_commits[package],
                failing_commits[package],
                package_dirs[package],
            )
            # Confirm they're sorted by commit date
            assert all(
                b[1] >= a[1]
                for a, b in zip(package_commits[package], package_commits[package][1:])
            )
            # Confirm the end values are included as expected
            assert passing_commits[package] == package_commits[package][0][0]
            assert failing_commits[package] == package_commits[package][-1][0]

    def build_and_test(commits: typing.Dict[str, str]) -> TestResult:
        """
        The main body of the bisection loop. Update the JAX/XLA commits, build XLA and
        jaxlib, and run the test command. Throws on error when checking out or
        building, and returns the status of the test command.
        """
        # Amortise container startup overhead by batching together git commands
        git_commands, git_refs = [], []
        for package, commit in commits.items():
            git_refs.append(f"{package}@{commit}")
            git_commands += [
                f"cd {package_dirs[package]}",
                "git stash",
                f"git checkout {commit}",
            ]
        out_dir = test_output_directory(bisection_url, commits=commits)
        with Container(bisection_url, test_output_host_directory=out_dir) as worker:
            logger.info(f"Checking out {' '.join(git_refs)} in {worker}")
            worker.check_exec(
                ["sh", "-c", " && ".join(git_commands)],
                policy="once_per_container",
            )
            # Build JAX
            # TODO: teach the tool how to build TransformerEngine too
            # TODO: do not build JAX/XLA/TransformerEngine if we know their commits did not change?
            before = time.monotonic()
            # Unfortunately the build system does not always seem to handle incremental
            # rebuilds correctly.
            worker.check_exec(
                ["bazel", "clean", "--expunge"],
                policy="once_per_container",
                workdir=package_dirs["jax"],
            )
            build_jax = [
                "build-jax.sh",
                f"--bazel-cache={args.bazel_cache}",
            ]
            worker.check_exec(
                build_jax, policy="once_per_container", workdir=package_dirs["jax"]
            )
            middle = time.monotonic()
            logger.info(f"Build completed in {middle - before:.1f}s")
            # Run the test
            test_result = worker.exec(args.test_command)
            test_time = time.monotonic() - middle
        add_summary_record(
            "commit",
            {
                "build_time": middle - before,
                "container": bisection_url,
                "output_directory": out_dir.as_posix(),
                "result": test_result.returncode == 0,
                "test_time": test_time,
            }
            | commits,
        )
        result_str = "pass" if test_result.returncode == 0 else "fail"
        logger.info(f"Test completed in {test_time:.1f}s ({result_str})")
        return TestResult(
            host_output_directory=out_dir,
            result=test_result.returncode == 0,
            stdouterr=test_result.stdout,
        )

    # Run the commit-level bisection
    result, last_known_good, first_known_bad = commit_search(
        commits=package_commits,
        build_and_test=build_and_test,
        logger=logger,
        skip_precondition_checks=args.skip_precondition_checks,
    )

    def symlink(result: typing.Optional[TestResult], symlink_name: str) -> None:
        if result is None:
            return
        symlink = args.output_prefix / symlink_name
        assert not symlink.exists(), symlink
        assert result.host_output_directory.parent == args.output_prefix
        symlink.symlink_to(result.host_output_directory.name)

    symlink(last_known_good, "last_known_good")
    symlink(first_known_bad, "first_known_bad")
    result["container"] = failing_url
    add_summary_record("result", result, scalar=True)
