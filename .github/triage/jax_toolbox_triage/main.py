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
from .local import LocalContainer
from .logic import container_search, TestResult, version_search
from .pyxis import PyxisContainer
from .utils import (
    container_url as container_url_base,
    get_logger,
    prepare_bazel_cache_mounts,
)


def get_env(worker: Container) -> typing.Dict[str, str]:
    """
    Get the runtime environment in the given container.

    Returns: {env_var: value} dictionary, sorted by key.
    """

    def impl() -> typing.Dict[str, str]:
        kvs = (
            worker.check_exec(["env", "-0"], policy="once", stderr="separate")
            .stdout[:-1]  # skip the trailing \0
            .split("\0")
        )
        return dict(kv.split("=", 1) for kv in kvs)

    # Remove any environment variables that differ between consecutive `env` calls, for
    # example some step-specific Slurm variables.
    env1, env2 = impl(), impl()
    # sorted(...) for run-to-run determinism
    return {k: env1[k] for k in sorted(env1.keys() & env2.keys()) if env1[k] == env2[k]}


def get_commits_and_dirs(
    worker: Container,
) -> typing.Tuple[typing.Dict[str, str], typing.Dict[str, str]]:
    """
    Get the git repository paths and current HEAD commits in the given environment of
    the software packages named in `compulsory_software` and `optional_software`.

    Returns: ({package: commit}, {package: directory})
    """
    # Formulated this way to avoid paying too many times for container startup.
    cmds = []
    for package in compulsory_software + optional_software:
        bits = [
            f"(cd /opt/{package} && git rev-parse HEAD && echo {package} && pwd)",
            f"(cd /opt/{package}-source && git rev-parse HEAD && echo {package} && pwd)",
        ]
        if package in optional_software:
            bits.append("true")
        cmds.append(f"({' || '.join(bits)})")
    result = worker.check_exec(
        ["sh", "-c", " && ".join(cmds)], policy="once", stderr="separate"
    )
    versions, dirs = {}, {}
    # Look over triplets of output lines
    for commit, package, dirname in zip(*([iter(result.stdout.splitlines())] * 3)):
        dirs[package] = dirname
        versions[package] = commit
    return versions, dirs


def get_versions_dirs_env(
    worker: Container,
    versions_from_env: bool,
) -> typing.Tuple[typing.Dict[str, str], typing.Dict[str, str], typing.Dict[str, str]]:
    """
    Get software versions in the given [container] environment, git repository paths
    where relevant, and the runtime environment.

    The list of software versions is drawn from git repositories at known container
    locations and, if `versions_from_env` is True, from the environment.

    Returns:
      versions: {package: version or commit},
      dirs: {package: git_repository_dir}
      env: {env_var: value}
    """
    # Get the git repository paths and commits from the container.
    versions, dirs = get_commits_and_dirs(worker)

    # Get the environment variables from the container.
    env = get_env(worker)

    if versions_from_env:
        # Promote any XXX_VERSION environment variables into `versions` if `XXX` is
        # not already there.
        for k, v in env.items():
            if not len(v) or not k.endswith("_VERSION"):
                continue
            package = k[:-8]
            assert package not in versions, (versions, package)
            versions[package] = v
    return versions, dirs, env


def main() -> None:
    args = parse_args()
    bazel_cache_mounts = prepare_bazel_cache_mounts(args.bazel_cache)
    logger = get_logger(args.output_prefix)
    logger.info("Arguments:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info(
        "Verbose output, including stdout/err of triage commands, will be written to "
        f"{(args.output_prefix / 'debug.log').resolve()}"
    )

    def test_output_directory(
        url: str, versions: typing.Dict[str, str] = {}
    ) -> pathlib.Path:
        # Construct an output directory name, on the host, for output files written by
        # the test case.
        hash_chars = 8
        urlhash = f"container-{hashlib.sha1(url.encode()).hexdigest()[:hash_chars]}"
        out_dirname = "-".join(
            itertools.chain(
                [urlhash],
                map(lambda t: f"{t[0]}-{t[1][:hash_chars]}", sorted(versions.items())),
            )
        )
        out_dir = args.output_prefix / out_dirname
        assert not out_dir.exists(), (
            f"{out_dir} should not already exist, maybe you are re-using {args.output_prefix}?"
        )
        out_dir.mkdir(mode=0o755)
        return out_dir.resolve()

    container_url = functools.partial(
        container_url_base,
        container=args.container,
        template=args.container_url_template,
    )

    def Container(
        url, test_output_host_directory: typing.Optional[pathlib.Path] = None
    ):
        if args.container_runtime == "local":
            return LocalContainer(logger=logger)

        Imp = DockerContainer if args.container_runtime == "docker" else PyxisContainer
        mounts = bazel_cache_mounts + args.container_mount
        if test_output_host_directory is not None:
            # This can be used to save useful output from the test case (e.g. HLOs)
            mounts.append((test_output_host_directory, "/triage-tool-output"))
        return Imp(url, logger=logger, mounts=mounts)

    def get_versions(
        container_url: typing.Optional[str],
        explicit_versions: typing.Optional[typing.Dict[str, str]],
        versions_from_env: bool,
    ) -> typing.Tuple[
        typing.Dict[str, str],
        typing.Optional[typing.Dict[str, str]],
        typing.Optional[typing.Dict[str, str]],
        typing.Optional[typing.Dict[str, str]],
    ]:
        """
        Given an optional container URL (e.g. --failing-container) and an optional set
        of overriden versions (e.g. --failing-versions), obtain a list of software
        versions to bookend the triage range.

        Also returns the container's runtime environment variables for diagnostic purposes.

        If *only* overrides are given, those are returned verbatim and all other return
        values are None.

        Otherwise, the software versions, git repository directories and runtime
        environment are extracted from the given container. If overrides are *also* given,
        these take precedence over the extracted values.

        It is an error for both `container_url` and `explicit_versions` to be None.

        Returns:
        versions: {packge: version} mapping defining one end of the triage range
        url_versions: {package: version} corresponding to the given container (or None)
        dirs: {package: git_repository_dir} (or None)
        env: {env_var: value} (or None)
        """
        if explicit_versions is not None and container_url is None:
            return explicit_versions, None, None, None
        assert container_url is not None
        logger.info(f"Extracting versions from {container_url} ...")
        with Container(container_url) as worker:
            url_versions, dirs, env = get_versions_dirs_env(worker, versions_from_env)
        overriden_versions = url_versions.copy()
        if explicit_versions is not None:
            overriden_versions.update(explicit_versions)
        return overriden_versions, url_versions, dirs, env

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

    versions_from_env = args.build_scripts_path is not None

    def check_container(
        date: datetime.date, *, test_output_log_level: int = logging.DEBUG
    ) -> TestResult:
        """
        See if the test passes in the given dated container.
        """
        before = time.monotonic()
        out_dir = test_output_directory(container_url(date))
        with Container(
            container_url(date), test_output_host_directory=out_dir
        ) as worker:
            versions, _, _ = get_versions_dirs_env(worker, versions_from_env)
            # This will stream interleaved stdout/stderr into the logger
            result = worker.exec(args.test_command, log_level=test_output_log_level)
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
            | versions,
        )
        return TestResult(
            host_output_directory=out_dir, result=test_pass, stdouterr=result.stdout
        )

    if args.container_runtime == "local":
        passing_url = "local"
        failing_url = "local"
    elif args.passing_container is None and args.failing_container is None:
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

    # Get the versions from the endpoint containers (if they exist), overridden by any
    # explicitly passed versions.
    passing_versions, original_passing_versions, passing_package_dirs, passing_env = (
        get_versions(passing_url, args.passing_versions, versions_from_env)
    )
    failing_versions, original_failing_versions, failing_package_dirs, failing_env = (
        get_versions(failing_url, args.failing_versions, versions_from_env)
    )

    # If we have two containers, print the differences between their environments. This
    # can be useful in the case that rebuilding the good versions in the bad container,
    # or vice versa, does not reproduce the expected result.
    if passing_env is not None and failing_env is not None:
        logger.info(f"Environment differences between {passing_url} and {failing_url}")
        for key in passing_env.keys() - failing_env.keys():
            logger.info(f"Only in {passing_url}: {key}={passing_env[key]}")
        for key in failing_env.keys() - passing_env.keys():
            logger.info(f"Only in {failing_url}: {key}={failing_env[key]}")
        for key in passing_env.keys() & failing_env.keys():
            if passing_env[key] == failing_env[key]:
                continue
            logger.info(
                f"{key}: {passing_env[key]} ({passing_url}) vs. {failing_env[key]} "
                f"({failing_url})"
            )

    # We should have versions for all the same software packages at both
    # ends of the range, one way or another. TODO: this could be relaxed.
    assert passing_versions.keys() == failing_versions.keys(), (
        passing_versions,
        failing_versions,
    )

    # Which packages have versions that are not always the same?
    dynamic_packages = {
        pkg for pkg, _ in set(passing_versions.items()) ^ set(failing_versions.items())
    }

    # Choose an environment to do the version-level bisection in; use directory names that
    # match it, and track what the initial versions of the different packages are
    if args.container_runtime == "local":
        bisection_url = "local"
        bisection_versions = original_failing_versions
        package_dirs = failing_package_dirs
    elif failing_url is not None:
        bisection_url = failing_url
        bisection_versions = original_failing_versions
        package_dirs = failing_package_dirs
    else:
        assert passing_url is not None
        bisection_url = passing_url
        bisection_versions = original_passing_versions
        package_dirs = passing_package_dirs
    assert package_dirs is not None
    # This is the set of versions that are already installed
    assert bisection_versions is not None

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

    # Fire up the container that will be used for the version-level search and use it to
    # extract the relevant history of the repositories that will be triaged.
    with Container(bisection_url) as worker:
        packages = passing_versions.keys()
        log_str = "Bisecting"
        for package in packages:
            log_str += (
                f" {package} [{passing_versions[package]}, {failing_versions[package]}]"
            )
        log_str += f" using {worker}"
        logger.info(log_str)
        # Get lists of (commit_hash, commit_date) pairs
        package_versions = collections.OrderedDict()
        for package in packages:
            if package not in package_dirs:
                # This is a version that came from the container environment, not a git
                # checkout directory in the container. Handle those below.
                continue
            package_versions[package] = get_commit_history(
                worker,
                passing_versions[package],
                failing_versions[package],
                package_dirs[package],
            )
            # Confirm they're sorted by commit date
            assert all(
                b[1] >= a[1]
                for a, b in zip(
                    package_versions[package], package_versions[package][1:]
                )
            )
            # Confirm the end values are included as expected
            assert passing_versions[package] == package_versions[package][0][0]
            assert failing_versions[package] == package_versions[package][-1][0]
        # For the packages that just have one or two version numbers, associate those
        # version numbers with the earliest and, if appropriate, latest XLA dates.
        for package in packages:
            if package in package_versions:
                continue
            package_versions[package] = [
                (passing_versions[package], package_versions["xla"][0][1]),
            ]
            if passing_versions[package] != failing_versions[package]:
                package_versions[package].append(
                    (failing_versions[package], package_versions["xla"][-1][1])
                )

        # Check up-front whether the installation scripts exist for the packages that
        # are being triaged by version + script rather than from a git repo + build.
        if args.build_scripts_path is not None:
            known_scripts = worker.check_exec(
                [
                    "find",
                    args.build_scripts_path,
                    "-maxdepth",
                    "1",
                    "-executable",
                    "-print0",
                ],
                policy="once",
                stderr="separate",
            ).stdout.split("\0")
            logger.debug(f"Found {known_scripts} inside {worker}")
            packages_with_scripts = {
                script[len(args.build_scripts_path) + 8 : -3]
                for script in known_scripts
                if script.startswith(args.build_scripts_path + "/install")
                and script.endswith(".sh")
            }
            logger.debug(f"Found installation scripts for {packages_with_scripts}")
            packages_needing_scripts = dynamic_packages - package_dirs.keys()
            packages_missing_scripts = packages_needing_scripts - packages_with_scripts
            if packages_missing_scripts:
                logger.warning(
                    f"No installation scripts found for: {packages_missing_scripts}, "
                    "whose versions change across the bisection range. These will be "
                    "excluded from the bisection, which may cause it not to converge!"
                )
                dynamic_packages -= packages_missing_scripts

    def build_and_test(
        *, versions: typing.Dict[str, str], test_output_log_level: int = logging.DEBUG
    ) -> TestResult:
        """
        The main body of the bisection loop. Update JAX/XLA/... versions, rebuild, and
        run the test command. Throws on error when checking out or building, and returns
        the status of the test command.
        """
        # Amortise container startup overhead by batching together git commands
        git_commands, changed, skipped = [], [], []
        for package in sorted(dynamic_packages):
            version = versions[package]
            if bisection_versions[package] == version:
                # If the existing version is the desired one, do nothing.
                skipped.append(f"{package}@{version}")
                continue
            # Cache which version is now going to be checked out in the container
            bisection_versions[package] = version
            changed.append(f"{package}@{version}")
            if package in package_dirs:
                # A git repository that exists in the container.
                git_commands += [
                    f"cd {package_dirs[package]}",
                    "git stash",
                    f"git checkout {version}",
                ]
            else:
                # Another software package, `version` is probably a version number.
                # Installation of this version is delegated to an installPACKAGE.sh
                # script that is assumed to be available in `args.build_scripts_path`.
                assert args.build_scripts_path is not None
                assert package in packages_with_scripts, (
                    package,
                    packages_with_scripts,
                )
                extra_env = {
                    # Need the static part for the .bc library
                    "NVSHMEM": "DEVEL=1 STATIC=1",
                }.get(package, "DEVEL=1")  # Always need development headers to rebuild
                git_commands += [
                    f"{extra_env} {args.build_scripts_path}/install{package}.sh {version}"
                ]
        # Keep the pathnames shorter by only including packages that actually have
        # multiple versions in the bisection range.
        brief_versions = {
            p: ver for p, ver in versions.items() if p in dynamic_packages
        }
        out_dir = test_output_directory(bisection_url, versions=brief_versions)
        with Container(bisection_url, test_output_host_directory=out_dir) as worker:
            change_str = " ".join(changed) if len(changed) else "<nothing>"
            info_str = f"Checking out {change_str} in {worker}"
            if len(skipped):
                info_str += f", leaving {' '.join(skipped)} unchanged"
            logger.info(info_str)
            worker.check_exec(
                ["sh", "-c", " && ".join(git_commands)],
                policy="once_per_container",
            )
            # Build JAX
            # TODO: teach the tool how to build TransformerEngine too
            # TODO: do not build JAX/XLA/TransformerEngine if we know their versions did not change?
            before = time.monotonic()
            # Unfortunately the build system does not always seem to handle incremental
            # rebuilds correctly, so clean the local cache and rely on the remote one.
            build_cmds = [
                "bazel clean --expunge",
                f"build-jax.sh --bazel-cache={args.bazel_cache}",
            ]
            worker.check_exec(
                ["sh", "-c", " && ".join(build_cmds)],
                policy="once_per_container",
                workdir=package_dirs["jax"],
            )
            middle = time.monotonic()
            logger.info(f"Build completed in {middle - before:.1f}s")
            # Run the test
            test_result = worker.exec(
                args.test_command, log_level=test_output_log_level
            )
            test_time = time.monotonic() - middle
        add_summary_record(
            "versions",
            {
                "build_time": middle - before,
                "container": bisection_url,
                "output_directory": out_dir.as_posix(),
                "result": test_result.returncode == 0,
                "test_time": test_time,
            }
            | versions,
        )
        result_str = "pass" if test_result.returncode == 0 else "fail"
        logger.info(f"Test completed in {test_time:.1f}s ({result_str})")
        return TestResult(
            host_output_directory=out_dir,
            result=test_result.returncode == 0,
            stdouterr=test_result.stdout,
        )

    # Run the version-level bisection
    result, last_known_good, first_known_bad = version_search(
        versions=package_versions,
        build_and_test=build_and_test,
        logger=logger,
        skip_precondition_checks=args.skip_precondition_checks,
    )

    def symlink(result: typing.Optional[TestResult], symlink_name: str) -> None:
        if result is None:
            return
        symlink = (args.output_prefix / symlink_name).resolve()
        assert not symlink.exists(), symlink
        assert symlink.parent == result.host_output_directory.parent, (
            symlink,
            result.host_output_directory,
        )
        symlink.symlink_to(result.host_output_directory.name)

    symlink(last_known_good, "last-known-good")
    symlink(first_known_bad, "first-known-bad")
    result["container"] = failing_url
    add_summary_record("result", result, scalar=True)
