#!/usr/bin/env python3
import datetime
import functools
import json
import logging
import time
import typing

from .args import parse_args
from .docker import DockerContainer
from .logic import commit_search, container_search
from .utils import (
    container_exists as container_exists_base,
    container_url as container_url_base,
    get_logger,
    prepare_bazel_cache_mounts,
)


def main():
    args = parse_args()
    bazel_cache_mounts = prepare_bazel_cache_mounts(args.bazel_cache)
    logger = get_logger(args.output_prefix)
    container_url = functools.partial(container_url_base, container=args.container)
    container_exists = functools.partial(
        container_exists_base, container=args.container, logger=logger
    )
    Container = functools.partial(
        DockerContainer, logger=logger, mounts=bazel_cache_mounts
    )
    bazel_cache_mount_args = []
    for src, dst in bazel_cache_mounts:
        bazel_cache_mount_args += ["-v", f"{src}:{dst}"]

    def add_summary_record(section, record, scalar=False):
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

    def get_commit(container: DockerContainer, repo: str) -> typing.Tuple[str, str]:
        """
        Get the commit of the given repository that was used in the given nightly container

        Arguments:
        date: nightly container date
        repo: repository, must be jax or xla
        """
        assert repo in {"jax", "xla"}
        # Older containers used /opt/jax-source etc.
        for suffix in ["", "-source"]:
            dirname = f"/opt/{repo}{suffix}"
            result = container.exec(["git", "rev-parse", "HEAD"], workdir=dirname)
            if result.returncode == 0:
                commit = result.stdout.strip()
                if len(commit) == 40:
                    return commit, dirname
        raise Exception(
            f"Could not extract commit of {repo} from {args.container} container {container}"
        )

    def check_container(date: datetime.date) -> bool:
        """
        See if the test passes in the given container.
        """
        before = time.monotonic()
        with Container(container_url(date)) as worker:
            result = worker.exec(args.test_command)
            test_time = time.monotonic() - before
            jax_commit = get_commit(worker, "jax")
            xla_commit = get_commit(worker, "xla")

        logger.debug(result.stdout)
        logger.info(f"Ran test case in {date} in {test_time:.1f}s")
        test_pass = result.returncode == 0
        add_summary_record(
            "container",
            {
                "container": container_url(date),
                "jax": jax_commit,
                "result": test_pass,
                "test_time": test_time,
                "xla": xla_commit,
            },
        )
        return test_pass

    # Search through the published containers, narrowing down to a pair of dates with
    # the property that the test passed on `range_start` and fails on `range_end`.
    range_start, range_end = container_search(
        container_exists=container_exists,
        container_passes=check_container,
        start_date=args.start_date,
        end_date=args.end_date,
        logger=logger,
        skip_precondition_checks=args.skip_precondition_checks,
        threshold_days=args.threshold_days,
    )

    # Container-level search is now complete. Triage proceeds inside the `range_end``
    # container. First, we check that rewinding JAX and XLA inside the `range_end``
    # container to the commits used in the `range_start` container passes, whereas
    # using the `range_end` commits reproduces the failure.

    with Container(container_url(range_start)) as worker:
        start_jax_commit, _ = get_commit(worker, "jax")
        start_xla_commit, _ = get_commit(worker, "xla")

    # Fire up the container that will be used for the fine search.
    with Container(container_url(range_end)) as worker:
        end_jax_commit, jax_dir = get_commit(worker, "jax")
        end_xla_commit, xla_dir = get_commit(worker, "xla")
        logger.info(
            (
                f"Bisecting JAX [{start_jax_commit}, {end_jax_commit}] and "
                f"XLA [{start_xla_commit}, {end_xla_commit}] using {container_url(range_end)}"
            )
        )

        # Get the full lists of JAX/XLA commits and dates
        def commits(start, end, dir):
            result = worker.check_exec(
                [
                    "git",
                    "log",
                    "--first-parent",
                    "--reverse",
                    "--format=%H %cI",
                    f"{start}^..{end}",
                ],
                workdir=dir,
            )
            data = []
            for line in result.stdout.splitlines():
                commit, date = line.split()
                date = datetime.datetime.fromisoformat(date).astimezone(
                    datetime.timezone.utc
                )
                data.append((commit, date))
            return data

        # Get lists of (commit_hash, commit_date) pairs
        jax_commits = commits(start_jax_commit, end_jax_commit, jax_dir)
        xla_commits = commits(start_xla_commit, end_xla_commit, xla_dir)
        # Confirm they're sorted by commit date
        assert all(b[1] >= a[1] for a, b in zip(jax_commits, jax_commits[1:]))
        assert all(b[1] >= a[1] for a, b in zip(xla_commits, xla_commits[1:]))
        # Confirm the end values are included as expected
        assert start_jax_commit == jax_commits[0][0]
        assert start_xla_commit == xla_commits[0][0]
        assert end_jax_commit == jax_commits[-1][0]
        assert end_xla_commit == xla_commits[-1][0]

        def build_and_test(
            jax_commit: str, xla_commit: str
        ) -> typing.Tuple[bool, str, str]:
            """
            The main body of the bisection loop. Update the JAX/XLA commits, build XLA and
            jaxlib, and run the test command. Throws on error when checking out or
            building, and returns the status of the test command.
            """
            worker.check_exec(["git", "stash"], workdir=xla_dir)
            worker.check_exec(["git", "stash"], workdir=jax_dir)
            worker.check_exec(["git", "checkout", xla_commit], workdir=xla_dir)
            worker.check_exec(["git", "checkout", jax_commit], workdir=jax_dir)
            logger.info(f"Checking out XLA {xla_commit} JAX {jax_commit}")
            # Build JAX
            before = time.monotonic()
            # Next two are workarounds for bugs in old containers
            worker.check_exec(["sh", "-c", f"rm -vf {jax_dir}/dist/jaxlib-*.whl"])
            # This will error out on newer containers, but that should be harmless
            worker.exec(
                [
                    "cp",
                    f"{jax_dir}/jax/version.py",
                    f"{jax_dir}/build/lib/jax/version.py",
                ]
            )
            # It seemed that this might be the origin of flaky behaviour.
            worker.check_exec(
                ["sh", "-c", "echo 'test --cache_test_results=no' > /root/.bazelrc"]
            )
            build_jax = [
                "build-jax.sh",
                # Leave the editable /opt/jax[-source] installation alone. Otherwise
                # test-jax.sh is broken by having a /usr/... installation directory.
                "--jaxlib_only",
                # Workaround bugs in old containers where the default was wrong.
                "--src-path-jax",
                jax_dir,
                f"--bazel-cache={args.bazel_cache}",
            ]
            worker.check_exec(build_jax, workdir=jax_dir)
            middle = time.monotonic()
            logger.info(f"Build completed in {middle - before:.1f}s")
            # Run the test
            test_result = worker.exec(args.test_command)
            test_time = time.monotonic() - middle
            add_summary_record(
                "commit",
                {
                    "build_time": middle - before,
                    "container": container_url(range_end),
                    "jax": jax_commit,
                    "result": test_result.returncode == 0,
                    "test_time": test_time,
                    "xla": xla_commit,
                },
            )
            logger.info(f"Test completed in {test_time:.1f}s")
            logger.debug(
                f"Test stdout:\n{test_result.stdout}\nTest stderr:\n{test_result.stderr}"
            )
            return test_result.returncode == 0, test_result.stdout, test_result.stderr

        # Run the commit-level bisection
        result = commit_search(
            jax_commits=jax_commits,
            xla_commits=xla_commits,
            build_and_test=build_and_test,
            logger=logger,
            skip_precondition_checks=args.skip_precondition_checks,
        )
        result["container"] = container_url(range_end)
        add_summary_record("result", result, scalar=True)
