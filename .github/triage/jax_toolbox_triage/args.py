import argparse
import datetime
import getpass
import os
import pathlib
import tempfile
import typing

# Software we know may exist in the containers that we might be able to triage
# We know how to recompile JAX/XLA, so it's OK that they include C++ code
# TransformerEngine is intentionally excluded because build-te.sh is not plumbed yet.
# Flax and MaxText are pure Python, so it's OK we don't have a way of compiling them,
# but they are not always installed in containers we want to triage.
# Note this is not a `set` for the sake of run-to-run determinism.
compulsory_software = ["xla", "jax"]
optional_software = ["flax", "maxtext"]


def parse_commit_argument(s: str) -> typing.Dict[str, str]:
    ret: typing.Dict[str, str] = {}
    for part in s.split(","):
        sw, commit = part.split(":", 1)
        assert sw not in ret, ret
        ret[sw] = commit
    return ret


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
            Triage failures in JAX/XLA-related tests. The expectation is that the given
            test command is failing in recent versions, but that it passed in the past. The
            script first triages the regression with a search of the nightly containers,
            and then refines the search to a particular commit of JAX or XLA.""",
    )

    container_search_args = parser.add_argument_group(
        title="Container-level search",
        description="""
            First, it is verified that the test command fails on the given end date, unless
            both --end-date and --skip-precondition-checks were passed. Then, the program
            searches backwards to find a container when the given test did pass. The
            --start-date option can be used to speed up this search, if you already know a
            date on which the test was passing. The earliest failure is located to within
            --threshold-days days.""",
    )
    commit_search_args = parser.add_argument_group(
        title="Commit-level search",
        description="""
            Second, the failure is localised to a commit of JAX or XLA by re-building and
            re-testing inside the earliest container that demonstrates the failure. At each
            point, the oldest JAX commit that is newer than XLA is used.""",
    )
    container_search_args.add_argument(
        "--container",
        help="""
            Container to use. Example: jax, maxtext. Used to construct the URLs of
            nightly containers, like ghcr.io/nvidia/jax:CONTAINER-YYYY-MM-DD.""",
    )
    container_search_args.add_argument(
        "--container-url-template",
        type=str,
        help="""
            Container URL pattern as a Python format string into which `container` and
            `date` will be substituted, e.g. ghcr.io/nvidia/jax:{container}-{date} for
            the JAX-Toolbox public nightlies.""",
    )
    parser.add_argument(
        "--output-prefix",
        default=datetime.datetime.now().strftime("triage-%Y-%m-%d-%H-%M-%S"),
        help="""
            Prefix for output log and JSON files. Default: triage-YYYY-MM-DD-HH-MM-SS.
            An INFO-and-above log is written as PREFIX.log, a DEBUG-and-above log is
            written as PREFIX-debug.log, and a JSON summary is written as
            PREFIX-summary.json""",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--skip-precondition-checks",
        action="store_true",
        help="""
            Skip checks that should pass by construction. This saves time, but may yield
            incorrect results if you are not careful. Specifically this means that the test
            is assumed to fail on --end-date (if specified), pass on --start-date (if
            specified), and fail after recompilation in the earliest-known-failure
            container. Careful use of this option, along with --start-date, --end-date and
            --threshold-days, allows the container-level search to be skipped.""",
    )
    parser.add_argument(
        "test_command",
        nargs="+",
        help="""
            Command to execute inside the container. This should be as targeted as
            possible.""",
    )
    container_search_args.add_argument(
        "--failing-container",
        help="""
            Skip the container-level search and pass this container to the commit-level
            search. If this is passed, --passing-container or --passing-commits must be
            too, but --container is not required. This can be used to apply the
            commit-level bisection search to containers not from the
            ghcr.io/nvidia/jax:CONTAINER-YYYY-MM-DD series, although they must have a
            similar structure.""",
    )
    container_search_args.add_argument(
        "--end-date",
        help="""
            Initial estimate of the earliest nightly container date where the test case
            fails. Defaults to the newest available nightly container date. If this and
            --skip-precondition-checks are both set then it will not be verified that the
            test case fails on this date.""",
        type=lambda s: datetime.date.fromisoformat(s),
    )
    container_search_args.add_argument(
        "--passing-container",
        help="""
            Skip the container-level search and pass this container to the commit-level
            search. If this is passed, --failing-container or --failing-commits must be
            too, but --container is not required. This can be used to apply the
            commit-level bisection search to containers not from the
            ghcr.io/nvidia/jax:CONTAINER-YYYY-MM-DD series, although they must have a
            similar structure.""",
    )
    container_search_args.add_argument(
        "--start-date",
        help="""
            Initial estimate of the latest nightly container date where the test case
            passes. Defaults to the day before --end-date, but setting this to a date
            further in the past may lead to faster convergence of the initial backwards
            search for a date when the test case passed. If this and
            --skip-precondition-checks are both set then the test case *must* pass on
            this date, which will *not* be verified.""",
        type=lambda s: datetime.date.fromisoformat(s),
    )
    container_search_args.add_argument(
        "--threshold-days",
        default=1,
        help="""
            Convergence threshold. Ideally, the container-level search will continue while
            the number of days separating the last known success and first known failure is
            smaller than this value. The minimum, and default, value is 1. Note that in
            case of nightly build failures the search may finish without reaching this
            threshold.""",
        type=int,
    )
    commit_search_args.add_argument(
        "--bazel-cache",
        default=os.path.join(
            tempfile.gettempdir(), f"{getpass.getuser()}-bazel-triage-cache"
        ),
        help="""
            Bazel cache to use when [re-]building JAX/XLA during the fine search. This can
            be a remote cache server or a local directory. Using a persistent cache can
            significantly speed up the commit-level search. By default, uses a temporary
            directory including the name of the current user.""",
    )
    commit_search_args.add_argument(
        "--failing-commits",
        help="""
            When combined with --passing-container, the commit-level triage will use
            that container and --failing-commits will specify the end of the commit
            range, rather than the commits being extracted from --failing-container.
            Expects an argument of form jax:jax_commit_hash,xla:xla_commit_hash.""",
        type=parse_commit_argument,
    )
    commit_search_args.add_argument(
        "--passing-commits",
        help="""
            When combined with --failing-container, the commit-level triage will use
            that container and --passing-commits will specify the start of the commit
            range, rather than the commits being extracted from --passing-container.
            Expects an argument of form jax:jax_commit_hash,xla:xla_commit_hash.""",
        type=parse_commit_argument,
    )
    parser.add_argument(
        "-v",
        "--container-mount",
        action="append",
        default=[],
        help="""
            Takes a SRC:DST value and mounts the (host) directory SRC into the container
            at DST; this can be used to pass in a test script, e.g. -v $PWD:/work before
            using /work/test.sh as a test command.""",
        type=lambda s: s.split(":", 1),
    )
    parser.add_argument(
        "--container-runtime",
        default="docker",
        help="Container runtime used, this can be either docker or pyxis.",
        type=lambda s: s.lower(),
    )
    args = parser.parse_args(args=args)
    assert args.container_runtime in {"docker", "pyxis"}, args.container_runtime
    passing_commits_known = (args.passing_container is not None) or (
        args.passing_commits is not None
    )
    failing_commits_known = (args.failing_container is not None) or (
        args.failing_commits is not None
    )
    sets_of_known_commits = passing_commits_known + failing_commits_known
    if sets_of_known_commits == 2:
        # If the container-level search is being skipped, because a valid combination
        # of --{passing,failing}-{commits,container} is passed, then no container-level
        # search options should be passed.
        assert (
            args.container is None and args.start_date is None and args.end_date is None
        ), (
            "No container-level search options should be passed if the passing/failing"
            " containers/commits have been passed explicitly."
        )
        assert (
            args.passing_container is not None or args.failing_container is not None
        ), "At least one of --passing-container and --failing-container must be passed."
        for prefix in ["passing", "failing"]:
            assert getattr(args, f"{prefix}_container") is not None or getattr(
                args, f"{prefix}_commits"
            ).keys() >= set(compulsory_software), (
                f"--{prefix}-commits must specify all of {compulsory_software} if "
                f"--{prefix}-container is not specified"
            )
    elif sets_of_known_commits == 1:
        raise Exception(
            "If --passing-{commits AND/OR container} is passed then "
            "--failing-{commits AND/OR container} should be too"
        )
    else:
        # None of --{passing,failing}-{commits,container} were passed, make sure the
        # compulsory arguments for the container-level search were passed
        assert args.container is not None, (
            "--container must be passed for the container-level search"
        )
    return args
