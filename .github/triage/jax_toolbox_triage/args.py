import argparse
import datetime
import getpass
import os
import pathlib
import tempfile


def parse_args(args=None):
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
    parser.add_argument(
        "--container",
        help="""
            Container to use. Example: jax, pax, triton. Used to construct the URLs of
            nightly containers, like ghcr.io/nvidia/jax:CONTAINER-YYYY-MM-DD.""",
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
            search. If this is passed, --passing-container must be too, but --container
            is not required. This can be used to apply the commit-level bisection
            search to containers not from the ghcr.io/nvidia/jax:CONTAINER-YYYY-MM-DD
            series, although they must have a similar structure.""",
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
            search. If this is passed, --failing-container must be too, but --container is
            not required. This can be used to apply the commit-level bisection search
            to containers not from the ghcr.io/nvidia/jax:CONTAINER-YYYY-MM-DD series,
            although they must have a similar structure.""",
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
    args = parser.parse_args(args=args)
    num_explicit_containers = (args.passing_container is not None) + (
        args.failing_container is not None
    )
    if num_explicit_containers == 1:
        raise Exception(
            "--passing-container and --failing-container must both be passed if either is"
        )
    if num_explicit_containers == 2:
        # Explicit mode, --container, --start-date and --end-date are all ignored
        if args.container:
            raise Exception(
                "--container must not be passed if --passing-container and --failing-container are"
            )
        if args.start_date:
            raise Exception(
                "--start-date must not be passed if --passing-container and --failing-container are"
            )
        if args.end_date:
            raise Exception(
                "--end-date must not be passed if --passing-container and --failing-container are"
            )
    elif num_explicit_containers == 0 and args.container is None:
        raise Exception(
            "--container must be passed if --passing-container and --failing-container are not"
        )
    return args
