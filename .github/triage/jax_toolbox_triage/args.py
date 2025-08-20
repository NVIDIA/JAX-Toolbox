import argparse
import datetime
import getpass
import os
import pathlib
import tempfile
import typing
import warnings

# Software we know may exist in the containers that we might be able to triage
# We know how to recompile JAX/XLA, so it's OK that they include C++ code
# TransformerEngine is intentionally excluded because build-te.sh is not plumbed yet.
# Flax and MaxText are pure Python, so it's OK we don't have a way of compiling them,
# but they are not always installed in containers we want to triage.
# Note this is not a `set` for the sake of run-to-run determinism.
compulsory_software = ["xla", "jax"]
optional_software = ["flax", "maxtext", "transformer-engine"]


def parse_cherry_picks(s: str) -> typing.Dict[str, typing.List[str]]:
    ret: typing.Dict[str, typing.List[str]] = {}
    for part in s.split(","):
        sw, commit = part.split(":", 1)
        if sw not in ret:
            ret[sw] = []
        ret[sw].append(commit)
    return ret


def parse_version_argument(s: str) -> typing.Dict[str, str]:
    ret: typing.Dict[str, str] = {}
    for part in s.split(","):
        sw, version = part.split(":", 1)
        assert sw not in ret, ret
        ret[sw] = version
    return ret


def parse_override_remotes(s: str) -> typing.Dict[str, str]:
    """Function to parse the override remote

    Inputs:
        s: (str) e.g. https://<token>@host/repo.git

    Returns:
        ret: (typing.Dict[str,str]) Dictionary with software as key and git-url as value.
    """
    ret: typing.Dict[str, str] = {}
    for part in s.split(","):
        sw, url = part.split(":", 1)
        assert sw not in ret, ret
        ret[sw] = url
    return ret


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
            Triage failures in JAX/XLA-related tests. The expectation is that the given
            test command is failing in recent versions, but that it passed in the past.
            The script first triages the regression with a search of the nightly
            containers, and then refines the search to a particular version of JAX, XLA,
            or another component in the container. If a git repository and build recipe
            are available for a component then the triage granularity is a single git
            commit. If a component does not have a full history, but is known to have
            different versions in the last-known-good and first-known-bad test
            environments then the granularity is just between those two versions.""",
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
    version_search_args = parser.add_argument_group(
        title="Version-level search",
        description="""
            Second, the failure is localised to a single version (commit of JAX/XLA/...,
            version number of cuBLAS/cuDNN/...) by re-building/re-installing/re-testing
            inside a single container/environment. This is based on a binary search that
            aligns the version histories of JAX, XLA, ... with each other by timestamp.""",
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
            Skip the container-level search and pass this container to the version-level
            search. If this is passed, --passing-container or --passing-versions must be
            too, and container-level search arguments such as --container must not be
            passed. This can be used to apply the version-level bisection search to
            containers not from the ghcr.io/nvidia/jax:CONTAINER-YYYY-MM-DD series,
            although they must have a similar structure.""",
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
            Skip the container-level search and pass this container to the version-level
            search. If this is passed, --failing-container or --failing-versions must be
            too, and container-level search arguments such as --container must not be
            passed. This can be used to apply the version-level bisection search to
            containers not from the ghcr.io/nvidia/jax:CONTAINER-YYYY-MM-DD series,
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
    version_search_args.add_argument(
        "--bazel-cache",
        help="""
            Bazel cache to use when [re-]building JAX/XLA during the fine search. This can
            be a remote cache server or a local directory. Using a persistent cache can
            significantly speed up the version-level search. By default, uses a temporary
            directory including the name of the current user and assumes that it can be
            read from all systems that launch containers (this can fail with the pyxis
            runtime where those containers may run on remote systems).""",
    )
    version_search_args.add_argument(
        "--failing-versions",
        help="""
            Explicitly specify component versions to use at the failing endpoint of the
            version-level triage. This can be combined with --passing-container to run
            a triage with a single container (i.e. without passing --failing-container),
            and/or combined with --failing-container to override the initial versions
            read from that container. Expects an argument of the form
            jax:commit_hash,xla:commit_hash,CUBLAS:version.number[,...].""",
        type=parse_version_argument,
    )
    version_search_args.add_argument(
        "--failing-commits",
        help="Deprecated alias for --failing-versions",
        type=parse_version_argument,
    )
    version_search_args.add_argument(
        "--passing-versions",
        help="""
            Explicitly specify component versions to use at the failing endpoint of the
            version-level triage. This can be combined with --failing-container to run
            a triage with a single container (i.e. without passing --passing-container),
            and/or combined with --passing-container to override the initial versions
            read from that container. Expects an argument of the form
            jax:commit_hash,xla:commit_hash,CUBLAS:version.number[,...].""",
        type=parse_version_argument,
    )
    version_search_args.add_argument(
        "--passing-commits",
        help="Deprecated alias for --passing-versions",
        type=parse_version_argument,
    )
    version_search_args.add_argument(
        "--cherry-pick",
        default={},
        help="""
            Cherry-pick the given fix(es) into the specified packages. Expects an
            argument of the form jax:hash1,jax:hash2,xla:hash3[,...] - when checking
            out a new version of the given package then those commit hashes will be
            cherry-picked in on top if they apply cleanly, otherwise they will be
            skipped. This is mainly intended to allow, for example, build fixes to be
            cherry-picked to allow triage of a test failure in a time period when the
            build was not succeeding.""",
        type=parse_cherry_picks,
    )
    version_search_args.add_argument(
        "--workaround-buggy-container",
        help="Use the parent of the commit read from the start container for the given components.",
        action="append",
        default=[],
        type=str,
    )
    version_search_args.add_argument(
        "--build-scripts-path",
        help="""
            This is a path inside the container that contains installPACKAGE.sh
            executables that the tool can use to move the versions of software
            components *other than* the git repositories (XLA, JAX, ...) that are
            explicitly supported. If this is given, the set of `PACKAGE` names is taken
            from environment variables of the form `{PACKAGE}_VERSION` in the container
            runtime environment. These executables will only be called if the component
            in question has different versions at the endpoints of the bisection range.
        """,
    )
    version_search_args.add_argument(
        "--override-remotes",
        type=parse_override_remotes,
        default={},
        help="""Remote URLs to be used for fetching, including auth token. E.g.:
            jax:https://<token>@host/repo.git,xla:https://<token>@host/repo.git
            """,
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
        help="Container runtime used, can be docker, pyxis, or local.",
        type=lambda s: s.lower(),
    )
    parser.add_argument(
        "--main-branch",
        type=str,
        default="main",
        help="The name of the main branch (e.g. main) to derive cherry-picks from",
    )
    args = parser.parse_args(args=args)
    assert args.container_runtime in {
        "docker",
        "pyxis",
        "local",
    }, args.container_runtime
    args.workaround_buggy_container = set(args.workaround_buggy_container)
    # --{passing,failing}-commits are deprecated aliases for --{passing,failing}-versions.
    for prefix in ["passing", "failing"]:
        commits = getattr(args, f"{prefix}_commits")
        delattr(args, f"{prefix}_commits")
        if commits is None:
            continue
        if getattr(args, f"{prefix}_versions") is None:
            warnings.warn(
                f"WARNING: deprecated alias --{prefix}-commits being used, please "
                f"migrate to --{prefix}-versions",
                DeprecationWarning,
            )
            setattr(args, f"{prefix}_versions", commits)
        else:
            raise Exception(
                f"Both --{prefix}-commits and --{prefix}-versions passed, "
                f"--{prefix}-commits is deprecated - please remove it."
            )

    if args.container_runtime == "pyxis":
        assert args.bazel_cache is not None, (
            "For pyxis runtime, --bazel-cache must be provided explicitly. You likely "
            "want to use a remote cache URL."
        )
    elif args.container_runtime == "docker" and args.bazel_cache is None:
        # In this case we can share a cache across containers by mounting in a
        # temporary directory on the host machine.
        args.bazel_cache = os.path.join(
            tempfile.gettempdir(), f"{getpass.getuser()}-bazel-triage-cache"
        )

    if args.container_runtime == "local":
        assert (
            args.passing_versions is not None and args.failing_versions is not None
        ), (
            "For local runtime, --passing-versions and --failing-versions must be provided."
        )
        assert (
            args.container is None
            and args.start_date is None
            and args.end_date is None
            and args.passing_container is None
            and args.failing_container is None
        ), "Container-level search options are not applicable for local runtime."
        return args

    passing_versions_known = (args.passing_container is not None) or (
        args.passing_versions is not None
    )
    failing_versions_known = (args.failing_container is not None) or (
        args.failing_versions is not None
    )
    sets_of_known_versions = passing_versions_known + failing_versions_known
    if sets_of_known_versions == 2:
        # If the container-level search is being skipped, because a valid combination
        # of --{passing,failing}-{versions,container} is passed, then no container-level
        # search options should be passed.
        assert (
            args.container is None and args.start_date is None and args.end_date is None
        ), (
            "No container-level search options should be passed if the passing/failing"
            " containers/versions have been passed explicitly."
        )
        assert (
            args.passing_container is not None or args.failing_container is not None
        ), "At least one of --passing-container and --failing-container must be passed."
        for prefix in ["passing", "failing"]:
            assert getattr(args, f"{prefix}_container") is not None or getattr(
                args, f"{prefix}_versions"
            ).keys() >= set(compulsory_software), (
                f"--{prefix}-commits must specify all of {compulsory_software} if "
                f"--{prefix}-container is not specified"
            )
    elif sets_of_known_versions == 1:
        raise Exception(
            "If --passing-{versions AND/OR container} is passed then "
            "--failing-{versions AND/OR container} should be too"
        )
    else:
        # None of --{passing,failing}-{versions,container} were passed, make sure the
        # compulsory arguments for the container-level search were passed
        assert args.container is not None, (
            "--container must be passed for the container-level search"
        )

    return args
