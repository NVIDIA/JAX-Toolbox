import argparse
import datetime
import logging
import typing
from dataclasses import dataclass

from .container import Container


@dataclass(frozen=True)
class CommitAndParent:
    package: str
    commit: str
    parent: str
    needs_fetch: bool


def _parse_git_date(date: str) -> datetime.datetime:
    # Python < 3.11 does not accept the trailing Z emitted by some Git versions.
    if date.endswith("Z"):
        date = date[:-1] + "+00:00"
    return datetime.datetime.fromisoformat(date).astimezone(datetime.timezone.utc)


def _commit_exists(worker: Container, revision: str, directory: str) -> bool:
    return (
        worker.exec(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            policy="once",
            stderr="separate",
            workdir=directory,
        ).returncode
        == 0
    )


def resolve_commit_and_parent(
    worker: Container,
    commit_spec: str,
    package_dirs: typing.Dict[str, str],
    override_remotes: typing.Dict[str, str],
) -> CommitAndParent:
    """Resolve ``[PACKAGE:]REVISION`` and its first parent.

    Bare revisions are located among the known source repositories. If the revision
    is not already present, each repository's configured remote is tried. An explicit
    package avoids those probes and gives clearer errors for abbreviated revisions.
    """
    if not commit_spec or any(c.isspace() for c in commit_spec):
        raise ValueError("--commit must be a non-empty Git revision without whitespace")

    package = None
    revision = commit_spec
    if ":" in commit_spec:
        package, revision = commit_spec.split(":", 1)
        if package not in package_dirs:
            known_packages = ", ".join(sorted(package_dirs))
            raise ValueError(
                f"Unknown package {package!r} in --commit; expected one of: "
                f"{known_packages}"
            )
    if not revision or revision.startswith("-"):
        raise ValueError(f"Invalid Git revision in --commit: {revision!r}")

    if package is not None:
        directory = package_dirs[package]
        needs_fetch = not _commit_exists(worker, revision, directory)
        if needs_fetch:
            worker.check_exec(
                ["git", "fetch", override_remotes.get(package, "origin"), revision],
                policy="once_per_container",
                stderr="separate",
                workdir=directory,
            )
        matching_packages = [package]
    else:
        matching_packages = [
            candidate_package
            for candidate_package, directory in package_dirs.items()
            if _commit_exists(worker, revision, directory)
        ]
        needs_fetch = False
        if not matching_packages:
            # A candidate can be newer than the checkout embedded in the bisection
            # container. Try each source repository's remote before giving up.
            for candidate_package, directory in package_dirs.items():
                fetch = worker.exec(
                    [
                        "git",
                        "fetch",
                        override_remotes.get(candidate_package, "origin"),
                        revision,
                    ],
                    policy="once_per_container",
                    stderr="separate",
                    workdir=directory,
                )
                if fetch.returncode == 0 and _commit_exists(
                    worker, revision, directory
                ):
                    matching_packages.append(candidate_package)
                    needs_fetch = True
                    break

    if not matching_packages:
        raise ValueError(
            f"Could not find --commit {revision!r} in any known source repository"
        )
    if len(matching_packages) > 1:
        packages = ", ".join(sorted(matching_packages))
        raise ValueError(
            f"--commit {revision!r} is ambiguous across {packages}; use "
            "PACKAGE:REVISION"
        )

    package = matching_packages[0]
    directory = package_dirs[package]
    commit_line = worker.check_exec(
        ["git", "rev-list", "--parents", "-n", "1", revision],
        policy="once",
        stderr="separate",
        workdir=directory,
    ).stdout.strip()
    commit_and_parents = commit_line.split()
    if len(commit_and_parents) == 1:
        raise ValueError(
            f"--commit {commit_and_parents[0]} is a root commit and has no parent to test"
        )

    commit, parent = commit_and_parents[:2]

    return CommitAndParent(
        package=package,
        commit=commit,
        parent=parent,
        needs_fetch=needs_fetch,
    )


def get_commit_history(
    worker: Container,
    package: str,
    start: str,
    end: str,
    dir: str,
    main_branch: str,
    logger: logging.Logger,
    args: argparse.Namespace,
) -> typing.Tuple[typing.List[typing.Tuple[str, datetime.datetime]], typing.List[str]]:
    """
    Get the commit history for a given package between two commits.

    Args:
        worker (Container): The container worker to execute commands.
        package (str): The name of the package.
        start (str): The starting commit hash.
        end (str): The ending commit hash.
        dir (str): The directory where the git repository is located.
        main_branch (str): The main branch name. Defaults is the default branch of the repo.
        logger (Logger, optional): Logger for debug information. Defaults to None.
        args: Additional arguments that may contain cherry-pick commits.

    Returns:
        data: list, list of all the commits
        cherry_pick_ranges: list[str], commits to attempt cherry-picking
    """
    # In particular the end commit might not already be known if the older,
    # passing, container is being used for triage.
    commits_known = (
        worker.exec(
            [
                "sh",
                "-c",
                f"git cat-file commit {start} && git cat-file commit {end}",
            ],
            policy="once",
            workdir=dir,
        ).returncode
        == 0
    )
    if not commits_known:
        worker.check_exec(
            ["git", "fetch", args.override_remotes.get(package, "origin"), start, end],
            policy="once_per_container",
            workdir=dir,
        )

    if package in args.workaround_buggy_container:
        # The automatic rebase of the JAX branch used in the internal nightly
        # containers was buggy for a while, leading to it re-writing commits that were
        # actually on upstream main to have different hashes:
        # b'          e'
        # |           |
        # a - b - c - d - e
        # where b=b' e=e' apart from commit message and hash, and b' and e' are the
        # commits in the containers. This unfortunately only differs from the 'true'
        # non-linear case by whether or not b=b' and e=e'.
        # b' = start
        # a  = start^
        # e' = end
        # The workaround here is to replace b' with its
        # parent a. This leaves open the possibility that the final result of the
        # triage could be reported as e', in which case the user can manually re-map it
        # to e, and makes the bisection range 1 commit wider than it really needs to be
        start = f"{start}^"

    # detect non-linear history
    is_linear = (
        worker.exec(
            ["git", "merge-base", "--is-ancestor", start, end],
            policy="once",
            workdir=dir,
        ).returncode
        == 0
    )
    cherry_pick_ranges = []
    if not is_linear:
        logger.debug(
            f"Using non-linear history logic for {package} with branch {main_branch}"
        )

        # 1. find the linear range on the main branch
        passing_and_failing_cmd = worker.check_exec(
            [
                "sh",
                "-c",
                " && ".join(
                    [
                        f"(git rev-parse --quiet --verify {main_branch} > /dev/null || git fetch {args.override_remotes.get(package, 'origin')} {main_branch}:{main_branch})",
                        f"git merge-base {start} {end}",
                        f"git merge-base {end} {main_branch}",
                    ]
                ),
            ],
            policy="once",
            stderr="separate",
            workdir=dir,
        ).stdout.strip()
        passing_main_commit, failing_main_commit = passing_and_failing_cmd.splitlines()

        # 2. find commits to cherry-pick from the failing branch
        if failing_main_commit != end:
            cherry_pick_ranges.append(f"{failing_main_commit}..{end}")
        if passing_main_commit != start:
            cherry_pick_ranges.append(f"{passing_main_commit}..{start}")

        # 3. now we can use the main branch commits for bisection
        start = passing_main_commit
        end = failing_main_commit

    logger.info(
        f"{package}: "
        + (f"{start}^..{end}" if start != end else start)
        + (
            f" (cherry_pick: {' '.join(cherry_pick_ranges)})"
            if len(cherry_pick_ranges)
            else ""
        )
    )

    # now create the right git command to retrieve the history between start..end
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
        stderr="separate",
        workdir=dir,
    )

    data = []
    for line in result.stdout.splitlines():
        commit, date = line.split()
        data.append((commit, _parse_git_date(date)))

    return data, cherry_pick_ranges
