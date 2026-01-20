import argparse
import datetime
import logging
import typing

from .container import Container


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
                f"git fetch {args.override_remotes.get(package, 'origin')} {main_branch}:{main_branch} && git merge-base {start} {end} && git merge-base {end} {main_branch}",
            ],
            policy="once",
            stderr="separate",
            workdir=dir,
        ).stdout.strip()
        passing_main_commit, failing_main_commit = passing_and_failing_cmd.splitlines()

        # 2. find commits to cherry-pick from the failing branch
        cherry_pick_ranges += [
            f"{failing_main_commit}..{end}",
            f"{passing_main_commit}..{start}",
        ]

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
        # for python < 3.11 we need to fix:
        if date.endswith("Z"):
            date = date[:-1] + "+00:00"
        date = datetime.datetime.fromisoformat(date).astimezone(datetime.timezone.utc)
        data.append((commit, date))

    return data, cherry_pick_ranges
