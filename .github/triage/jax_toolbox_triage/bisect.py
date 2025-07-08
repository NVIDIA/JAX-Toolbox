import datetime
import subprocess


def get_commit_history(
    worker,
    package,
    start,
    end,
    dir,
    main_branch=None,
    logger=None,
    args=None,
):
    """
    Get the commit history for a given package between two commits.

    Args:
        worker (Container): The container worker to execute commands.
        package (str): The name of the package.
        start (str): The starting commit hash.
        end (str): The ending commit hash.
        dir (str): The directory where the git repository is located.
        main_branch (str, optional): The main branch name. Defaults to None.
        logger (Logger, optional): Logger for debug information. Defaults to None.
        args: Additional arguments that may contain cherry-pick commits.

    Returns:
        list: A list of tuples containing commit hashes and their corresponding dates.
    """
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
        if worker.exec(["git", "remote"]).stout.strip():
            worker.check_exec(
                ["git", "fetch"], policy="once_per_container", workdir=dir
            )
        else:
            logger.warning("No remote found, skipping fetch.")

    # detect non-linear history
    is_ancestor_cmd = f"git merge-base --is-ancestor {start} {end}"
    is_ancestor_result = worker.exec(
        ["sh", "-c", is_ancestor_cmd],
        workdir=dir,
    )
    is_linear = is_ancestor_result.returncode == 0

    if not is_linear and package in ["jax", "xla"]:
        logger.info(f"Using non-linear history logic with main branch {main_branch}")

        # 1. find the linear range on the main branch
        passing_main_commit_cmd = f"git merge-base {start} {end}"
        failing_main_commit_cmd = f"git merge-base {end} {main_branch}"

        passing_main_commit = worker.check_exec(
            ["sh", "-c", passing_main_commit_cmd], workdir=dir
        ).stdout.strip()
        failing_main_commit = worker.check_exec(
            ["sh", "-c", failing_main_commit_cmd], workdir=dir
        ).stdout.strip()

        # 2. find commits to cherry-pick from the failing branch
        cherry_pick_cmd = f"git rev-list --reverse {failing_main_commit}..{end}"
        cherry_pick_commits_list = (
            worker.check_exec(["sh", "-c", cherry_pick_cmd], workdir=dir)
            .stdout.strip()
            .splitlines()
        )
        if cherry_pick_commits_list:
            args.cherry_pick_commits[package] = cherry_pick_commits_list
        logger.info(f"Cherry-pick commits: {cherry_pick_commits_list}")

        # 3. now we can use the main branch  commits for bisection
        start = passing_main_commit
        end = failing_main_commit

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
        # for python < 3.11 we nee dto fix:
        if date.endswith("Z"):
            date = date[:-1] + "+00:00"
        date = datetime.datetime.fromisoformat(date).astimezone(datetime.timezone.utc)
        data.append((commit, date))
    return data
