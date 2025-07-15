import datetime
import subprocess


def get_commit_history(
    worker,
    package,
    start,
    end,
    dir,
    main_branch,
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
        main_branch (str): The main branch name. Defaults is the default branch of the repo.
        logger (Logger, optional): Logger for debug information. Defaults to None.
        args: Additional arguments that may contain cherry-pick commits.

    Returns:
        data: list, list of all the commits
        cherry_pick_range: str, range of cherry pick commits if any
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
        worker.check_exec(["git", "fetch", args.override_remotes.get(package, "origin"), start, end], workdir=dir)```

    # detect non-linear history
    is_ancestor_result = worker.exec(
        ["git", "merge-base", "--is-ancestor", start, end],
        workdir=dir,
    )
    is_linear = is_ancestor_result.returncode == 0
    cherry_pick_range = {}

    if not is_linear:
        logger.info(f"Using non-linear history logic with branch {main_branch}")

        # 1. find the linear range on the main branch
        passing_and_failing_cmd = worker.check_exec(
            [
                "sh",
                "-c",
                f"git merge-base {start} {end} && git merge-base {end} {main_branch}",
            ],
            workdir=dir,
        ).stdout.strip()
        passing_main_commit, failing_main_commit = passing_and_failing_cmd.splitlines()

        # 2. find commits to cherry-pick from the failing branch
        # TODO: as an alternative approach we may need to consider `{passing_main_commit}..{start}`
        cherry_pick_range[package] = f"{failing_main_commit}..{end}"

        # 3. now we can use the main branch  commits for bisection
        start = passing_main_commit
        end = failing_main_commit

    logger.info(
        f"INFO: cherry_pick_range {cherry_pick_range}, start: {start} and end {end}"
    )
    # check if the start is the root commit. We may have to deal with the very start of the repo
    # so we need to handle this case too
    parent_check_result = worker.check_exec(
        ["git", "rev-list", "--parents", "-n", "1", start], workdir=dir
    )
    is_root_commit = len(parent_check_result.stdout.strip().split()) == 1
    log_range = f"{start}..{end}" if is_root_commit else f"{start}^..{end}"

    # now create the right git command to retrieve the history between start..end
    result = worker.check_exec(
        ["git", "log", "--first-parent", "--reverse", "--format=%H %cI", log_range],
        policy="once",
        stderr=subprocess.PIPE,
        workdir=dir,
    )

    data = []
    for line in result.stdout.splitlines():
        commit, date = line.split()
        # for python < 3.11 we nee dto fix:
        if date.endswith("Z"):
            date = date[:-1] + "+00:00"
        date = datetime.datetime.fromisoformat(date).astimezone(datetime.timezone.utc)
        data.append((commit, date))

    return data, cherry_pick_range
