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
        if args.override_remotes and package in args.override_remotes:
            remote_url = args.override_remotes[package]
            fetch_cmd = ["git", "fetch", remote_url, start, end]
            worker.check_exec(fetch_cmd, workdir=dir)
        else:
            # default behaviour
            # re: https://stackoverflow.com/questions/4089430/how-to-determine-the-url-that-a-local-git-repository-was-originally-cloned-from
            remote_url_result = worker.check_exec(
                ["git", "config", "--get", "remote.origin.url"], workdir=dir
            )
            if remote_url_result.returncode == 0:
                remote_url = remote_url_result.stdout().strip()
                fetch_cmd = ["git", "fetch", remote_url, start, end]
                worker.check_exec(fetch_cmd, workdir=dir)
            else:
                logger.error(
                    "No remote 'origin' found and no override provided. Cannot fetch missing commits"
                )
                raise Exception("Cannot find commits and no remote is configured")

    # detect non-linear history
    is_ancestor_result = worker.exec(
        ["git", "merge-base", "--is-ancestor", start, end],
        workdir=dir,
    )
    is_linear = is_ancestor_result.returncode == 0
    cherry_pick_range = {}

    if not is_linear:
        logger.info(f"Using non-linear history logic with main branch {main_branch}")

        # 1. find the linear range on the main branch
        passing_and_failing_cmd = (
            worker.check_exec(
                [
                    "sh",
                    "-c",
                    f"git merge-base {start} {end} && git merge-base {end} {main_branch}",
                ],
                workdir=dir,
            )
            .stdout()
            .strip()
        )
        passing_main_commit, failing_main_commit = passing_and_failing_cmd.splitlines()

        # 2. find commits to cherry-pick from the failing branch
        cherry_pick_range[package] = f"{failing_main_commit}..{end}"

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

    return data, cherry_pick_range
