import typing
from .container import Container
from .args import compulsory_software, optional_software


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
            f"(cd ${{JAX_TOOLBOX_TRIAGE_PREFIX}}/opt/{package} && git rev-parse HEAD && echo {package} && echo /opt/{package})",
            f"(cd ${{JAX_TOOLBOX_TRIAGE_PREFIX}}/opt/{package}-source && git rev-parse HEAD && echo {package} && echo /opt/{package}-source)",
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
    missing_compulsory_software = set(compulsory_software) - versions.keys()
    assert len(missing_compulsory_software) == 0, (
        f"Only found git repositories for {set(versions.keys())}, missing {missing_compulsory_software}"
    )
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
