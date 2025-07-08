import subprocess
import tempfile
import pathlib
import os
import logging
from collections import OrderedDict
import pytest

from jax_toolbox_triage.bisect import get_commit_history
from jax_toolbox_triage.logic import version_search, TestResult
from jax_toolbox_triage.container import Container


def run_command(command, cwd=None, env=None):
    """Simple function to run a command in a subprocess.

    Args:
        command (list): The command to run as a list of strings.
        cwd (str, optional): The working directory to run the command in.
        env (dict, optional): Environment variables to set for the command.
    Returns:
        str: The standard output of the command.
    """
    try:
        result = subprocess.run(
            command, cwd=cwd, env=env, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Command '{' '.join(command)}' failed with error: {e}")
        raise e


class MockContainer(Container):
    """A mock container class for testing purposes."""

    def __init__(self, mock_scripts_path, logger):
        super().__init__(logger=logger)
        self.mock_scripts_path = mock_scripts_path
        self._env = os.environ.copy()
        self._env["PATH"] = f"{self.mock_scripts_path}:{self._env['PATH']}"

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass

    def __repr__(self):
        return "MockContainer"

    def check_exec(self, cmd, **kwargs):
        """Override the check_exec"""
        return super().check_exec(cmd, **kwargs)

    def exec(
        self,
        command,
        *,
        policy="default",
        stderr="interleaved",
        workdir=None,
        log_level=logging.DEBUG,
    ):
        self._logger.debug(f"Executing command: {command} in {workdir}")
        is_shell_command = command[0] == "sh" and command[1] == "-c"
        cmd_to_run = command[2] if is_shell_command else command
        try:
            return subprocess.run(
                cmd_to_run,
                capture_output=True,
                text=True,
                cwd=workdir,
                env=self._env,
                shell=is_shell_command,
            )
        except FileNotFoundError as e:
            return subprocess.CompletedProcess(command, 127, stderr=str(e))

    def exists(self) -> bool:
        return True


@pytest.fixture
def triage_test_env():
    """
    Fixture to set up the test environment for triage tests.

    The fixture creates a temp directory and a git repo with a
    defined linear and non-linear history.

    The fixture yields a dictionary of paths and commit hashes
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        repo_path = temp_path / "repos"
        output_path = temp_path / "output"
        mock_scripts_path = temp_path / "mock_scripts"
        repo_path.mkdir()
        output_path.mkdir()
        mock_scripts_path.mkdir()

        # Generation of mock scripts
        # build-jax.sh
        source_scripts_dir = pathlib.Path(__file__).parent / "mock_scripts"
        build_script_content = (source_scripts_dir / "build-jax.sh").read_text()
        (mock_scripts_path / "build-jax.sh").write_text(build_script_content)
        os.chmod(mock_scripts_path / "build-jax.sh", 0o755)
        # test-case.sh helper test script
        test_case_content = (source_scripts_dir / "test-case.sh").read_text()
        (mock_scripts_path / "test-case.sh").write_text(test_case_content)
        os.chmod(mock_scripts_path / "test-case.sh", 0o755)

        # Create a git repository
        jax_repo_path = repo_path / "jax"
        jax_repo_path.mkdir()

        def git_cmd(command, *args):
            return run_command(["git", command, *args], cwd=jax_repo_path)

        # main
        git_cmd("init", "-b", "main")
        git_cmd("remote", "add", "origin", str(jax_repo_path))
        git_cmd("config", "user.name", "Test User")
        git_cmd("config", "user.email", "test@user.it")
        # Create a linear commit history
        git_cmd("commit", "--allow-empty", "-m", "M1")
        m1 = git_cmd("rev-parse", "HEAD")
        git_cmd("commit", "--allow-empty", "-m", "M2")  # good commit
        m2 = git_cmd("rev-parse", "HEAD")
        git_cmd("commit", "--allow-empty", "-m", "M3")  # bad commit
        m3 = git_cmd("rev-parse", "HEAD")
        # create a feature branch
        git_cmd("checkout", "-b", "feature", m1)
        (jax_repo_path / "feature_file.txt").write_text("feature")
        git_cmd("add", "feature_file.txt")
        git_cmd("commit", "-m", "F1")
        f1 = git_cmd("rev-parse", "HEAD")

        git_cmd("checkout", "-b", "passing_nonlinear", m2)
        git_cmd("cherry-pick", f1)
        passing_nonlinear = git_cmd("rev-parse", "HEAD")
        git_cmd("checkout", "-b", "failing_nonlinear", m3)
        git_cmd("cherry-pick", f1)
        failing_nonlinear = git_cmd("rev-parse", "HEAD")
        git_cmd("checkout", "main")

        # yield all the info
        yield {
            "paths": {
                "repo": repo_path,
                "output": output_path,
                "scripts": mock_scripts_path,
            },
            "commits": {
                "good_main": m2,
                "bad_main": m3,
                "feature": f1,
                "passing_nonlinear": passing_nonlinear,
                "failing_nonlinear": failing_nonlinear,
            },
        }


# TEST CASES
@pytest.mark.parametrize(
    "scenario, passing_commit_key, failing_commit_key, expected_good_key, expected_bad_key",
    [
        (
            "Non-Linear History",
            "passing_nonlinear",
            "failing_nonlinear",
            "good_main",
            "bad_main",
        ),
        ("Linear History", "good_main", "bad_main", "good_main", "bad_main"),
    ],
)
def test_triage_scenarios(
    triage_test_env,
    scenario,
    passing_commit_key,
    failing_commit_key,
    expected_good_key,
    expected_bad_key,
):
    """Check if we nee dot restructure this + add types"""
    paths = triage_test_env["paths"]
    all_commits = triage_test_env["commits"]
    jax_repo_path = paths["repo"] / "jax"

    class MockArgs:
        main_branch = "main"
        bazel_cache = ""
        build_scripts_path = None
        test_command = ["test-case.sh", str(jax_repo_path), all_commits["bad_main"]]
        cherry_pick_commits = {}

    args = MockArgs()
    logger = logging.getLogger(f"Scenario-{scenario}")
    logging.basicConfig(level=logging.INFO)

    passing_versions = {"jax": all_commits[passing_commit_key]}
    failing_versions = {"jax": all_commits[failing_commit_key]}
    package_dirs = {"jax": str(jax_repo_path)}
    mock_container = MockContainer(paths["scripts"], logger)

    package_versions = OrderedDict()
    package_versions["jax"] = get_commit_history(
        worker=mock_container,
        package="jax",
        start=passing_versions["jax"],
        end=failing_versions["jax"],
        dir=package_dirs["jax"],
        main_branch=args.main_branch,
        args=args,
        logger=logger,
    )

    # build and test
    def build_and_test_wrapper(*, versions, test_output_log_level=logging.DEBUG):
        workdir = package_dirs["jax"]
        mock_container.check_exec(
            ["git", "stash", "--include-untracked"], workdir=workdir
        )
        cherry_pick_commits = args.cherry_pick_commits.get("jax", [])

        if cherry_pick_commits:
            build_script = paths["scripts"] / "build-jax.sh"
            mock_container.check_exec(
                ["git", "checkout", versions["jax"]], workdir=workdir
            )
            mock_container.check_exec(
                ["git", "cherry-pick"] + cherry_pick_commits, workdir=workdir
            )
        else:
            build_script = paths["scripts"] / "build-jax-linear.sh"
            build_script.write_text("#!/bin/sh\nexit 0")
            os.chmod(build_script, 0o755)
            mock_container.check_exec(
                ["git", "checkout", versions["jax"]], workdir=workdir
            )

        mock_container.check_exec([str(build_script)], workdir=workdir)
        result = mock_container.exec(args.test_command, workdir=workdir)
        return TestResult(
            host_output_directory=paths["output"],
            result=result.returncode == 0,
            stdouterr=" ",
        )

    # bisection
    result, _, _ = version_search(
        versions=package_versions,
        build_and_test=build_and_test_wrapper,
        logger=logger,
        skip_precondition_checks=False,
    )

    # test
    assert result.get("jax_good") == all_commits[expected_good_key]
    assert result.get("jax_bad") == all_commits[expected_bad_key]
