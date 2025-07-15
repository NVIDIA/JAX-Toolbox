import subprocess
import tempfile
import pathlib
import logging
import pytest

from jax_toolbox_triage.args import parse_args
from jax_toolbox_triage.triage_tool import TriageTool


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


@pytest.fixture
def triage_env():
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
        source_scripts_dir = pathlib.Path(__file__).parent / "mock_scripts"
        repo_path.mkdir()
        output_path.mkdir()
        jax_repo_path = repo_path / "jax"
        jax_repo_path.mkdir()

        def git_cmd(command, *args):
            return run_command(["git", command, *args], cwd=jax_repo_path)

        # NON-LINEAR HISTORY
        # why don't we push the scripts and use them in repo
        git_cmd("init", "-b", "main")
        git_cmd("config", "user.name", "Test User")
        git_cmd("config", "user.email", "test@example.com")
        # Create a linear commit history
        git_cmd("commit" "--allow-empty", "-m", "M0_base")
        git_cmd("commit", "--allow-empty", "-m", "M1")
        m1 = git_cmd("rev-parse", "HEAD")
        git_cmd("commit", "--allow-empty", "-m", "M2")  # good commit
        m2 = git_cmd("rev-parse", "HEAD")
        git_cmd("commit", "--allow-empty", "-m", "M3")  # bad commit
        m3 = git_cmd("rev-parse", "HEAD")
        # create a feature branch from  M1
        git_cmd("checkout", "-b", "feature", m1)
        (jax_repo_path / "feature_file.txt").write_text("feature")
        git_cmd("add", "feature_file.txt")
        git_cmd("commit", "-m", "F1")
        passing_nonlinear = git_cmd("rev-parse", "HEAD")  # F1
        # and then we apply the feature to the bad commit
        # this simulated the rebase scenario
        git_cmd("checkout", "-b", "failing_nonlinear", m3)
        git_cmd("cherry-pick", passing_nonlinear)
        failing_nonlinear = git_cmd("rev-parse", "HEAD")  # F1'
        # so now we have:
        # M0--M1 --- M2 --- M3
        #     |             |
        #     F1           F1'
        # where F1 = passing
        # and F1' = failing

        # LINEAR HISTORY
        git_cmd("checkout", "-b", "linear_feature_branch", passing_nonlinear)
        git_cmd("commit", "--allow-empty", "-m", "L1")
        l1_good_commit = git_cmd("rev-parse", "HEAD")  # L1
        git_cmd("commit", "--allow-empty", "-m", "L2_BAD")  # L2 bad commit
        l2_bad_linear_commit = git_cmd("rev-parse", "HEAD")
        git_cmd("commit", "--allow-empty", "-m", "L3")  # L3
        l3_failing_linear = git_cmd("rev-parse", "HEAD")
        # so the linear repo would be
        # M1 -- M2 -- M3
        # |
        # F1
        # |
        # L1 -- L2 -- L3

        # yield all the info
        yield {
            "paths": {
                "repo": repo_path,
                "output": output_path,
                "scripts": source_scripts_dir,
            },
            "commits": {
                "good_main": m2,  # last good commit
                "bad_main": m3,
                "passing_nonlinear": passing_nonlinear,
                "failing_nonlinear": failing_nonlinear,
                "good_linear": l1_good_commit,
                "bad_commit_for_linear": l2_bad_linear_commit,
                "failing_linear": l3_failing_linear,
            },
        }


@pytest.mark.parametrize(
    "scenario, passing_commit_key, failing_commit_key, bad_commit_key, expected_good_key, expected_bad_key",
    [
        (
            "Non-Linear History",
            "passing_nonlinear",
            "failing_nonlinear",
            "bad_main",
            "good_main",
            "bad_main",
        ),
        (
            "Linear History",
            "good_linear",
            "failing_linear",
            "bad_commit_for_linear",
            "good_linear",
            "bad_commit_for_linear",
        ),
    ],
)
def test_triage_scenarios(
    triage_env,
    monkeypatch,
    scenario,
    passing_commit_key,
    failing_commit_key,
    bad_commit_key,
    expected_good_key,
    expected_bad_key,
):
    """Test the get_commit_history for linear and non-linear histories."""
    paths = triage_env["paths"]
    all_commits = triage_env["commits"]
    jax_repo_path = paths["repo"] / "jax"
    passing_versions = {"jax": all_commits[passing_commit_key]}
    failing_versions = {"jax": all_commits[failing_commit_key]}
    passing_versions_str = f"jax:{all_commits[passing_commit_key]}"
    failing_versions_str = f"jax:{all_commits[failing_commit_key]}"

    bazel_cache_path = (paths["output"] / "bazel-cache").resolve()
    bazel_cache_path.mkdir()

    arg_list = [
        "--main-branch",
        "main",
        "--output-prefix",
        str(paths["output"]),
        "--container-runtime",
        "local",
        "--passing-versions",
        passing_versions_str,
        "--failing-versions",
        failing_versions_str,
        "--bazel-cache",
        str(bazel_cache_path),
        "--",
        str(paths["scripts"] / "test-case.sh"),
        str(jax_repo_path),
        all_commits[bad_commit_key],
    ]
    args = parse_args(arg_list)
    logger = logging.getLogger(f"Scenario-{scenario}")
    logging.basicConfig(level=logging.INFO)
    # Ensure bazel and build-jax.sh can be found.
    monkeypatch.setenv("PATH", paths["scripts"], prepend=":")

    tool = TriageTool(args, logger)
    tool.package_dirs = {"jax": str(jax_repo_path)}
    tool.dynamic_packages = {"jax"}
    tool.bisection_url = "local"

    # run the bisection
    summary_data = tool.run_version_bisection(passing_versions, failing_versions)

    assert "result" in summary_data, "No result section was created"
    result = summary_data["result"]

    assert result.get("jax_good") == all_commits[expected_good_key]
    assert result.get("jax_bad") == all_commits[expected_bad_key]
