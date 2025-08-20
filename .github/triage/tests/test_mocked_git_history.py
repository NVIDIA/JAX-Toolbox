import subprocess
import tempfile
import pathlib
import logging
import pytest

from jax_toolbox_triage.args import parse_args
from jax_toolbox_triage.triage_tool import TriageTool


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

        def git_cmd(*args):
            return subprocess.run(
                ["git"] + list(args),
                capture_output=True,
                check=True,
                cwd=jax_repo_path,
                text=True,
            ).stdout.strip()

        # NON-LINEAR HISTORY
        git_cmd("init", "-b", "main")
        git_cmd("config", "user.name", "Test User")
        git_cmd("config", "user.email", "test@example.com")
        # Create a linear commit history
        git_cmd("commit", "--allow-empty", "-m", "M0_base")
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
        (jax_repo_path / "test.txt").write_text("blub\nglub\nbloop\n")
        git_cmd("add", "test.txt")
        git_cmd("commit", "-m", "L1")
        l1_good_commit = git_cmd("rev-parse", "HEAD")  # L1
        git_cmd("commit", "--allow-empty", "-m", "L2_BAD")  # L2 bad commit
        l2_bad_linear_commit = git_cmd("rev-parse", "HEAD")
        (jax_repo_path / "test.txt").write_text("blub\nglub\nargh\nbloop\n")
        git_cmd("add", "test.txt")
        git_cmd("commit", "-m", "L3")  # L3
        l3_failing_linear = git_cmd("rev-parse", "HEAD")
        # so the linear repo would be
        # M1 -- M2 -- M3
        # |
        # F1
        # |
        # L1 -- L2 -- L3

        # LINEAR HISTORY with container bug
        # base commit is F1, containing feature_file.txt, so build-jax.sh always works
        # L1'       L3'
        # |         |
        # F1 - L1 - L2 - L3
        # without mitigation, non-linear logic will identify [F1, L2] as the true range
        # and cherry-pick L3' onto it
        git_cmd("checkout", "-b", "buggy_l1", passing_nonlinear)  # F1
        git_cmd("cherry-pick", "-x", l1_good_commit)
        l1_rehashed = git_cmd("rev-parse", "HEAD")
        assert l1_rehashed != l1_good_commit
        git_cmd("checkout", "-b", "buggy_l3", l2_bad_linear_commit)
        git_cmd("cherry-pick", "-x", l3_failing_linear)
        l3_rehashed = git_cmd("rev-parse", "HEAD")
        assert l3_rehashed != l3_failing_linear

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
                "good_linear_rehashed": l1_rehashed,
                "failing_linear_rehashed": l3_rehashed,
            },
        }


@pytest.mark.parametrize(
    "passing_commit_key,failing_commit_key,bad_commit_key,expected_good_key,extra_args",
    [
        # Non-linear history
        (
            "passing_nonlinear",
            "failing_nonlinear",
            "bad_main",
            "good_main",
            [],
        ),
        # Linear history
        (
            "good_linear",
            "failing_linear",
            "bad_commit_for_linear",
            "good_linear",
            [],
        ),
        # Buggy linear history
        (
            "good_linear_rehashed",
            "failing_linear_rehashed",
            "bad_commit_for_linear",
            "good_linear",
            [
                "--main-branch",
                "linear_feature_branch",
                "--workaround-buggy-container",
                "jax",
            ],
        ),
    ],
)
def test_triage_scenarios(
    caplog,
    triage_env,
    monkeypatch,
    passing_commit_key,
    failing_commit_key,
    bad_commit_key,
    expected_good_key,
    extra_args,
):
    """Test the get_commit_history for linear and non-linear histories."""
    caplog.set_level(logging.DEBUG)
    paths = triage_env["paths"]
    all_commits = triage_env["commits"]
    jax_repo_path = paths["repo"] / "jax"
    passing_versions = {"jax": all_commits[passing_commit_key]}
    failing_versions = {"jax": all_commits[failing_commit_key]}
    passing_versions_str = f"jax:{all_commits[passing_commit_key]}"
    failing_versions_str = f"jax:{all_commits[failing_commit_key]}"

    arg_list = (
        [
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
        ]
        + extra_args
        + [
            "--",
            str(paths["scripts"] / "test-case.sh"),
            str(jax_repo_path),
            all_commits[bad_commit_key],
        ]
    )
    args = parse_args(arg_list)
    logger = logging.getLogger()
    # Ensure bazel and build-jax.sh can be found.
    monkeypatch.setenv("PATH", str(paths["scripts"]), prepend=":")

    tool = TriageTool(args, logger)
    tool.package_dirs = {"jax": str(jax_repo_path)}
    tool.dynamic_packages = {"jax"}
    tool.bisection_url = "local"

    # run the bisection
    summary_data = tool.run_version_bisection(passing_versions, failing_versions)

    assert "result" in summary_data, "No result section was created"
    result = summary_data["result"]

    assert result.get("jax_good") == all_commits[expected_good_key]
    assert result.get("jax_bad") == all_commits[bad_commit_key]
