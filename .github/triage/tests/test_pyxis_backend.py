import functools
import logging
import pathlib
import pytest
import shutil
import subprocess
import tempfile


from jax_toolbox_triage.args import compulsory_software, parse_args
from jax_toolbox_triage.triage_tool import TriageTool


def git_cmd(*args, cwd=None):
    return subprocess.run(
        ["git"] + list(args),
        capture_output=True,
        check=True,
        cwd=cwd,
        text=True,
    ).stdout.strip()


@pytest.fixture(params=["linear", "non-linear"])
def passing_container(request):
    scenario = request.param
    with tempfile.TemporaryDirectory(suffix=f"-passing-{scenario}") as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        (temp_path / "root").mkdir()
        metadata = {"prefix": temp_path, "scenario": scenario}
        for project in compulsory_software:
            repo_path = temp_path / "opt" / project
            repo_path.mkdir(parents=True)
            git = functools.partial(git_cmd, cwd=repo_path)
            git("init", "-b", "main")
            git("config", "user.name", "Test User")
            git("config", "user.email", "test@example.com")
            if scenario == "non-linear":
                # Regular commits underneath F0
                git("commit", "--allow-empty", "-m", "C0")
                git("commit", "--allow-empty", "-m", "C1")
                git("checkout", "-b", "feature-1")
            # This is needed for the mock build-jax.sh to succeed
            (repo_path / "feature_file.txt").write_text("feature")
            git("add", "feature_file.txt")
            git("commit", "-m", "F0")
            metadata[f"{project}_feature_commit"] = git("rev-parse", "HEAD")
            if scenario == "linear":
                # Regular commits on top of F0, which is always present
                git("commit", "--allow-empty", "-m", "C0")
                git("commit", "--allow-empty", "-m", "C1")
            metadata[f"{project}_passing_container"] = git("rev-parse", "HEAD")
        yield metadata


@pytest.fixture
def failing_container(passing_container):
    scenario = passing_container["scenario"]
    with tempfile.TemporaryDirectory(suffix=f"-failing-{scenario}") as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        metadata = {"prefix": temp_path}
        shutil.copytree(passing_container["prefix"], temp_path, dirs_exist_ok=True)
        for project in compulsory_software:
            git = functools.partial(git_cmd, cwd=temp_path / "opt" / project)
            if scenario == "non-linear":
                git("checkout", "main")
            git("commit", "--allow-empty", "-m", "C2")
            metadata[f"{project}_good"] = git("rev-parse", "HEAD")
            git("commit", "--allow-empty", "-m", "C3")
            metadata[f"{project}_bad"] = git("rev-parse", "HEAD")
            git("commit", "--allow-empty", "-m", "C4")
            git("commit", "--allow-empty", "-m", "C5")
            git("commit", "--allow-empty", "-m", "C6")
            if scenario == "non-linear":
                git("checkout", "-b", "feature-2")
                git("cherry-pick", passing_container[f"{project}_feature_commit"])
            metadata[f"{project}_failing_container"] = git("rev-parse", "HEAD")
        yield metadata


@pytest.mark.parametrize("num_nodes", [1, 2])
@pytest.mark.parametrize("processes_per_node", [1, 2])
def test_mock_containers(
    monkeypatch,
    passing_container,
    failing_container,
    num_nodes,
    processes_per_node,
):
    # Doesn't really add anything to test both values
    bad_package = compulsory_software[0]
    # Tell the mock `srun` how to behave
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_NODES", str(num_nodes))
    monkeypatch.setenv(
        "JAX_TOOLBOX_TRIAGE_MOCK_SRUN_PROCS_PER_NODE", str(processes_per_node)
    )
    # Ensure bazel, build-jax.sh, srun etc. stubs can be found.
    monkeypatch.setenv(
        "PATH", str(pathlib.Path(__file__).parent / "mock_scripts"), prepend=":"
    )
    with tempfile.TemporaryDirectory() as output_prefix:
        arg_list = [
            # Currently no way to use the pyxis backend without a cache
            "--bazel-cache=https://example.com/does-not-exist",
            "--container-runtime=pyxis",
            "--output-prefix",
            output_prefix,
            "--passing-container",
            str(passing_container["prefix"]),
            "--failing-container",
            str(failing_container["prefix"]),
            "--",
            "test-case.sh",
            f"/opt/{bad_package}",
            failing_container[f"{bad_package}_bad"],
        ]
        args = parse_args(arg_list)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        tool = TriageTool(args, logger)
        # Check the correct versions are extracted from the two pseudocontainers
        passing_versions, failing_versions = tool.gather_version_info(
            args.passing_container, args.failing_container
        )
        for package in compulsory_software:
            assert (
                passing_versions[package]
                == passing_container[f"{package}_passing_container"]
            )
            assert (
                failing_versions[package]
                == failing_container[f"{package}_failing_container"]
            )
        # Run the bisection
        summary_data = tool.run_version_bisection(passing_versions, failing_versions)
        assert "result" in summary_data, summary_data
        assert (
            summary_data["result"][f"{bad_package}_good"]
            == failing_container[f"{bad_package}_good"]
        )
        assert (
            summary_data["result"][f"{bad_package}_bad"]
            == failing_container[f"{bad_package}_bad"]
        )
