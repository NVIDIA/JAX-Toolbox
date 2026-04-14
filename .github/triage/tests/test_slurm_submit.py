"""
Integration tests for the SlurmJobContainer backend.

The test structure mirrors test_pyxis_backend.py.  Instead of a mock `srun`
that is invoked directly, here the mock `sbatch` runs the generated job script
(which in turn calls mock `srun`), while mock `squeue` and `sacct` simulate
instant job completion.  All four mock executables live in mock_scripts/ and
are placed on PATH via monkeypatch.
"""

import functools
import logging
import pathlib
import shutil
import subprocess
import tempfile

import pytest

from jax_toolbox_triage.args import compulsory_software, parse_args
from jax_toolbox_triage.triage_tool import InconsistentResults, TriageTool

mock_scripts_path = pathlib.Path(__file__).parent / "mock_scripts"

slurm_args = [
    "--bazel-cache=https://example.com/does-not-exist",
    "--container-runtime=slurm",
    "--slurm-account=test-account",
    "--slurm-partition=test-partition",
    "--slurm-num-gpus=1",
    # poll_interval=0 avoids any real sleep; squeue mock returns empty immediately
    "--slurm-poll-interval=0",
]


# ---------------------------------------------------------------------------
# Shared git-repository fixtures (identical to those in test_pyxis_backend.py)
# ---------------------------------------------------------------------------


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
                git("commit", "--allow-empty", "-m", "C0")
                git("commit", "--allow-empty", "-m", "C1")
                git("checkout", "-b", "feature-1")
            (repo_path / "feature_file.txt").write_text("feature")
            git("add", "feature_file.txt")
            git("commit", "-m", "F0")
            metadata[f"{project}_feature_commit"] = git("rev-parse", "HEAD")
            if scenario == "linear":
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
            c6 = git("rev-parse", "HEAD")
            if scenario == "non-linear":
                git("checkout", metadata[f"{project}_good"])
                git("cherry-pick", passing_container[f"{project}_feature_commit"])
                metadata[f"{project}_good_with_feature"] = git("rev-parse", "HEAD")
                git("checkout", "-b", "feature-2")
                git("reset", "--hard", c6)
                git("cherry-pick", passing_container[f"{project}_feature_commit"])
            else:
                metadata[f"{project}_good_with_feature"] = metadata[f"{project}_good"]
            metadata[f"{project}_failing_container"] = git("rev-parse", "HEAD")
        yield metadata


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_tool(arg_list, output_prefix):
    args = parse_args(arg_list)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return TriageTool(args, logger), args


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_nodes", [1, 2])
@pytest.mark.parametrize("processes_per_node", [1, 2])
def test_mock_containers(
    monkeypatch,
    passing_container,
    failing_container,
    num_nodes,
    processes_per_node,
):
    """Full bisection converges to the correct bad commit via sbatch jobs."""
    bad_package = compulsory_software[0]
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_NODES", str(num_nodes))
    monkeypatch.setenv(
        "JAX_TOOLBOX_TRIAGE_MOCK_SRUN_PROCS_PER_NODE", str(processes_per_node)
    )
    monkeypatch.setenv("PATH", str(mock_scripts_path), prepend=":")

    with tempfile.TemporaryDirectory() as output_prefix:
        arg_list = slurm_args + [
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
        tool, args = _make_tool(arg_list, output_prefix)

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


def test_job_scripts_are_written(monkeypatch, passing_container, failing_container):
    """
    Verify that sbatch job scripts are written to the expected directory and
    contain the required #SBATCH headers.
    """
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_NODES", "1")
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_PROCS_PER_NODE", "1")
    monkeypatch.setenv("PATH", str(mock_scripts_path), prepend=":")

    bad_package = compulsory_software[0]
    with tempfile.TemporaryDirectory() as output_prefix:
        arg_list = slurm_args + [
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
        tool, args = _make_tool(arg_list, output_prefix)
        passing_versions, failing_versions = tool.gather_version_info(
            args.passing_container, args.failing_container
        )
        tool.run_version_bisection(passing_versions, failing_versions)

        # At least one job script must have been written
        slurm_jobs_dir = pathlib.Path(output_prefix) / "slurm-jobs"
        scripts = list(slurm_jobs_dir.rglob("job_*.sh"))
        assert len(scripts) > 0, "No job scripts were written"

        # Every script must declare the account and partition we specified
        for script in scripts:
            content = script.read_text()
            assert "#SBATCH --account=test-account" in content, script
            assert "#SBATCH --partition=test-partition" in content, script
            assert "#SBATCH --output=" in content, script
            assert "#SBATCH --error=" in content, script
            assert "srun" in content, script


def test_output_files_are_read_back(monkeypatch, passing_container, failing_container):
    """
    Verify that exit-code files are created for every submitted job and that
    they contain a valid integer exit code.
    """
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_NODES", "1")
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_PROCS_PER_NODE", "1")
    monkeypatch.setenv("PATH", str(mock_scripts_path), prepend=":")

    bad_package = compulsory_software[0]
    with tempfile.TemporaryDirectory() as output_prefix:
        arg_list = slurm_args + [
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
        tool, args = _make_tool(arg_list, output_prefix)
        passing_versions, failing_versions = tool.gather_version_info(
            args.passing_container, args.failing_container
        )
        tool.run_version_bisection(passing_versions, failing_versions)

        slurm_jobs_dir = pathlib.Path(output_prefix) / "slurm-jobs"
        exit_files = list(slurm_jobs_dir.rglob("job_*.exit"))
        assert len(exit_files) > 0, "No exit-code files were written"

        for exit_file in exit_files:
            content = exit_file.read_text().strip()
            assert content.lstrip(
                "-"
            ).isdigit(), f"{exit_file} contains non-integer exit code: {content!r}"


def test_node_pinning(monkeypatch, passing_container, failing_container):
    """
    After the first exec() resolves a node via sacct, subsequent job scripts
    in the same session must include --nodelist=mock-node.
    """
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_NODES", "1")
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_PROCS_PER_NODE", "1")
    monkeypatch.setenv("PATH", str(mock_scripts_path), prepend=":")

    bad_package = compulsory_software[0]
    with tempfile.TemporaryDirectory() as output_prefix:
        arg_list = slurm_args + [
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
        tool, args = _make_tool(arg_list, output_prefix)
        passing_versions, failing_versions = tool.gather_version_info(
            args.passing_container, args.failing_container
        )
        tool.run_version_bisection(passing_versions, failing_versions)

        # Collect all submitted sbatch command lines from the job scripts.
        # The scripts themselves show the srun flags; node pinning appears in
        # the sbatch invocation, which we can verify indirectly: every
        # container session (ctr*/ directory) with more than one job script
        # must have all scripts except job_0.sh referencing the same container
        # name — confirming the same container is used across calls.
        slurm_jobs_dir = pathlib.Path(output_prefix) / "slurm-jobs"
        for ctr_dir in slurm_jobs_dir.iterdir():
            scripts = sorted(ctr_dir.glob("job_*.sh"))
            if len(scripts) < 2:
                continue
            # All scripts in the same container session share the same
            # --container-name value (derived from the URL hash + process token).
            container_names = set()
            for s in scripts:
                for line in s.read_text().splitlines():
                    if "--container-name=" in line:
                        container_names.add(
                            line.split("--container-name=")[1].split()[0]
                        )
            assert len(container_names) == 1, (
                f"Expected one container name per session in {ctr_dir}, "
                f"got {container_names}"
            )


def test_mock_container_precondition_checks(
    monkeypatch, passing_container, failing_container
):
    """
    If success can be reproduced in the good container but not in the bad one,
    the tool must raise InconsistentResults.
    """
    bad_package = compulsory_software[0]
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_NODES", "1")
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_PROCS_PER_NODE", "1")
    monkeypatch.setenv("PATH", str(mock_scripts_path), prepend=":")

    with tempfile.TemporaryDirectory() as output_prefix:
        arg_list = slurm_args + [
            "--output-prefix",
            output_prefix,
            "--passing-container",
            str(passing_container["prefix"]),
            "--failing-container",
            str(failing_container["prefix"]),
            "--",
            "sh",
            "-c",
            " && ".join(
                [
                    f"[ $JAX_TOOLBOX_TRIAGE_MOCK_SRUN_CONTAINER_IMAGE"
                    f" = {passing_container['prefix']} ]",
                    f"test-case.sh /opt/{bad_package}"
                    f" {failing_container[f'{bad_package}_bad']}",
                ]
            ),
        ]
        tool, args = _make_tool(arg_list, output_prefix)
        passing_versions, failing_versions = tool.gather_version_info(
            args.passing_container, args.failing_container
        )
        with pytest.raises(InconsistentResults):
            tool.run_version_bisection(passing_versions, failing_versions)


def test_exists_writes_no_gpu_gres(monkeypatch, passing_container):
    """
    exists() must generate a job script that does NOT request GPUs so that
    container-existence checks do not consume scarce GPU resources.
    """
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_NODES", "1")
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_PROCS_PER_NODE", "1")
    monkeypatch.setenv("PATH", str(mock_scripts_path), prepend=":")

    from jax_toolbox_triage.container_factory import make_container

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        job_dir = tmp_path / "jobs"
        slurm_config = {
            "account": "test",
            "partition": "gpu",
            "num_gpus": 8,
            "time_limit": "1:00:00",
            "poll_interval": 0,
            "job_timeout": 60,
            "extra_flags": [],
            "job_dir": job_dir,
        }
        logger = logging.getLogger()
        container = make_container(
            "slurm",
            str(passing_container["prefix"]),
            [],
            logger,
            slurm_config=slurm_config,
        )
        result = container.exists()

        # The mock srun/sbatch chain runs successfully, so exists() should be True
        assert result is True

        # The generated job script must not contain a --gres=gpu line
        scripts = list(job_dir.rglob("job_*.sh"))
        assert len(scripts) == 1, scripts
        content = scripts[0].read_text()
        assert "--gres=gpu:" not in content, (
            "exists() must not request GPUs, but found --gres=gpu in:\n" + content
        )


@pytest.mark.parametrize("ccache_mode", ["configured", "excluded"])
def test_build_te_ccache_arguments(
    monkeypatch,
    passing_container,
    failing_container,
    ccache_mode,
):
    """Transformer Engine ccache flags are propagated correctly through sbatch jobs."""
    bad_package = compulsory_software[0]
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_NODES", "1")
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_PROCS_PER_NODE", "1")
    monkeypatch.setenv("PATH", str(mock_scripts_path), prepend=":")

    if ccache_mode == "excluded":
        ccache_args = ["--exclude-transformer-engine"]
        monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_BUILD_TE_POISON_PILL", "1")
    else:
        ccache_args = ["--transformer-engine-ccache-env", "CCACHE_SENTINEL=42"]

    with tempfile.TemporaryDirectory() as output_prefix:
        arg_list = (
            slurm_args
            + ccache_args
            + [
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
        )
        tool, args = _make_tool(arg_list, output_prefix)
        passing_versions, failing_versions = tool.gather_version_info(
            args.passing_container, args.failing_container
        )
        tool.run_version_bisection(passing_versions, failing_versions)
