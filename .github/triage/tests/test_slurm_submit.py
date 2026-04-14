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
from jax_toolbox_triage.logic import CouldNotReproduceFailure
from jax_toolbox_triage.triage_tool import InconsistentResults, TriageTool

mock_scripts_path = pathlib.Path(__file__).parent / "mock_scripts"

slurm_args = [
    "--bazel-cache=https://example.com/does-not-exist",
    "--container-runtime=slurm",
    "--slurm-account=test-account",
    "--slurm-partition=test-partition",
    "--slurm-num-nodes=1",
    "--slurm-ntasks-per-node=1",
    # poll_interval=0 avoids any real sleep; squeue mock returns empty immediately
    "--slurm-poll-interval=0",
]


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


def _make_tool(arg_list, output_prefix):
    args = parse_args(arg_list)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return TriageTool(args, logger), args


def test_mock_containers(monkeypatch, passing_container, failing_container):
    """Full bisection converges to the correct bad commit via sbatch jobs.

    The container URL template encodes the commit under test as the last path
    component.  slurm-test-case.sh extracts the commit from the URL and checks
    git ancestry to determine pass/fail without checking out anything.
    """
    bad_package = compulsory_software[0]
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_NODES", "1")
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_PROCS_PER_NODE", "1")
    monkeypatch.setenv("PATH", str(mock_scripts_path), prepend=":")

    # Encode the bad_package commit as the last URL path component so the mock
    # test script can check ancestry without a git checkout inside the job.
    url_template = f"{failing_container['prefix']}/{{{bad_package}}}"

    with tempfile.TemporaryDirectory() as output_prefix:
        arg_list = slurm_args + [
            f"--slurm-container-url-template={url_template}",
            "--output-prefix",
            output_prefix,
            "--passing-container",
            str(passing_container["prefix"]),
            "--failing-container",
            str(failing_container["prefix"]),
            "--",
            "slurm-test-case.sh",
            f"/opt/{bad_package}",
            failing_container[f"{bad_package}_bad"],
            str(failing_container["prefix"]),
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
    Verify that sbatch job scripts are written to the expected directory,
    contain the required #SBATCH headers, and contain only the test command
    (no git checkout or build steps).
    """
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_NODES", "1")
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_PROCS_PER_NODE", "1")
    monkeypatch.setenv("PATH", str(mock_scripts_path), prepend=":")

    bad_package = compulsory_software[0]
    url_template = f"{failing_container['prefix']}/{{{bad_package}}}"

    with tempfile.TemporaryDirectory() as output_prefix:
        arg_list = slurm_args + [
            f"--slurm-container-url-template={url_template}",
            "--output-prefix",
            output_prefix,
            "--passing-container",
            str(passing_container["prefix"]),
            "--failing-container",
            str(failing_container["prefix"]),
            "--",
            "slurm-test-case.sh",
            f"/opt/{bad_package}",
            failing_container[f"{bad_package}_bad"],
            str(failing_container["prefix"]),
        ]
        tool, args = _make_tool(arg_list, output_prefix)
        passing_versions, failing_versions = tool.gather_version_info(
            args.passing_container, args.failing_container
        )
        tool.run_version_bisection(passing_versions, failing_versions)

        slurm_jobs_dir = pathlib.Path(output_prefix) / "slurm-jobs"
        scripts = list(slurm_jobs_dir.rglob("job_*.sh"))
        assert len(scripts) > 0, "No job scripts were written"

        for script in scripts:
            content = script.read_text()
            assert "#SBATCH --account=test-account" in content, script
            assert "#SBATCH --partition=test-partition" in content, script
            assert "#SBATCH --nodes=" in content, script
            assert "#SBATCH --ntasks-per-node=" in content, script
            assert "#SBATCH --output=" in content, script
            assert "#SBATCH --error=" in content, script
            assert "srun" in content, script
            # SLURM jobs must NOT contain build steps
            assert "build-jax.sh" not in content, script
            assert "git checkout" not in content, script


def test_output_files_are_read_back(monkeypatch, passing_container, failing_container):
    """
    Verify that exit-code files are created for every submitted job and that
    they contain a valid integer exit code.
    """
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_NODES", "1")
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_PROCS_PER_NODE", "1")
    monkeypatch.setenv("PATH", str(mock_scripts_path), prepend=":")

    bad_package = compulsory_software[0]
    url_template = f"{failing_container['prefix']}/{{{bad_package}}}"

    with tempfile.TemporaryDirectory() as output_prefix:
        arg_list = slurm_args + [
            f"--slurm-container-url-template={url_template}",
            "--output-prefix",
            output_prefix,
            "--passing-container",
            str(passing_container["prefix"]),
            "--failing-container",
            str(failing_container["prefix"]),
            "--",
            "slurm-test-case.sh",
            f"/opt/{bad_package}",
            failing_container[f"{bad_package}_bad"],
            str(failing_container["prefix"]),
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


def test_container_name_consistency(monkeypatch, passing_container, failing_container):
    """
    All job scripts within the same SlurmJobContainer instance must use the
    same --container-name (derived from URL hash + process token), confirming
    that enroot reuses the same cached image across calls.
    """
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_NODES", "1")
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_PROCS_PER_NODE", "1")
    monkeypatch.setenv("PATH", str(mock_scripts_path), prepend=":")

    bad_package = compulsory_software[0]
    url_template = f"{failing_container['prefix']}/{{{bad_package}}}"

    with tempfile.TemporaryDirectory() as output_prefix:
        arg_list = slurm_args + [
            f"--slurm-container-url-template={url_template}",
            "--output-prefix",
            output_prefix,
            "--passing-container",
            str(passing_container["prefix"]),
            "--failing-container",
            str(failing_container["prefix"]),
            "--",
            "slurm-test-case.sh",
            f"/opt/{bad_package}",
            failing_container[f"{bad_package}_bad"],
            str(failing_container["prefix"]),
        ]
        tool, args = _make_tool(arg_list, output_prefix)
        passing_versions, failing_versions = tool.gather_version_info(
            args.passing_container, args.failing_container
        )
        tool.run_version_bisection(passing_versions, failing_versions)

        slurm_jobs_dir = pathlib.Path(output_prefix) / "slurm-jobs"
        for ctr_dir in slurm_jobs_dir.iterdir():
            scripts = sorted(ctr_dir.glob("job_*.sh"))
            if len(scripts) < 2:
                continue
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
    If the test always passes regardless of the commit being tested the tool
    must raise InconsistentResults (failing_versions should fail, but doesn't).
    """
    bad_package = compulsory_software[0]
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_NODES", "1")
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_PROCS_PER_NODE", "1")
    monkeypatch.setenv("PATH", str(mock_scripts_path), prepend=":")

    url_template = f"{failing_container['prefix']}/{{{bad_package}}}"

    with tempfile.TemporaryDirectory() as output_prefix:
        arg_list = (
            slurm_args
            + [
                f"--slurm-container-url-template={url_template}",
                "--output-prefix",
                output_prefix,
                "--passing-container",
                str(passing_container["prefix"]),
                "--failing-container",
                str(failing_container["prefix"]),
                "--",
                "true",  # always succeeds → failing_versions won't fail → InconsistentResults
            ]
        )
        tool, args = _make_tool(arg_list, output_prefix)
        passing_versions, failing_versions = tool.gather_version_info(
            args.passing_container, args.failing_container
        )
        with pytest.raises((InconsistentResults, CouldNotReproduceFailure)):
            tool.run_version_bisection(passing_versions, failing_versions)
