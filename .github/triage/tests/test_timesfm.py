import json
import logging
import os
import subprocess
import sys

import pytest

from jax_toolbox_triage import triage_tool as triage_module
from jax_toolbox_triage.args import parse_args
from jax_toolbox_triage.docker import DockerContainer
from jax_toolbox_triage.local import LocalContainer
from jax_toolbox_triage.triage_tool import TriageTool
from jax_toolbox_triage.versions import get_commits_and_dirs


def git(directory, *args):
    return subprocess.run(
        ["git", "-C", str(directory), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def source(tmp_path, monkeypatch):
    # Ignore developer Git signing/hooks, and keep commit timestamps deterministic.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    remote = tmp_path / "remote"
    remote.mkdir()
    git(remote, "init", "-b", "main")
    git(remote, "config", "user.name", "Test")
    git(remote, "config", "user.email", "test@example.com")
    (remote / "dtype.txt").write_text("float32\n")
    commits = {}
    for revision in range(-1, 3):
        date = f"2026-01-0{revision + 2}T00:00:00+00:00"
        monkeypatch.setenv("GIT_AUTHOR_DATE", date)
        monkeypatch.setenv("GIT_COMMITTER_DATE", date)
        (remote / "revision.txt").write_text(str(revision))
        git(remote, "add", ".")
        git(remote, "commit", "-m", f"Revision {revision}")
        commits[revision] = git(remote, "rev-parse", "HEAD")
    return remote, commits


@pytest.mark.parametrize("directory", [None, "timesfm", "timesfm-source"])
def test_timesfm_discovery(tmp_path, monkeypatch, source, directory):
    remote, commits = source
    prefix = tmp_path / "container"
    (prefix / "opt").mkdir(parents=True)
    packages = ["xla", "jax"] + ([] if directory is None else [directory])
    for package in packages:
        git(prefix, "clone", str(remote), f"opt/{package}")
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_PREFIX", str(prefix))
    args = parse_args(
        ["--passing-container", "good", "--failing-container", "bad", "--", "true"]
    )
    versions, dirs = get_commits_and_dirs(
        LocalContainer(logger=logging.getLogger()), args.optional_software
    )
    assert versions["jax"] == versions["xla"] == commits[2]
    if directory is None:
        assert "timesfm" not in versions
        assert "timesfm" not in dirs
    else:
        assert versions["timesfm"] == commits[2]
        assert dirs["timesfm"] == f"/opt/{directory}"


@pytest.mark.parametrize("runtime", ["docker", "plugin"])
@pytest.mark.parametrize("patched", [False, True])
@pytest.mark.parametrize(
    "candidate,passing_revision", [(None, 0), (0, 0), (1, 0), (2, 0), (1, 2)]
)
def test_timesfm_bisection(
    tmp_path, monkeypatch, source, runtime, patched, candidate, passing_revision
):
    """Exercise real shallow Git histories; stub GPU builds and pip installation."""
    remote, commits = source
    directories = {}
    for package in ["xla", "jax", "timesfm"]:
        git(tmp_path, "clone", "--depth=1", remote.as_uri(), package)
        directories[package] = str(tmp_path / package)
        git(directories[package], "config", "user.name", "Test")
        git(directories[package], "config", "user.email", "test@example.com")
    checkout = tmp_path / "timesfm"
    if patched:
        (checkout / "dtype.txt").write_text("bfloat16\n")
    monkeypatch.delenv("JAX_TOOLBOX_TRIAGE_PREFIX", raising=False)
    installed = tmp_path / "installed-revision.txt"
    installed.write_text("2")
    output = tmp_path / "output"
    output.mkdir()
    args = parse_args(
        [
            "--passing-container",
            "good",
            "--failing-container",
            "bad",
            "--container-runtime",
            runtime,
            "--output-prefix",
            str(output),
            "--exclude-transformer-engine",
            "--confirmation-iterations",
            "0",
        ]
        + ([] if candidate is None else ["--commit", f"timesfm:{commits[candidate]}"])
        + [
            "--",
            sys.executable,
            "-c",
            (
                "import pathlib, sys; "
                "sys.exit(int(pathlib.Path(sys.argv[1]).read_text()) >= 1)"
            ),
            str(installed),
        ]
    )
    tool = TriageTool(args, logging.getLogger())
    tool.package_dirs = directories
    tool.dynamic_packages = {"timesfm"} if passing_revision != 2 else set()
    tool.bisection_url = "bad"
    tool.bisection_versions = dict.fromkeys(directories, commits[2])
    worker = LocalContainer(logger=tool.logger)

    def container(*args, **kwargs):
        if "test_output_directory" in kwargs:
            # Each build starts from the original image, not the preceding checkout.
            git(checkout, "checkout", commits[2])
        return worker

    monkeypatch.setattr(tool, "_make_container", container)
    builds = []

    def build(command):
        assert "python3 -m pip install --no-deps --force-reinstall" in command
        assert str(checkout) in command
        assert (checkout / "dtype.txt").read_text() == (
            "bfloat16\n" if patched else "float32\n"
        )
        # Model a non-editable install: tests only see revisions copied by a build.
        installed.write_text((checkout / "revision.txt").read_text())
        builds.append(int(installed.read_text()))
        return subprocess.CompletedProcess([], 0, stdout="Build succeeded\n")

    original_exec = worker.exec

    def execute(command, **kwargs):
        if command[:2] == ["sh", "-c"] and "build-jax.sh" in command[2]:
            return build(command[2])
        return original_exec(command, **kwargs)

    monkeypatch.setattr(worker, "exec", execute)
    if runtime == "plugin":
        monkeypatch.setattr(DockerContainer, "exists", lambda self: False)

        def build_image(command, **kwargs):
            assert command[:3] == ["docker", "buildx", "build"]
            git(checkout, "checkout", commits[2])
            build_args = dict(
                arg[len("--build-arg=") :].split("=", 1)
                for arg in command
                if arg.startswith("--build-arg=")
            )
            original_exec(["sh", "-c", build_args["GIT_CMD"]]).check_returncode()
            return build(build_args["BUILD_CMD"])

        def plugin(*, output_prefix, **kwargs):
            result = original_exec(args.test_command)
            (output_prefix / "metrics.json").write_text(
                json.dumps({"exit_code": result.returncode})
            )
            return result

        monkeypatch.setattr(triage_module, "run_and_log", build_image)
        monkeypatch.setattr(tool, "_run_plugin", plugin)

    summary = tool.run_version_bisection(
        {**tool.bisection_versions, "timesfm": commits[passing_revision]},
        tool.bisection_versions,
    )
    assert summary["result"]["timesfm_bad"] == commits[1]
    assert summary["result"]["timesfm_good"] == commits[0]
    assert summary["result"]["jax_ref"] == commits[2]
    assert summary["result"]["xla_ref"] == commits[2]
    assert {0, 1} <= set(builds)
    if candidate == 1:
        assert builds == [1, 0]
