from jax_toolbox_triage.args import parse_args
from jax_toolbox_triage.container_factory import make_container
from jax_toolbox_triage.logic import CouldNotReproduceFailure, CouldNotReproduceSuccess
from jax_toolbox_triage.triage_tool import TriageTool
import logging
import os
import pathlib
import pytest
import shutil
import subprocess
import uuid

srun_available = shutil.which("srun") is not None and "SLURM_JOBID" in os.environ
docker_available = shutil.which("docker") is not None
skip_if_no_docker = pytest.mark.skipif(
    not docker_available, reason="Plugin backend needs docker"
)
mock_scripts_path = pathlib.Path(__file__).parent / "mock_scripts"

_BASE_CONTAINER = "ubuntu:24.04"


@pytest.fixture(scope="module")
def logger():
    logger = logging.getLogger("triage-tests")
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    return logger


@pytest.mark.parametrize("runtime", ["docker", "pyxis"])
def test_container_semantics(logger, runtime):
    if runtime == "docker" and not docker_available:
        pytest.skip("Docker is not available")
    elif runtime == "pyxis" and not srun_available:
        pytest.skip("Pyxis is not available")
    with make_container(
        runtime=runtime, url=_BASE_CONTAINER, mounts=[], logger=logger
    ) as worker:
        # The file should not exist in the container image we pulled
        assert worker.exec(["cat", "/root/token"]).returncode != 0
        token = uuid.uuid1().hex
        worker.check_exec(["sh", "-c", f"echo -n {token} > /root/token"])
        read_token = worker.check_exec(["cat", "/root/token"], stderr="separate").stdout
        # While the context manager is active, there is a single container instance
        assert token == read_token, (token, read_token)
    with make_container(
        runtime=runtime, url=_BASE_CONTAINER, mounts=[], logger=logger
    ) as worker:
        # We should not be able to see the old token
        assert worker.exec(["cat", "/root/token"]).returncode != 0


def build_container(logger, target):
    tag = uuid.uuid1().hex
    subprocess.run(
        [
            "docker",
            "buildx",
            "build",
            "--build-arg",
            f"BASE_IMAGE={_BASE_CONTAINER}",
            "--target",
            target,
            "--tag",
            tag,
            str(mock_scripts_path),
        ],
        check=True,
    )
    with make_container(runtime="docker", url=tag, mounts=[], logger=logger) as worker:
        jax_commit = worker.check_exec(
            ["git", "rev-parse", "HEAD"], workdir="/opt/jax"
        ).stdout.strip()
    return tag, jax_commit


@pytest.fixture(scope="module")
def passing_container(logger):
    return build_container(logger, "passing")


@pytest.fixture(scope="module")
def passing_container_with_later_version(logger):
    return build_container(logger, "passing_with_later_version")


@pytest.fixture(scope="module")
def failing_container(logger):
    return build_container(logger, "failing")


@pytest.fixture(scope="module")
def failing_container_with_later_version(logger):
    return build_container(logger, "failing_with_later_version")


@pytest.fixture
def run_with_plugin_backend(logger, tmp_path):
    def wrapped(*, tool=[], plugin=[]):
        arg_list = (
            [
                "--output-prefix",
                str(tmp_path),
                "--container-runtime",
                "plugin",
            ]
            + tool
            + ["--", str(mock_scripts_path / "mock_plugin.py")]
            + plugin
        )
        args = parse_args(arg_list)
        triage_tool = TriageTool(args, logger)
        passing_versions, failing_versions = triage_tool.gather_version_info(
            args.passing_container, args.failing_container
        )
        return triage_tool.run_version_bisection(passing_versions, failing_versions)

    return wrapped


@skip_if_no_docker
def test_plugin_backend_exit_code(
    run_with_plugin_backend,
    passing_container,
    failing_container,
):
    passing_container_url, good_jax_commit = passing_container
    failing_container_url, bad_jax_commit = failing_container
    result = run_with_plugin_backend(
        tool=[
            "--passing-container",
            passing_container_url,
            "--failing-container",
            failing_container_url,
        ],
    )
    assert result["result"]["jax_bad"] == bad_jax_commit
    assert result["result"]["jax_good"] == good_jax_commit
    assert "xla_ref" in result["result"]


metric_args = [
    "--metric-name",
    "test_metric",
    "--passing-metric",
    "1.0",
    "--failing-metric",
    "0.0",
]


@skip_if_no_docker
@pytest.mark.parametrize("exit_code", [0, 1])
def test_plugin_backend_metric(
    run_with_plugin_backend,
    passing_container,
    failing_container,
    exit_code,
):
    passing_container_url, good_jax_commit = passing_container
    failing_container_url, bad_jax_commit = failing_container
    result = run_with_plugin_backend(
        tool=[
            "--passing-container",
            passing_container_url,
            "--failing-container",
            failing_container_url,
        ]
        + metric_args,
        plugin=[
            # Tell mock_plugin.py what code to exit with
            "--exit-code",
            str(exit_code),
        ],
    )
    assert result["result"]["jax_bad"] == bad_jax_commit
    assert result["result"]["jax_good"] == good_jax_commit
    assert "xla_ref" in result["result"]


@skip_if_no_docker
def test_plugin_backend_metric_always_good(
    run_with_plugin_backend,
    passing_container,
    passing_container_with_later_version,
):
    """
    Passing container URLs that are different but where the returned metric value is
    always 'good' should yield an error.
    """
    passing_container_url, _ = passing_container
    failing_container_url, _ = passing_container_with_later_version
    with pytest.raises(CouldNotReproduceFailure):
        run_with_plugin_backend(
            tool=[
                "--passing-container",
                passing_container_url,
                "--failing-container",
                failing_container_url,
            ]
            + metric_args,
        )


@skip_if_no_docker
def test_plugin_backend_metric_always_bad(
    run_with_plugin_backend,
    failing_container,
    failing_container_with_later_version,
):
    """
    Passing container URLs that are different but where the returned metric value is
    always 'bad' should yield an error.
    """
    passing_container_url, _ = failing_container
    failing_container_url, _ = failing_container_with_later_version
    with pytest.raises(CouldNotReproduceSuccess):
        run_with_plugin_backend(
            tool=[
                "--passing-container",
                passing_container_url,
                "--failing-container",
                failing_container_url,
            ]
            + metric_args,
        )
