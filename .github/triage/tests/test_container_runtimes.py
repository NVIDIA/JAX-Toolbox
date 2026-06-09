from jax_toolbox_triage.container_factory import make_container
import logging
import os
import pytest
import shutil
import uuid

srun_available = shutil.which("srun") is not None and "SLURM_JOBID" in os.environ
docker_available = shutil.which("docker") is not None


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
    url = "ubuntu:24.04"
    with make_container(runtime=runtime, url=url, mounts=[], logger=logger) as worker:
        # The file should not exist in the container image we pulled
        assert worker.exec(["cat", "/root/token"]).returncode != 0
        token = uuid.uuid1().hex
        worker.check_exec(["sh", "-c", f"echo -n {token} > /root/token"])
        read_token = worker.check_exec(["cat", "/root/token"], stderr="separate").stdout
        # While the context manager is active, there is a single container instance
        assert token == read_token, (token, read_token)
    with make_container(runtime=runtime, url=url, mounts=[], logger=logger) as worker:
        # We should not be able to see the old token
        assert worker.exec(["cat", "/root/token"]).returncode != 0
