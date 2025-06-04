import hashlib
import logging
import pathlib
import secrets
import subprocess
import typing

from .container import Container
from .utils import run_and_log

# Used to make sure the {url: name} mapping is consistent within a process, but that
# names are not re-used by different invocations of the triage tool. Using a consistent
# mapping is a simple way of avoiding re-creating the container multiple times, e.g.
# for .exists()
_process_token = secrets.token_bytes()


class PyxisContainer(Container):
    def __init__(
        self,
        url: str,
        *,
        logger: logging.Logger,
        mounts: typing.List[typing.Tuple[pathlib.Path, pathlib.Path]],
    ):
        super().__init__(logger=logger)
        mount_str = ",".join(map(lambda t: f"{t[0]}:{t[1]}", mounts))
        self._mount_args = [f"--container-mounts={mount_str}"] if mount_str else []
        self._name = hashlib.sha256(url.encode() + _process_token).hexdigest()
        self._url = url

    def __enter__(self):
        self._logger.debug(f"Launching {self}")
        # Workaround for pyxis backend with some bazel versions
        # https://github.com/bazelbuild/bazel/issues/22955#issuecomment-2293899428
        self.check_exec(
            [
                "sh",
                "-c",
                "echo 'startup --host_jvm_args=-XX:-UseContainerSupport' > /root/.bazelrc",
            ]
        )
        return self

    def __exit__(self, *exc_info):
        pass

    def __repr__(self):
        return f"Pyxis({self._url})"

    def exec(
        self,
        command: typing.List[str],
        policy: typing.Literal["once", "once_per_container", "default"] = "default",
        stderr: typing.Literal["interleaved", "separate"] = "interleaved",
        workdir=None,
    ) -> subprocess.CompletedProcess:
        """
        Run a command inside a persistent container.
        """
        workdir = [] if workdir is None else [f"--container-workdir={workdir}"]
        policy_args = {
            "once": ["--ntasks=1"],
            "once_per_container": ["--ntasks-per-node=1"],
            "default": [],
        }[policy]
        command = (
            [
                "srun",
                f"--container-image={self._url}",
                f"--container-name={self._name}",
                "--container-remap-root",
                "--no-container-mount-home",
            ]
            + self._mount_args
            + policy_args
            + workdir
            + command
        )
        return run_and_log(command, logger=self._logger, stderr=stderr)

    def exists(self) -> bool:
        return self.exec(["true"]).returncode == 0
