import logging
import pathlib
import shutil
import subprocess
import typing

from .container import Container
from .utils import run_and_log


class DockerContainer(Container):
    def __init__(
        self,
        url: str,
        *,
        logger: logging.Logger,
        mounts: typing.List[typing.Tuple[pathlib.Path, pathlib.Path]],
    ):
        super().__init__(logger=logger)
        self._mount_args = []
        for src, dst in mounts:
            self._mount_args += ["-v", f"{src}:{dst}"]
        self._url = url

    def __enter__(self):
        self._logger.debug(f"Launching {self}")
        have_gpus = shutil.which("nvidia-smi") is not None
        if not have_gpus:
            self._logger.warning("No GPUs detected!")
        gpu_args = ["--gpus=all"] if have_gpus else []
        result = run_and_log(
            [
                "docker",
                "run",
                "--detach",
                # Otherwise bazel shutdown hangs.
                "--init",
                "--shm-size=1g",
            ]
            + gpu_args
            + self._mount_args
            + [
                self._url,
                "sleep",
                "infinity",
            ],
            logger=self._logger,
            stderr="separate",
        )
        if result.returncode != 0:
            self._logger.error(
                f"Could not launch {self}, exit code {result.returncode}:"
            )
            self._logger.error("stdout:")
            self._logger.error(result.stdout)
            self._logger.error("stderr:")
            self._logger.error(result.stderr)
        result.check_returncode()
        self._id = result.stdout.strip()
        return self

    def __exit__(self, *exc_info):
        run_and_log(
            ["docker", "stop", self._id],
            logger=self._logger,
            stderr="interleaved",
        ).check_returncode()

    def __repr__(self):
        return f"Docker({self._url})"

    def exec(
        self,
        command: typing.List[str],
        *,
        policy: typing.Literal["once", "once_per_container", "default"] = "default",
        stderr: typing.Literal["interleaved", "separate"] = "interleaved",
        workdir: typing.Optional[str] = None,
        log_level: int = logging.DEBUG,
    ) -> subprocess.CompletedProcess:
        """
        Run a command inside a persistent container.
        """
        wd_arg = [] if workdir is None else ["--workdir", workdir]
        return run_and_log(
            ["docker", "exec"] + wd_arg + [self._id] + command,
            logger=self._logger,
            log_level=log_level,
            stderr=stderr,
        )

    def exists(self) -> bool:
        """
        Check if the given container exists.
        """
        result = run_and_log(
            ["docker", "pull", self._url],
            logger=self._logger,
            stderr="interleaved",
        )
        return result.returncode == 0
