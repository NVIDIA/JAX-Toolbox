import logging
import pathlib
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
        result = subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                # Otherwise bazel shutdown hangs.
                "--init",
                "--gpus=all",
                "--shm-size=1g",
            ]
            + self._mount_args
            + [
                self._url,
                "sleep",
                "infinity",
            ],
            check=True,
            encoding="utf-8",
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self._id = result.stdout.strip()
        return self

    def __exit__(self, *exc_info):
        subprocess.run(
            ["docker", "stop", self._id],
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

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
        result = subprocess.run(
            ["docker", "pull", self._url],
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            encoding="utf-8",
        )
        self._logger.debug(result.stdout)
        return result.returncode == 0
