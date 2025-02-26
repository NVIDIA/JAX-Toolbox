import logging
import pathlib
import secrets
import subprocess
import typing


class PyxisContainer:
    def __init__(
        self,
        url: str,
        *,
        logger: logging.Logger,
        mounts: typing.List[typing.Tuple[pathlib.Path, pathlib.Path]],
    ):
        self._logger = logger
        mount_str = ",".join(map(":".join, mounts))
        self._mount_args = [f"--container-mounts={mount_str}"] if mount_str else []
        self._name = secrets.token_urlsafe()
        self._url = url

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass

    def exec(
        self, command: typing.List[str], workdir=None
    ) -> subprocess.CompletedProcess:
        """
        Run a command inside a persistent container.
        """
        workdir = [] if workdir is None else [f"--container-workdir={workdir}"]
        command = (
            [
                "srun",
                f"--container-image={self._url}",
                f"--container-name={self._name}",
                "--container-remap-root",
            ]
            + self._mount_args
            + workdir
            + command
        )
        result = subprocess.run(
            command,
            encoding="utf-8",
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        return result

    def check_exec(
        self, cmd: typing.List[str], **kwargs
    ) -> subprocess.CompletedProcess:
        result = self.exec(cmd, **kwargs)
        if result.returncode != 0:
            self._logger.fatal(
                f"{' '.join(cmd)} exited with return code {result.returncode}"
            )
            self._logger.fatal(result.stdout)
            self._logger.fatal(result.stderr)
            result.check_returncode()
        return result
