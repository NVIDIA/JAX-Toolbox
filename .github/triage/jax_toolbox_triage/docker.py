import logging
import pathlib
import subprocess
import typing


class DockerContainer:
    def __init__(
        self,
        url: str,
        *,
        logger: logging.Logger,
        mounts: typing.List[typing.Tuple[pathlib.Path, pathlib.Path]],
    ):
        self._logger = logger
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
        policy: typing.Literal["once", "once_per_container", "default"] = "default",
        workdir=None,
    ) -> subprocess.CompletedProcess:
        """
        Run a command inside a persistent container.
        """
        workdir = [] if workdir is None else ["--workdir", workdir]
        return subprocess.run(
            ["docker", "exec"] + workdir + [self._id] + command,
            encoding="utf-8",
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

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
