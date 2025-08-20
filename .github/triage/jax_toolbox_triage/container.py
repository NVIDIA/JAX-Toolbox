from abc import ABC, abstractmethod
import logging
import subprocess
import typing


class Container(ABC):
    def __init__(self, *, logger: logging.Logger):
        self._logger = logger

    @abstractmethod
    def __enter__(self) -> "Container":
        """
        Launch the container instance
        """
        pass

    @abstractmethod
    def __exit__(self, *exc_info) -> None:
        """
        Shut down the container instance
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
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
        pass

    def check_exec(
        self,
        cmd: typing.List[str],
        *,
        log_level: int = logging.DEBUG,
        policy: typing.Literal["once", "once_per_container", "default"] = "default",
        stderr: typing.Literal["interleaved", "separate"] = "interleaved",
        workdir: typing.Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        result = self.exec(
            cmd, log_level=log_level, policy=policy, stderr=stderr, workdir=workdir
        )
        if result.returncode != 0:
            self._logger.fatal(
                f"{' '.join(cmd)} exited with return code {result.returncode}"
            )
            if stderr == "separate":
                self._logger.fatal("stderr:")
                self._logger.fatal(result.stderr)
                self._logger.fatal("stdout:")
            self._logger.fatal(result.stdout)
            result.check_returncode()
        return result

    @abstractmethod
    def exists(self) -> bool:
        """
        Check if the container exists.
        """
        pass
