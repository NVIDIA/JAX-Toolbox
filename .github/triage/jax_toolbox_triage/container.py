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

    def exec_sequence(
        self,
        stages: typing.List[typing.Dict[str, typing.Any]],
    ) -> typing.List[typing.Optional[subprocess.CompletedProcess]]:
        """
        Run a list of command stages sequentially, returning one result per stage.

        Each stage is a dict with:
          "command"         List[str]  — required
          "policy"          str        — forwarded to exec() (default "default")
          "stderr"          str        — forwarded to exec() (default "interleaved")
          "workdir"         str|None   — forwarded to exec()
          "log_level"       int        — forwarded to exec()
          "check"           bool       — raise CalledProcessError on non-zero exit
          "stop_on_failure" bool       — skip remaining stages on non-zero exit

        Stages that were not reached (because a prior stop_on_failure stage failed)
        are represented as None in the returned list.

        Subclasses may override this to submit all stages as a single batch job.
        """
        _EXEC_KEYS = {"policy", "stderr", "workdir", "log_level"}
        results: typing.List[typing.Optional[subprocess.CompletedProcess]] = []
        for stage in stages:
            kwargs = {k: v for k, v in stage.items() if k in _EXEC_KEYS}
            result = self.exec(stage["command"], **kwargs)
            results.append(result)
            if stage.get("check", False) and result.returncode != 0:
                self._logger.fatal(
                    f"{' '.join(stage['command'])} exited with return code "
                    f"{result.returncode}"
                )
                self._logger.fatal(result.stdout)
                result.check_returncode()
            if stage.get("stop_on_failure", False) and result.returncode != 0:
                results.extend([None] * (len(stages) - len(results)))
                break
        return results

    @abstractmethod
    def exists(self) -> bool:
        """
        Check if the container exists.
        """
        pass
