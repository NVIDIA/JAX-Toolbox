import logging
import subprocess
import typing

from .container import Container
from .utils import run_and_log


class LocalContainer(Container):
    def __init__(self, *, logger: logging.Logger):
        super().__init__(logger=logger)

    def __enter__(self) -> Container:
        self._logger.debug("Running local mode inside current container")
        return self

    def __exit__(self, *exc_info) -> None:
        pass

    def __repr__(self) -> str:
        return "Local"

    def exec(
        self,
        command: typing.List[str],
        policy: typing.Literal["once"]
        | typing.Literal["once_per_container"]
        | typing.Literal["default"] = "default",
        stderr: typing.Literal["interleaved"]
        | typing.Literal["separate"] = "interleaved",
        workdir=None,
    ) -> subprocess.CompletedProcess:
        return run_and_log(command, logger=self._logger, stderr=stderr, cwd=workdir)

    def exists(self) -> bool:
        """The local environment always exists."""
        return True
