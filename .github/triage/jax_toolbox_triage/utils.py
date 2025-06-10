from contextlib import contextmanager
import datetime
import logging
import pathlib
import shlex
import subprocess
import typing


def container_url(
    date: datetime.date, *, container: str, template: typing.Optional[str] = None
) -> str:
    """
    Construct the URL for --container on the given date.

    Arguments:
    date: YYYY-MM-DD format.
    """
    date_str = date.isoformat()
    if template is not None:
        return template.format(container=container, date=date_str)
    # Around 2024-02-09 the naming scheme changed.
    elif date > datetime.date(year=2024, month=2, day=9):
        return f"ghcr.io/nvidia/jax:{container}-{date_str}"
    else:
        return f"ghcr.io/nvidia/{container}:nightly-{date_str}"


def get_logger(output_prefix: pathlib.Path) -> logging.Logger:
    output_prefix.mkdir()
    logger = logging.getLogger("triage")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    trace_file = logging.FileHandler(filename=output_prefix / "info.log", mode="w")
    debug_file = logging.FileHandler(filename=output_prefix / "debug.log", mode="w")
    console.setLevel(logging.INFO)
    trace_file.setLevel(logging.INFO)
    debug_file.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    trace_file.setFormatter(formatter)
    debug_file.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(trace_file)
    logger.addHandler(debug_file)
    return logger


@contextmanager
def console_log_level(logger, level):
    # Temporarily change all handlers' levels to `level`
    old = []
    for handler in logger.handlers:
        old.append((handler.level, handler))
        handler.setLevel(level)
    try:
        yield None
    finally:
        for old_level, handler in old:
            handler.setLevel(old_level)


def prepare_bazel_cache_mounts(
    bazel_cache: str,
) -> typing.Sequence[typing.Tuple[pathlib.Path, pathlib.Path]]:
    if (
        bazel_cache.startswith("http://")
        or bazel_cache.startswith("https://")
        or bazel_cache.startswith("grpc://")
    ):
        # Remote cache, no mount needed
        return []
    elif (bazel_cache_path := pathlib.Path(bazel_cache)).is_absolute():
        bazel_cache_path.mkdir(exist_ok=True)
        return [(bazel_cache_path, bazel_cache_path)]
    else:
        raise Exception(
            "--bazel-cache should be an http/https/grpc URL or an absolute path"
        )


def run_and_log(
    command,
    logger: logging.Logger,
    stderr: typing.Literal["interleaved", "separate"],
    cwd: typing.Optional[str] = None,
) -> subprocess.CompletedProcess:
    logger.debug(f"Executing in {cwd or '.'}: {shlex.join(command)}")
    result = subprocess.Popen(
        command,
        encoding="utf-8",
        stderr=subprocess.STDOUT if stderr == "interleaved" else subprocess.PIPE,
        stdout=subprocess.PIPE,
        cwd=cwd,
    )
    assert result.stdout is not None
    stdouterr = ""
    for line in iter(result.stdout.readline, ""):
        stdouterr += line
        logger.debug(line.strip())
    result.wait()
    return subprocess.CompletedProcess(
        args=command,
        returncode=result.returncode,
        stdout=stdouterr,
        stderr=result.stderr.read() if result.stderr is not None else "",
    )
