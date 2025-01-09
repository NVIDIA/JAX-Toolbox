import datetime
import logging
import pathlib
import subprocess
import typing


def container_url(date: datetime.date, *, container: str) -> str:
    """
    Construct the URL for --container on the given date.

    Arguments:
    date: YYYY-MM-DD format.
    """
    # Around 2024-02-09 the naming scheme changed.
    if date > datetime.date(year=2024, month=2, day=9):
        return f"ghcr.io/nvidia/jax:{container}-{date.isoformat()}"
    else:
        return f"ghcr.io/nvidia/{container}:nightly-{date.isoformat()}"


def container_exists(
    date: datetime.date, *, container: str, logger: logging.Logger
) -> bool:
    """
    Check if the given container exists.
    """
    result = subprocess.run(
        ["docker", "pull", container_url(date, container=container)],
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        encoding="utf-8",
    )
    logger.debug(result.stdout)
    return result.returncode == 0


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
