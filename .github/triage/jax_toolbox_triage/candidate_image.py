import dataclasses
import json
import logging
import pathlib
import shlex
import time
import typing

from .pyxis import build_srun_command
from .slurm_submit import (
    SlurmConfig,
    SlurmJobRunner,
    slurm_candidate_key,
)


@dataclasses.dataclass(frozen=True)
class CandidateImage:
    key: str
    directory: pathlib.Path
    image_path: pathlib.Path
    metadata_path: pathlib.Path


@dataclasses.dataclass(frozen=True)
class CandidateImageBuildResult:
    candidate: CandidateImage
    reused_existing: bool
    build_time: float
    stdout: str
    stderr: str


def candidate_image_from_versions(
    *,
    config: SlurmConfig,
    base_image: str,
    versions: typing.Dict[str, str],
    cherry_picks: typing.Dict[str, typing.List[str]],
    exclude_transformer_engine: bool,
    build_scripts_path: typing.Optional[str],
) -> CandidateImage:
    key = slurm_candidate_key(
        base_image=base_image,
        versions=versions,
        cherry_picks=cherry_picks,
        exclude_transformer_engine=exclude_transformer_engine,
        build_scripts_path=build_scripts_path,
    )
    directory = config.image_dir / key
    return CandidateImage(
        key=key,
        directory=directory,
        image_path=directory / "candidate.sqsh",
        metadata_path=directory / "metadata.json",
    )


def materialize_candidate_image(
    *,
    logger: logging.Logger,
    config: SlurmConfig,
    runner: SlurmJobRunner,
    candidate: CandidateImage,
    base_image: str,
    container_name: str,
    mount_args: typing.List[str],
    build_script_lines: typing.List[str],
    metadata: typing.Dict[str, typing.Any],
) -> CandidateImageBuildResult:
    candidate.directory.mkdir(parents=True, exist_ok=True)
    if candidate.image_path.exists():
        stdout_path = runner.base_dir / "job_0.out"
        stdout = ""
        if stdout_path.exists():
            stdout = stdout_path.read_text(errors="replace")
        logger.info(
            f"Reusing cached candidate image {candidate.image_path} ({candidate.key})"
        )
        return CandidateImageBuildResult(
            candidate=candidate,
            reused_existing=True,
            build_time=0.0,
            stdout=stdout,
            stderr="",
        )

    start = time.monotonic()
    tmp_image = candidate.directory / f"candidate.{candidate.key}.tmp.sqsh"
    build_cmd = build_srun_command(
        container_image=base_image,
        container_name=container_name,
        mount_args=mount_args,
        command=["bash", "-lc", "\n".join(build_script_lines)],
        policy="once_per_container",
        container_writable=True,
        container_save=tmp_image,
    )
    result = runner.run(
        job_name=f"triage-build-{candidate.key[:8]}",
        resources=config.build_resources,
        stderr="separate",
        body="\n".join(
            [
                f"mkdir -p {shlex.quote(candidate.directory.as_posix())}",
                shlex.join(build_cmd),
                f"rm -rf {shlex.quote(candidate.image_path.as_posix())}",
                f"mv {shlex.quote(tmp_image.as_posix())} {shlex.quote(candidate.image_path.as_posix())}",
            ]
        ),
    )
    build_time = time.monotonic() - start
    if result.returncode == 0:
        candidate.metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True)
        )
    return CandidateImageBuildResult(
        candidate=candidate,
        reused_existing=False,
        build_time=build_time,
        stdout=result.stdout,
        stderr=result.stderr,
    )
