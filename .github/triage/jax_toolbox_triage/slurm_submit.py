import dataclasses
import hashlib
import itertools
import json
import logging
import pathlib
import shlex
import subprocess
import time
import typing

from .pyxis import PyxisContainer, build_srun_command

_runner_counter = itertools.count()


@dataclasses.dataclass(frozen=True)
class SlurmResources:
    account: typing.Optional[str]
    partition: typing.Optional[str]
    num_nodes: int
    ntasks_per_node: int
    gpus_per_node: typing.Optional[int]
    cpus_per_task: typing.Optional[int]
    mem: typing.Optional[str]
    time_limit: typing.Optional[str]

    def sbatch_headers(self) -> typing.List[str]:
        lines = []
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        lines.append(f"#SBATCH --nodes={self.num_nodes}")
        lines.append(f"#SBATCH --ntasks-per-node={self.ntasks_per_node}")
        if self.gpus_per_node is not None:
            lines.append(f"#SBATCH --gpus-per-node={self.gpus_per_node}")
        if self.cpus_per_task is not None:
            lines.append(f"#SBATCH --cpus-per-task={self.cpus_per_task}")
        if self.mem is not None:
            lines.append(f"#SBATCH --mem={self.mem}")
        if self.time_limit:
            lines.append(f"#SBATCH --time={self.time_limit}")
        return lines


@dataclasses.dataclass(frozen=True)
class SlurmConfig:
    job_dir: pathlib.Path
    image_dir: pathlib.Path
    poll_interval: int
    utility_resources: SlurmResources
    test_resources: SlurmResources
    build_resources: SlurmResources


@dataclasses.dataclass(frozen=True)
class SlurmJobResult:
    job_id: str
    state: str
    node_list: str
    returncode: int
    stdout: str
    stderr: str
    output_path: pathlib.Path
    error_path: pathlib.Path
    exit_path: pathlib.Path
    script_path: pathlib.Path


def make_slurm_config(args) -> SlurmConfig:
    return SlurmConfig(
        job_dir=args.slurm_job_dir,
        image_dir=args.slurm_image_dir,
        poll_interval=args.slurm_poll_interval,
        utility_resources=SlurmResources(
            account=args.slurm_build_account,
            partition=args.slurm_build_partition,
            num_nodes=args.slurm_build_num_nodes,
            ntasks_per_node=args.slurm_build_ntasks_per_node,
            gpus_per_node=args.slurm_build_gpus_per_node,
            cpus_per_task=args.slurm_build_cpus_per_task,
            mem=args.slurm_build_mem,
            time_limit=args.slurm_build_time_limit,
        ),
        test_resources=SlurmResources(
            account=args.slurm_account,
            partition=args.slurm_partition,
            num_nodes=args.slurm_num_nodes,
            ntasks_per_node=args.slurm_ntasks_per_node,
            gpus_per_node=args.slurm_gpus_per_node,
            cpus_per_task=args.slurm_cpus_per_task,
            mem=args.slurm_mem,
            time_limit=args.slurm_time_limit,
        ),
        build_resources=SlurmResources(
            account=args.slurm_build_account,
            partition=args.slurm_build_partition,
            num_nodes=args.slurm_build_num_nodes,
            ntasks_per_node=args.slurm_build_ntasks_per_node,
            gpus_per_node=args.slurm_build_gpus_per_node,
            cpus_per_task=args.slurm_build_cpus_per_task,
            mem=args.slurm_build_mem,
            time_limit=args.slurm_build_time_limit,
        ),
    )


class SlurmJobRunner:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        config: SlurmConfig,
        base_dir: typing.Optional[pathlib.Path] = None,
    ):
        self._logger = logger
        self._config = config
        if base_dir is None:
            instance_id = next(_runner_counter)
            base_dir = config.job_dir / f"runner{instance_id:04d}"
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._job_counter = 0

    @property
    def base_dir(self) -> pathlib.Path:
        return self._base_dir

    def run(
        self,
        *,
        job_name: str,
        resources: SlurmResources,
        body: str,
        stderr: typing.Literal["interleaved", "separate"] = "interleaved",
    ) -> SlurmJobResult:
        job_n = self._job_counter
        self._job_counter += 1
        script_path = self._base_dir / f"job_{job_n}.sh"
        output_path = self._base_dir / f"job_{job_n}.out"
        error_path = (
            output_path
            if stderr == "interleaved"
            else self._base_dir / f"job_{job_n}.err"
        )
        exit_path = self._base_dir / f"job_{job_n}.exit"
        script_path.write_text(
            self._render_script(
                job_name=job_name,
                resources=resources,
                body=body,
                output_path=output_path,
                error_path=error_path,
                exit_path=exit_path,
            )
        )
        self._logger.debug(f"Wrote SLURM script {script_path}")
        job_id = self._submit(script_path)
        state, node_list = self._wait_for_job(job_id)
        stdout = output_path.read_text(errors="replace") if output_path.exists() else ""
        stderr_text = (
            stdout
            if error_path == output_path
            else error_path.read_text(errors="replace")
            if error_path.exists()
            else ""
        )
        returncode = int(exit_path.read_text().strip()) if exit_path.exists() else 1
        return SlurmJobResult(
            job_id=job_id,
            state=state,
            node_list=node_list,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr_text,
            output_path=output_path,
            error_path=error_path,
            exit_path=exit_path,
            script_path=script_path,
        )

    def _render_script(
        self,
        *,
        job_name: str,
        resources: SlurmResources,
        body: str,
        output_path: pathlib.Path,
        error_path: pathlib.Path,
        exit_path: pathlib.Path,
    ) -> str:
        lines = ["#!/bin/bash", f"#SBATCH --job-name={job_name}"]
        lines += resources.sbatch_headers()
        lines.append(f"#SBATCH --output={output_path}")
        lines.append(f"#SBATCH --error={error_path}")
        lines += [
            "",
            "set -uo pipefail",
            f'EXIT_PATH="{exit_path}"',
            "rc=0",
            "(",
            "  set -e",
        ]
        lines += [f"  {line}" if line else "" for line in body.splitlines()]
        lines += [
            ") || rc=$?",
            'echo "${rc}" > "${EXIT_PATH}"',
            'exit "${rc}"',
            "",
        ]
        return "\n".join(lines)

    def _submit(self, script_path: pathlib.Path) -> str:
        sbatch_cmd = ["sbatch", "--parsable", str(script_path)]
        self._logger.debug(f"Submitting: {shlex.join(sbatch_cmd)}")
        result = subprocess.run(sbatch_cmd, capture_output=True, check=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"sbatch submission failed (exit {result.returncode}):\n"
                f"{result.stderr.strip()}"
            )
        job_id = result.stdout.strip()
        self._logger.info(f"Submitted SLURM job {job_id} ({script_path.name})")
        return job_id

    def _wait_for_job(self, job_id: str) -> typing.Tuple[str, str]:
        terminal_states = {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "PREEMPTED",
        }
        while True:
            squeue = subprocess.run(
                ["squeue", "--job", job_id, "--noheader", "--format=%T|%N"],
                capture_output=True,
                check=False,
                text=True,
            )
            status = squeue.stdout.strip()
            if not status:
                break
            state, _, node_list = status.partition("|")
            self._logger.debug(
                f"SLURM job {job_id}: state={state} nodes={node_list or '<unknown>'}"
            )
            if state in terminal_states:
                break
            time.sleep(self._config.poll_interval)

        sacct = subprocess.run(
            [
                "sacct",
                "--job",
                job_id,
                "--format=State,NodeList",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        for line in sacct.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            state, _, node_list = line.partition("|")
            return state.strip(), node_list.strip()
        return "UNKNOWN", ""


class SlurmContainer(PyxisContainer):
    def __init__(
        self,
        url: str,
        *,
        logger: logging.Logger,
        mounts: typing.List[typing.Tuple[pathlib.Path, pathlib.Path]],
        slurm_config: SlurmConfig,
        resource_kind: typing.Literal["utility", "test"] = "test",
    ):
        super().__init__(url, logger=logger, mounts=mounts)
        self._slurm_config = slurm_config
        self._resource_kind = resource_kind
        runner_dir = slurm_config.job_dir / f"ctr-{self._name[:12]}"
        self._runner = SlurmJobRunner(
            logger=logger,
            config=slurm_config,
            base_dir=runner_dir,
        )
        self._state_image = runner_dir / "working.sqsh"

    def __repr__(self) -> str:
        return f"Slurm({self._url})"

    def __enter__(self):
        self._logger.debug(f"Launching {self}")
        return self

    def _current_image(self) -> str:
        return self._state_image.as_posix() if self._state_image.exists() else self._url

    def _resources(self) -> SlurmResources:
        return (
            self._slurm_config.utility_resources
            if self._resource_kind == "utility"
            else self._slurm_config.test_resources
        )

    def exec(
        self,
        command: typing.List[str],
        *,
        policy: typing.Literal["once", "once_per_container", "default"] = "default",
        stderr: typing.Literal["interleaved", "separate"] = "interleaved",
        workdir: typing.Optional[str] = None,
        log_level: int = logging.DEBUG,
    ) -> subprocess.CompletedProcess:
        persist_state = policy == "once_per_container"
        tmp_state_path = self._state_image.with_name(
            f"{self._state_image.stem}-{self._runner._job_counter}.tmp"
        )
        srun_cmd = build_srun_command(
            container_image=self._current_image(),
            container_name=self._name,
            mount_args=self._mount_args,
            command=command,
            policy=policy,
            workdir=workdir,
            container_writable=persist_state,
            container_save=tmp_state_path if persist_state else None,
        )
        body_lines = [shlex.join(srun_cmd)]
        if persist_state:
            body_lines += [
                f"rm -rf {shlex.quote(self._state_image.as_posix())}",
                f"mv {shlex.quote(tmp_state_path.as_posix())} {shlex.quote(self._state_image.as_posix())}",
            ]
        result = self._runner.run(
            job_name=f"triage-exec-{self._name[:8]}",
            resources=self._resources(),
            body="\n".join(body_lines),
            stderr=stderr,
        )
        for line in result.stdout.splitlines():
            self._logger.log(log_level, line)
        if stderr == "separate":
            for line in result.stderr.splitlines():
                self._logger.log(log_level, line)
        if persist_state and result.returncode != 0 and tmp_state_path.exists():
            tmp_state_path.unlink()
        return subprocess.CompletedProcess(
            args=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def exists(self) -> bool:
        return self.exec(["true"], policy="once").returncode == 0


def slurm_candidate_key(
    *,
    base_image: str,
    versions: typing.Dict[str, str],
    cherry_picks: typing.Dict[str, typing.List[str]],
    exclude_transformer_engine: bool,
    build_scripts_path: typing.Optional[str],
) -> str:
    payload = {
        "base_image": base_image,
        "versions": versions,
        "cherry_picks": cherry_picks,
        "exclude_transformer_engine": exclude_transformer_engine,
        "build_scripts_path": build_scripts_path,
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
