import itertools
import logging
import pathlib
import shlex
import subprocess
import time
import typing

from .pyxis import PyxisContainer

# Monotonic counter so every SlurmJobContainer instance gets its own job directory,
# even when multiple containers share the same URL within a single process.
_instance_counter = itertools.count()


class SlurmJobContainer(PyxisContainer):
    """
    Pyxis/enroot container backend that submits work as sbatch jobs.

    Extends PyxisContainer, reusing its container-naming scheme, mount-argument
    construction, srun-command building, __enter__ bazelrc workaround, and
    __exit__ no-op.  The key behavioural difference is *how srun commands are
    dispatched*:

    - PyxisContainer    → calls srun directly (requires an existing salloc)
    - SlurmJobContainer → wraps srun in sbatch scripts submitted from the login
                          node; GPUs are held only for the duration of each job

    All commands — whether a single exec() call or a multi-stage exec_sequence()
    — are submitted as a single sbatch job via the same code path.  Each srun
    step within that job writes its output to per-rank log files named:

        <job_dir>/<instance>/job_N-<slurm_job_id>/stage-K-node-M-rank-R.log
    """

    def __init__(
        self,
        url: str,
        *,
        logger: logging.Logger,
        mounts: typing.List[typing.Tuple[pathlib.Path, pathlib.Path]],
        slurm_config: typing.Dict[str, typing.Any],
    ):
        # Delegate container naming, mount construction, and url storage to the
        # parent; __enter__ (bazelrc workaround) and __exit__ (no-op) are also
        # inherited unchanged.
        super().__init__(url, logger=logger, mounts=mounts)
        self._slurm_config = slurm_config
        self._job_counter = 0
        # Each instance gets its own subdirectory so concurrent container objects
        # for the same URL do not overwrite each other's job scripts and output files.
        instance_id = next(_instance_counter)
        base_job_dir = pathlib.Path(slurm_config["job_dir"])
        self._job_dir = base_job_dir / f"ctr{instance_id:04d}"
        self._job_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"SlurmJob({self._url})"

    # ------------------------------------------------------------------
    # Shared sbatch header builder
    # ------------------------------------------------------------------

    def _sbatch_headers(self, use_gpu: bool = True) -> typing.List[str]:
        """Return the #SBATCH directive lines common to all job scripts."""
        cfg = self._slurm_config
        lines: typing.List[str] = []
        if cfg.get("account"):
            lines.append(f"#SBATCH --account={cfg['account']}")
        if cfg.get("partition"):
            lines.append(f"#SBATCH --partition={cfg['partition']}")
        lines.append(f"#SBATCH --nodes={cfg['num_nodes']}")
        lines.append(f"#SBATCH --ntasks-per-node={cfg['ntasks_per_node']}")
        if cfg.get("time_limit"):
            lines.append(f"#SBATCH --time={cfg['time_limit']}")

        return lines

    def _generate_sequence_job_script(
        self,
        stages: typing.List[typing.Dict[str, typing.Any]],
        job_n: int,
        sbatch_log: pathlib.Path,
        use_gpu: bool = True,
    ) -> str:
        """
        Return a bash script that runs every stage in *stages* as a separate
        srun step, writing per-rank logs and per-stage exit codes.

        Log directory layout (created inside the job using $SLURM_JOB_ID):

            <job_dir>/<instance>/job_N-<slurm_job_id>/
                stage-0-node-0-rank-0.log
                stage-0-node-0-rank-1.log
                stage-1-node-0-rank-0.log
                ...

        Per-stage exit codes are written to fixed paths so the login node can
        read them without knowing the SLURM job ID in advance:

            <job_dir>/<instance>/job_N-stage-K.exit
        """
        lines = ["#!/bin/bash"]
        lines += self._sbatch_headers(use_gpu=use_gpu)
        lines.append(f"#SBATCH --output={sbatch_log}")
        lines.append(f"#SBATCH --error={sbatch_log}")
        lines.append("")

        # Log directory includes SLURM_JOB_ID (set by the runtime, not us)
        log_dir_expr = f"{self._job_dir}/job_{job_n}-${{SLURM_JOB_ID}}"
        lines.append(f'LOG_DIR="{log_dir_expr}"')
        lines.append('mkdir -p "$LOG_DIR"')
        lines.append("")

        for i, stage in enumerate(stages):
            policy = stage.get("policy", "default")
            workdir = stage.get("workdir")
            srun_cmd = self._build_srun_command(stage["command"], policy, workdir)

            # Insert --output/--error with shell-variable log paths BEFORE the
            # static srun flags.  We cannot use shlex.join for these because
            # the paths contain $LOG_DIR which must expand at runtime.
            log_prefix = f"$LOG_DIR/stage-{i}-node-%n-rank-%t.log"
            srun_line = (
                f'srun --output="{log_prefix}" --error="{log_prefix}" '
                + shlex.join(srun_cmd[1:])  # everything after "srun"
            )
            exit_path = self._job_dir / f"job_{job_n}-stage-{i}.exit"

            lines.append(f"# --- stage {i} ---")
            lines.append(srun_line)
            lines.append(f"echo $? > {exit_path}")
            if stage.get("stop_on_failure", False):
                # Short-circuit: remaining stages are not executed
                lines.append(f"[ $(cat {exit_path}) -ne 0 ] && exit 0")
            lines.append("")

        return "\n".join(lines) + "\n"

    def _submit_job(self, script_path: pathlib.Path) -> str:
        """Submit *script_path* via sbatch and return the assigned job ID."""
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

    def _wait_for_job(self, job_id: str) -> None:
        """
        Block until *job_id* leaves the SLURM queue.

        Uses squeue for live polling and sacct for the authoritative final
        state.  Cancels the job and raises TimeoutError if the configured
        timeout is exceeded.
        """
        poll_interval = self._slurm_config.get("poll_interval", 60)
        job_timeout = self._slurm_config.get("job_timeout", 14400)
        start = time.monotonic()

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
            elapsed = time.monotonic() - start
            if elapsed > job_timeout:
                subprocess.run(["scancel", job_id], capture_output=True, check=False)
                raise TimeoutError(
                    f"SLURM job {job_id} exceeded timeout of {job_timeout}s; cancelled"
                )

            squeue = subprocess.run(
                ["squeue", "--job", job_id, "--noheader", "--format=%T"],
                capture_output=True,
                check=False,
                text=True,
            )

            if not squeue.stdout.strip():
                break  # job has left the queue → finished

            state = squeue.stdout.strip()
            self._logger.debug(
                f"SLURM job {job_id}: state={state} elapsed={elapsed:.0f}s"
            )

            if state in terminal_states:
                break

            time.sleep(poll_interval)

        # Confirm final state via sacct
        sacct = subprocess.run(
            ["sacct", "--job", job_id, "--format=State", "--noheader", "--parsable2"],
            capture_output=True,
            check=False,
            text=True,
        )
        if sacct.stdout.strip():
            final_state = sacct.stdout.strip().split("\n")[0].strip()
            self._logger.debug(f"SLURM job {job_id}: final state={final_state}")

    def _read_stage_output(
        self,
        job_n: int,
        job_id: str,
        stage_idx: int,
        command: typing.List[str],
    ) -> typing.Optional[subprocess.CompletedProcess]:
        """
        Read the results of one stage from a sequence job.

        Returns None if the stage was never reached (exit code file absent).
        Per-rank log files are concatenated in sorted order to form stdout.
        """
        exit_path = self._job_dir / f"job_{job_n}-stage-{stage_idx}.exit"
        if not exit_path.exists():
            return None  # stage was skipped due to an earlier stop_on_failure

        returncode = int(exit_path.read_text().strip())

        # Collect per-rank log files written by srun inside the job
        log_dir = self._job_dir / f"job_{job_n}-{job_id}"
        log_files = sorted(log_dir.glob(f"stage-{stage_idx}-*.log"))
        stdout = "".join(f.read_text(errors="replace") for f in log_files)

        return subprocess.CompletedProcess(
            args=command,
            returncode=returncode,
            stdout=stdout,
            stderr="",
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
        """
        Submit *command* as a single-stage sbatch job and block until it finishes.

        Delegates to exec_sequence() so all submissions share one code path.
        The stderr parameter is accepted for interface compatibility but ignored:
        output is always written to per-rank log files.
        """
        results = self.exec_sequence(
            [
                {
                    "command": command,
                    "policy": policy,
                    "workdir": workdir,
                    "log_level": log_level,
                }
            ]
        )
        return results[0]

    def exec_sequence(
        self,
        stages: typing.List[typing.Dict[str, typing.Any]],
        *,
        use_gpu: bool = True,
    ) -> typing.List[typing.Optional[subprocess.CompletedProcess]]:
        """
        Submit all *stages* as a SINGLE sbatch job and return one result per stage.

        Because all stages run inside one job:

        - No node pinning is needed (SLURM guarantees one job = one node set)
        - GPUs are released as soon as the whole pipeline finishes
        - Each srun step writes per-rank logs to a directory named with the
          SLURM job ID:

              <job_dir>/<instance>/job_N-<slurm_job_id>/
                  stage-K-node-M-rank-R.log

        Stages not reached due to a prior stop_on_failure failure are
        represented as None in the returned list.
        """
        n = self._job_counter
        self._job_counter += 1

        script_path = self._job_dir / f"job_{n}.sh"
        sbatch_log = self._job_dir / f"job_{n}-sbatch.log"

        script_path.write_text(
            self._generate_sequence_job_script(stages, n, sbatch_log, use_gpu=use_gpu)
        )
        self._logger.debug(f"Wrote sequence job script to {script_path}")

        job_id = self._submit_job(script_path)
        self._wait_for_job(job_id)
        self._logger.info(f"SLURM sequence job {job_id} finished")

        results: typing.List[typing.Optional[subprocess.CompletedProcess]] = []
        for i, stage in enumerate(stages):
            result = self._read_stage_output(n, job_id, i, stage["command"])
            results.append(result)

            if result is None:
                # Stage was not reached; fill remaining with None
                results.extend([None] * (len(stages) - len(results)))
                break

            if stage.get("check", False) and result.returncode != 0:
                self._logger.fatal(
                    f"Stage {i} exited with return code {result.returncode}"
                )
                self._logger.fatal(result.stdout)
                result.check_returncode()

            if stage.get("stop_on_failure", False) and result.returncode != 0:
                results.extend([None] * (len(stages) - len(results)))
                break

        return results

    def exists(self) -> bool:
        """
        Return True if the container image can be pulled successfully.

        Submits a minimal CPU-only job (no GPU) that runs 'true' inside the
        container.  Used by the container-level search to skip dates for which
        no nightly build was published.
        """
        try:
            results = self.exec_sequence(
                [{"command": ["true"], "policy": "once"}],
                use_gpu=False,
            )
        except Exception as exc:
            self._logger.debug(f"exists() job failed: {exc}")
            return False

        result = results[0]
        return result is not None and result.returncode == 0
