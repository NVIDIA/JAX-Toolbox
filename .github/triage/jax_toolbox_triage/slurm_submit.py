import hashlib
import itertools
import logging
import pathlib
import secrets
import shlex
import subprocess
import time
import typing

from .container import Container

# Makes container names process-unique (same intent as in pyxis.py)
_process_token = secrets.token_bytes()

# Monotonic counter so every SlurmJobContainer instance gets its own job directory,
# even when multiple containers share the same URL within a single process.
_instance_counter = itertools.count()


class SlurmJobContainer(Container):
    """
    Container backend that submits each exec() call as an sbatch job.

    Intended for use on login nodes where srun/salloc would be inappropriate.
    GPUs are requested only for the duration of each individual job, so no
    resources are held between bisection steps.

    Container state (git checkouts, build artefacts) persists across jobs because
    enroot stores named containers on the node's local filesystem between allocations.
    Subsequent exec() calls within the same context manager are pinned to the first
    node that was assigned by SLURM, ensuring state continuity.
    """

    def __init__(
        self,
        url: str,
        *,
        logger: logging.Logger,
        mounts: typing.List[typing.Tuple[pathlib.Path, pathlib.Path]],
        slurm_config: typing.Dict[str, typing.Any],
    ):
        super().__init__(logger=logger)
        mount_str = ",".join(f"{src}:{dst}" for src, dst in mounts)
        self._mount_args = [f"--container-mounts={mount_str}"] if mount_str else []
        self._name = hashlib.sha256(url.encode() + _process_token).hexdigest()
        self._url = url
        self._slurm_config = slurm_config
        # Node to which subsequent jobs in this session are pinned (set after first job)
        self._node: typing.Optional[str] = None
        self._job_counter = 0
        # Each instance gets its own subdirectory so concurrent container objects
        # for the same URL do not overwrite each other's job scripts and output files.
        instance_id = next(_instance_counter)
        base_job_dir = pathlib.Path(slurm_config["job_dir"])
        self._job_dir = base_job_dir / f"ctr{instance_id:04d}"
        self._job_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> "SlurmJobContainer":
        self._logger.debug(f"Launching {self}")
        # Workaround for pyxis backend with some bazel versions
        # https://github.com/bazelbuild/bazel/issues/22955#issuecomment-2293899428
        self.check_exec(
            [
                "sh",
                "-c",
                "echo 'startup --host_jvm_args=-XX:-UseContainerSupport' > ${JAX_TOOLBOX_TRIAGE_PREFIX}/root/.bazelrc",
            ]
        )
        return self

    def __exit__(self, *exc_info) -> None:
        # Container files remain on the pinned node; enroot cleans them up on its own
        # schedule.  Nothing to do here.
        pass

    def __repr__(self) -> str:
        return f"SlurmJob({self._url})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_job_script(
        self,
        command: typing.List[str],
        policy: typing.Literal["once", "once_per_container", "default"],
        workdir: typing.Optional[str],
        stdout_path: pathlib.Path,
        stderr_path: pathlib.Path,
        exit_code_path: pathlib.Path,
        use_gpu: bool = True,
    ) -> str:
        """Return a bash script suitable for submission via sbatch."""
        policy_args: typing.List[str] = {
            "once": ["--ntasks=1"],
            "once_per_container": ["--ntasks-per-node=1"],
            "default": [],
        }[policy]
        wd_arg = [] if workdir is None else [f"--container-workdir={workdir}"]

        srun_cmd = (
            [
                "srun",
                f"--container-image={self._url}",
                f"--container-name={self._name}",
                "--container-remap-root",
                "--no-container-mount-home",
            ]
            + self._mount_args
            + policy_args
            + wd_arg
            + command
        )

        cfg = self._slurm_config
        lines = ["#!/bin/bash"]
        if cfg.get("account"):
            lines.append(f"#SBATCH --account={cfg['account']}")
        if cfg.get("partition"):
            lines.append(f"#SBATCH --partition={cfg['partition']}")
        if use_gpu and cfg.get("num_gpus", 0) > 0:
            lines.append(f"#SBATCH --gres=gpu:{cfg['num_gpus']}")
        lines.append("#SBATCH --nodes=1")
        lines.append(f"#SBATCH --output={stdout_path}")
        lines.append(f"#SBATCH --error={stderr_path}")
        if cfg.get("time_limit"):
            lines.append(f"#SBATCH --time={cfg['time_limit']}")
        for flag in cfg.get("extra_flags", []):
            lines.append(f"#SBATCH {flag}")

        lines.append("")
        lines.append(shlex.join(srun_cmd))
        # Write the exit code to a dedicated file so the login node can read it
        # after the job finishes.  We do this explicitly rather than relying on
        # sacct's ExitCode field, which reflects signals/OOM kills rather than
        # the application's own exit status.
        lines.append(f"echo $? > {exit_code_path}")

        return "\n".join(lines) + "\n"

    def _submit_job(self, script_path: pathlib.Path) -> str:
        """Submit *script_path* via sbatch and return the assigned job ID."""
        sbatch_cmd = ["sbatch", "--parsable"]
        if self._node:
            sbatch_cmd.append(f"--nodelist={self._node}")
        sbatch_cmd.append(str(script_path))

        self._logger.debug(f"Submitting: {shlex.join(sbatch_cmd)}")
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"sbatch submission failed (exit {result.returncode}):\n"
                f"{result.stderr.strip()}"
            )
        job_id = result.stdout.strip()
        self._logger.info(f"Submitted SLURM job {job_id} ({script_path.name})")
        return job_id

    def _wait_for_job(self, job_id: str) -> typing.Optional[str]:
        """
        Block until *job_id* leaves the SLURM queue, then return the node it ran on.

        Uses squeue for live polling and sacct for the authoritative final state.
        Cancels the job and raises TimeoutError if the configured timeout is exceeded.
        """
        poll_interval = self._slurm_config.get("poll_interval", 30)
        job_timeout = self._slurm_config.get("job_timeout", 14400)
        start = time.monotonic()
        node: typing.Optional[str] = None

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
                subprocess.run(["scancel", job_id], capture_output=True)
                raise TimeoutError(
                    f"SLURM job {job_id} exceeded timeout of {job_timeout}s; cancelled"
                )

            squeue = subprocess.run(
                ["squeue", "--job", job_id, "--noheader", "--format=%T|%N"],
                capture_output=True,
                text=True,
            )

            if not squeue.stdout.strip():
                # Job no longer tracked by squeue → it has finished
                break

            parts = squeue.stdout.strip().split("|", 1)
            state = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                node = parts[1].strip()

            self._logger.debug(
                f"SLURM job {job_id}: state={state} node={node} "
                f"elapsed={elapsed:.0f}s"
            )

            if state in terminal_states:
                # squeue can briefly show terminal states; stop polling immediately
                break

            time.sleep(poll_interval)

        # sacct is the authoritative source once the job has left the queue
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
            text=True,
        )
        if sacct.stdout.strip():
            first_line = sacct.stdout.strip().split("\n")[0]
            parts = first_line.split("|", 1)
            final_state = parts[0].strip()
            sacct_node = parts[1].strip() if len(parts) > 1 else ""
            if sacct_node:
                node = sacct_node
            self._logger.debug(
                f"SLURM job {job_id}: final state={final_state} node={node}"
            )

        return node or None

    def _read_job_output(
        self,
        job_n: int,
        stdout_path: pathlib.Path,
        stderr_path: pathlib.Path,
        exit_code_path: pathlib.Path,
        stderr_mode: str,
    ) -> subprocess.CompletedProcess:
        """Read job output files and return a CompletedProcess-like object."""
        stdout = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""
        stderr = ""
        if stderr_mode == "separate" and stderr_path.exists():
            stderr = stderr_path.read_text(errors="replace")

        if exit_code_path.exists():
            returncode = int(exit_code_path.read_text().strip())
        else:
            self._logger.warning(
                f"Exit code file missing for job_{job_n} — "
                "job was likely killed before the command completed "
                "(OOM, preemption, or node failure)"
            )
            returncode = -1

        return subprocess.CompletedProcess(
            args=[],
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    # ------------------------------------------------------------------
    # Container ABC implementation
    # ------------------------------------------------------------------

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
        Submit *command* as an sbatch job, block until it finishes, and return
        the result.

        On the first call within a context-manager session, records the node
        assigned by SLURM and pins all subsequent calls to that same node so
        that enroot's named-container state is preserved between jobs.
        """
        n = self._job_counter
        self._job_counter += 1

        script_path = self._job_dir / f"job_{n}.sh"
        stdout_path = self._job_dir / f"job_{n}.out"
        stderr_path = self._job_dir / f"job_{n}.err"
        exit_code_path = self._job_dir / f"job_{n}.exit"

        # In interleaved mode both stdout and stderr go to the same file
        sbatch_stderr_path = stdout_path if stderr == "interleaved" else stderr_path

        script = self._generate_job_script(
            command,
            policy,
            workdir,
            stdout_path,
            sbatch_stderr_path,
            exit_code_path,
        )
        script_path.write_text(script)
        self._logger.debug(f"Wrote job script to {script_path}")

        job_id = self._submit_job(script_path)
        node = self._wait_for_job(job_id)

        if self._node is None and node:
            self._node = node
            self._logger.debug(f"Pinned container session to node {self._node}")

        result = self._read_job_output(
            n, stdout_path, stderr_path, exit_code_path, stderr
        )
        self._logger.log(
            log_level,
            f"SLURM job {job_id} finished: returncode={result.returncode}",
        )
        # Stream the captured output through the logger so it appears in debug.log
        for line in result.stdout.splitlines():
            self._logger.log(log_level, line)

        return result

    def exists(self) -> bool:
        """
        Return True if the container image can be pulled successfully.

        Submits a minimal job (no GPU requested) that runs 'true' inside the
        container.  This is used by the container-level search to skip dates
        for which no nightly build was published.
        """
        n = self._job_counter
        self._job_counter += 1

        script_path = self._job_dir / f"job_{n}.sh"
        stdout_path = self._job_dir / f"job_{n}.out"
        exit_code_path = self._job_dir / f"job_{n}.exit"

        script = self._generate_job_script(
            ["true"],
            "once",
            None,
            stdout_path,
            stdout_path,
            exit_code_path,
            use_gpu=False,
        )
        script_path.write_text(script)

        try:
            job_id = self._submit_job(script_path)
            self._wait_for_job(job_id)
        except Exception as exc:
            self._logger.debug(f"exists() job failed: {exc}")
            return False

        return exit_code_path.exists() and exit_code_path.read_text().strip() == "0"
