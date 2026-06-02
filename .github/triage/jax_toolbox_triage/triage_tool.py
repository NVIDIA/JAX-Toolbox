import collections
import contextlib
import datetime
import functools
import hashlib
import json
import logging
import pathlib
import platform
import time
from typing import Dict, Tuple, Union, Any, Optional, Set

from .container import Container
from .logic import (
    _EXIT_CODE_METRIC,
    _REPETITION_KEY,
    _WORKLOAD_VERSION_KEY,
    ClassifiedTestOutcome,
    container_search,
    ExitCodeClassifier,
    TestExecutionOutcome,
    TestResult,
    version_search,
    CouldNotReproduceFailure,
    CouldNotReproduceSuccess,
)
from .metric_classifier import MetricClassifier
from .versions import get_versions_dirs_env
from .summary import (
    add_summary_record,
    CONTAINER_CACHE_SECTION,
    create_output_symlinks,
    load_summary,
    result_cache_from_summary,
    VERSION_CACHE_SECTION,
)
from .bisect import get_commit_history
from .docker import DockerContainer
from .utils import (
    container_url as container_url_base,
    prepare_bazel_cache_mounts,
    run_and_log,
)
from .container_factory import make_container


class InconsistentResults(Exception):
    pass


class TriageTool:
    """
    This is the main class that orchestrates the whole triage process.
    """

    def __init__(self, args, logger):
        self.args = args
        # prefix test_command with jax cache option
        self.args.test_command = [
            "env",
            "JAX_ENABLE_COMPILATION_CACHE=false",
        ] + self.args.test_command
        self.logger = logger
        self.bisection_url = None
        self.bisection_versions = {}
        self.package_dirs = None
        self.dynamic_packages = set()
        self.packages_with_scripts = set()
        self.bazel_cache_mounts = prepare_bazel_cache_mounts(self.args.bazel_cache)
        self.check_success_before_failure = True
        self.restart_cache = (
            result_cache_from_summary(
                self.args.output_prefix, summary=load_summary(self.args.output_prefix)
            )
            if args.restart
            else {}
        )

        self.logger.info("Arguments:")
        for k, v in vars(self.args).items():
            logger.info(f"  {k}: {v}")
        logger.info(
            "Verbose output, including stdout/err of triage commands, will be written "
            f"to {(self.args.output_prefix / 'debug.log').resolve()}"
        )

    def _version_slug(self, url: str, versions: Dict[str, str]) -> str:
        hash_chars = 8
        components = {"container": hashlib.sha1(url.encode()).hexdigest()}
        components.update(versions)
        return "-".join(f"{k}-{v[:hash_chars]}" for k, v in components.items())

    def _test_output_directory(
        self, url: str, versions: Union[Dict[str, str], None]
    ) -> pathlib.Path:
        """
        Create a directory for test output based on the container URL and versions.

        Args:
            url (str): The URL of the container.
            versions (dict): A dictionary of software versions.
        Returns:
            pathlib.Path: The path to the output directory.
        """
        out_dirname = self._version_slug(url=url, versions=versions or {})
        out_dir = self.args.output_prefix / out_dirname
        if out_dir.exists() and self.args.restart:
            base_out_dir = out_dir
            n = 1
            while out_dir.exists():
                out_dir = base_out_dir.with_name(f"{base_out_dir.name}-restart-{n}")
                n += 1
        assert not out_dir.exists(), (
            f"{out_dir} should not already exist, maybe you are re-using {self.args.output_prefix}?"
        )
        out_dir.mkdir(mode=0o755)
        return out_dir.resolve()

    def _make_container(
        self, url: str, test_output_directory: Optional[pathlib.Path] = None
    ) -> Container:
        """
        Wrapper for make_container factory

        Args:
            url: (str), the input url of the docker image
            test_output_directory: (pathlib.Path), the path to the output directory

        Returns:
            Container object
        """
        mounts = self.bazel_cache_mounts + self.args.container_mount
        if test_output_directory is not None:
            mounts.append((test_output_directory, "/triage-tool-output"))
        runtime = self.args.container_runtime
        if runtime == "plugin":
            runtime = "docker"
        return make_container(runtime, url, mounts, self.logger)

    def _get_versions(
        self,
        container_url: str,
        explicit_versions: Dict[str, str],
        versions_from_env: bool,
    ):
        """
        Get the versions of the software packages in the container.

        Args:
            container_url (str): The URL of the container.
            explicit_versions (str): Explicit versions to use.
            versions_from_env (bool): Whether to get versions from environment variables.
        Returns:
            overriden_versions (dict): The versions with explicit overrides.
            url_versions (dict): The versions from the container URL.
            dirs (dict): The directories of the software packages.
            env (dict): The environment variables in the container.
        """
        if explicit_versions is not None and container_url is None:
            return explicit_versions, None, None, None
        assert container_url is not None, (
            "Container URL must be provided if explicit versions are not set."
        )

        with self._make_container(container_url) as worker:
            url_versions, dirs, env = get_versions_dirs_env(
                worker=worker,
                versions_from_env=versions_from_env,
                optional_software=self.args.optional_software,
            )
        overriden_versions = url_versions.copy()
        if explicit_versions is not None:
            overriden_versions.update(explicit_versions)

        return overriden_versions, url_versions, dirs, env

    def _gather_histories(
        self,
        worker: Container,
        passing_versions: Dict[str, str],
        failing_versions: Dict[str, str],
    ) -> collections.OrderedDict:
        """
        Gather the commit histories for the passing and failing versions.

        Args:
            worker (Container): The container in which to run the commands.
            passing_versions (dict): The versions that passed.
            failing_versions (dict): The versions that failed.
        Returns:
            collections.OrderDict: The commit histories for passing and failing versions.
        """
        packages = passing_versions.keys()
        package_versions = collections.OrderedDict()

        for package in packages:
            if package not in self.package_dirs:
                continue
            history, cherry_pick_ranges = get_commit_history(
                worker,
                package,
                passing_versions[package],
                failing_versions[package],
                self.package_dirs[package],
                main_branch=self.args.main_branch,
                logger=self.logger,
                args=self.args,
            )
            package_versions[package] = history
            if package in self.args.cherry_pick:
                # If explicit commits to cherry-pick were given on the commandline,
                # make sure they are known to the local working copy. They might not be
                # if the fix being cherry-picked is newer, or only lives in a remote
                # that is being passed in via --override-remotes.
                worker.check_exec(
                    ["git", "fetch", self.args.override_remotes.get(package, "origin")]
                    + self.args.cherry_pick[package],
                    policy="once_per_container",
                    workdir=self.package_dirs[package],
                )
            for cherry_pick_range in cherry_pick_ranges:
                if package not in self.args.cherry_pick:
                    self.args.cherry_pick[package] = []
                self.args.cherry_pick[package].append(cherry_pick_range)

            assert all(
                b[1] >= a[1]
                for a, b in zip(
                    package_versions[package], package_versions[package][1:]
                )
            )

        for package in packages:
            if package in package_versions:
                continue
            package_versions[package] = [
                (passing_versions[package], package_versions["xla"][0][1])
            ]
            if passing_versions[package] != failing_versions[package]:
                package_versions[package].append(
                    (failing_versions[package], package_versions["xla"][-1][1])
                )

        return package_versions

    def _log_environment_differences(
        self, url1: str, url2: str, env1: Dict[str, str], env2: Dict[str, str]
    ):
        """
         If we have two containers, print the differences between their environments. This
        can be useful in the case that rebuilding the good versions in the bad container,
        or vice versa, does not reproduce the expected result.

        Args:
            url1 (str): The URL of the first container.
            url2 (str): The URL of the second container.
            env1 (dict): The environment variables of the first container.
            env2 (dict): The environment variables of the second container.

        Returns:
            None
        """
        if env1 is None or env2 is None:
            return
        self.logger.info(f"Environment differences between {url1} and {url2}:")
        for key in env1.keys() - env2.keys():
            self.logger.info(f"  Only in {url1}: {key}={env1[key]}")
        for key in env2.keys() - env1.keys():
            self.logger.info(f"  Only in {url2}: {key}={env2[key]}")
        for key in env1.keys() & env2.keys():
            if env1[key] != env2[key]:
                self.logger.info(
                    f"  {key}: {env1[key]} ({url1}) vs. {env2[key]} ({url2})"
                )

    def _check_container_by_url(
        self, container_url: str, *, test_output_log_level: int = logging.DEBUG
    ) -> TestResult:
        """
        See if the test passes in the given container.

        Args:
            container_url (str): The URL of the container to check.
            test_output_log_level (int): The log level for test output.
        Returns:
            TestResult: The result of the test, including whether it passed and the output.
        """
        cached_result = self.restart_cache.get((CONTAINER_CACHE_SECTION, container_url))
        if cached_result is not None:
            self.logger.info(f"Reusing cached container result for {container_url}")
            return cached_result

        out_dir = self._test_output_directory(container_url, None)

        with self._make_container(
            container_url, test_output_directory=out_dir
        ) as worker:
            versions, _, _ = get_versions_dirs_env(
                worker=worker,
                versions_from_env=self.args.build_scripts_path is not None,
                optional_software=self.args.optional_software,
            )
            if self.args.container_runtime == "plugin":
                workload_version = None
                if self.args.passing_container == container_url:
                    workload_version = (self.args.passing_versions or {}).get(
                        _WORKLOAD_VERSION_KEY
                    )
                elif self.args.failing_container == container_url:
                    workload_version = (self.args.failing_versions or {}).get(
                        _WORKLOAD_VERSION_KEY
                    )
                elif (
                    _WORKLOAD_VERSION_KEY
                    in (self.args.passing_versions or {}).keys()
                    | (self.args.failing_versions or {}).keys()
                ):
                    self.logger.warning(
                        f"{_WORKLOAD_VERSION_KEY} was passed explicitly for some "
                        f"containers but not {container_url}"
                    )

                def _test():
                    return (
                        self._run_plugin(
                            container_url=container_url,
                            output_prefix=out_dir,
                            log_level=test_output_log_level,
                            workload_version=workload_version,
                        ),
                        out_dir,
                        container_url,
                    )
            else:

                def _test():
                    return (
                        worker.exec(
                            self.args.test_command, log_level=test_output_log_level
                        ),
                        out_dir,
                        container_url,
                    )

            test_result = self._run_test(_test)

        add_summary_record(
            self.args.output_prefix,
            "container",
            {
                **{
                    "container": container_url,
                    "output_directory": test_result.host_output_directory.as_posix(),
                    "result": str(test_result.result),
                    "test_time": test_result.time,
                    "metrics": test_result.metrics,
                },
                **versions,
            },
        )
        return test_result

    def _check_container_by_date(
        self, date: datetime.date, *, test_output_log_level: int = logging.DEBUG
    ) -> TestResult:
        """
        See if the test passes in the given dated container.

        Args:
            date (datetime.date): The date of the container to check.
            test_output_log_level (int): The log level for test output.
        Returns:
            TestResult: The result of the test, including whether it passed and the output.
        """
        container_url = container_url_base(
            date,
            container=self.args.container,
            template=self.args.container_url_template,
        )
        return self._check_container_by_url(
            container_url, test_output_log_level=test_output_log_level
        )

    def _check_installation_scripts(self, worker: Container) -> Set[str]:
        """
        Look for installation scripts that can be used to change the versions
        of packages like cuBLAS and cuDNN. These are expected to be named
        {build_scripts_path}/installPACKAGE.sh and to be executable.

        Args:
            worker (Container): The container in which to run the commands.
        Returns:
            packages_needing_scripts (set[str]): Packages whose versions are
                not static, but for which installation scripts were not found
        """
        if self.args.build_scripts_path is None:
            return set()

        known_scripts_result = worker.exec(
            [
                "sh",
                "-c",
                f'find ${{JAX_TOOLBOX_TRIAGE_PREFIX}}{self.args.build_scripts_path} -maxdepth 1 -type f -and -executable -print0 | sed -e "s|^${{JAX_TOOLBOX_TRIAGE_PREFIX}}||"',
            ],
            policy="once",
            stderr="separate",
        )
        if known_scripts_result.returncode != 0:
            raise Exception(
                f"Failed to find known installation scripts in {self.args.build_scripts_path}: {known_scripts_result.stderr}"
            )
        else:
            # Drop trailing \0
            known_scripts = known_scripts_result.stdout.split("\0")[:-1]
        if len(known_scripts) == 0:
            raise Exception(
                f"Failed to find known installation scripts in {self.args.build_scripts_path}"
            )

        self.logger.debug(f"Found {known_scripts} inside {worker}")

        self.packages_with_scripts = {
            script[len(self.args.build_scripts_path) + 8 : -3]
            for script in known_scripts
            if script.startswith(self.args.build_scripts_path + "/install")
            and script.endswith(".sh")
        }
        self.logger.debug(
            f"Found installation scripts for {self.packages_with_scripts}"
        )
        packages_needing_scripts = self.dynamic_packages - self.package_dirs.keys()
        packages_missing_scripts = packages_needing_scripts - self.packages_with_scripts
        return packages_missing_scripts

    def _run_plugin(
        self,
        container_url: str,
        output_prefix: pathlib.Path,
        log_level: int,
        workload_version: Optional[str] = None,
    ):
        assert self.args.container_runtime == "plugin"
        test_cmd = self.args.test_command + [
            "--container",
            container_url,
            "--output-prefix",
            str(output_prefix),
        ]
        if workload_version is not None:
            test_cmd += ["--workload-version", workload_version]
        return run_and_log(
            test_cmd,
            log_level=log_level,
            logger=self.logger,
            stderr="interleaved",
        )

    def _run_test(self, kernel):
        before = time.monotonic()
        result, out_dir, container_url = kernel()
        duration = time.monotonic() - before
        with open(out_dir / "test.log", "w") as log:
            log.write(result.stdout)
        metrics = {_EXIT_CODE_METRIC: result.returncode}
        if self.args.metric_name is None:
            # For non-metric triage there is always a result: the exit code
            result_enum = TestExecutionOutcome.TEST_YIELDED_RESULTS
        else:
            # metric-based triage
            metrics_file = out_dir / "metrics.json"
            try:
                with open(metrics_file) as ifile:
                    metrics.update(json.load(ifile))
                result_enum = TestExecutionOutcome.TEST_YIELDED_RESULTS
            except Exception as e:
                result_enum = TestExecutionOutcome.TEST_ERROR
                self.logger.fatal(f"Failed to extract metrics: {e}")
        self.logger.info(
            f"Test completed in {duration:.1f}s with metric values {metrics} in {container_url}"
        )
        return TestResult(
            build_stdouterr=None,  # May be filled in by the caller
            host_output_directory=out_dir,
            result=result_enum,
            stdouterr=result.stdout,
            metrics=metrics,
            time=duration,
        )

    def _build_and_test(
        self,
        *,
        versions: Dict[str, str],
        test_repetition: int = 0,
        test_output_log_level: int = logging.DEBUG,
    ) -> TestResult:
        """
        The main body of the bisection loop. Update JAX/XLA/... versions, rebuild, and
        run the test command. Throws on error when checking out or building, and returns
        the status of the test command.

        Args:
            versions (dict): The versions of the software packages to use.
            test_output_log_level (int): The log level for test output.

        Returns:
            TestResult: The result of the test, including whether it passed and the output.
        """
        # Amortise container startup overhead by batching together git commands
        git_commands, changed, skipped = [], [], []
        for package in sorted(self.dynamic_packages):
            version = versions[package]
            if self.bisection_versions.get(package) == version:
                # If the existing version is the desired one, do nothing.
                skipped.append(f"{package}@{version}")
                continue
            # We used to update bisection_versions here, which used to work with the
            # pyxis backend because it used to be the case that:
            #
            # with make_container(blah) as worker:
            #   worker.exec(echo foo > /bar)
            # with make_container(blah) as worker:
            #   worker.exec(cat /bar) # prints foo
            #
            # but now each new context manager is a fresh instance of `blah`, which was
            # how the docker backend already worked. This is also how the "plugin"
            # backend works. It does mean that e.g. bazel repository caching does not
            # work as well as it otherwise would out of the box.
            changed.append(f"{package}@{version}")
            if package in self.package_dirs:
                git_commands.append(
                    f"cd ${{JAX_TOOLBOX_TRIAGE_PREFIX}}{self.package_dirs[package]}"
                )
                git_commands.append("git stash")
                # this is a checkout on the main branch
                git_commands.append(f"git checkout {version}")
                for cherry_pick_range in self.args.cherry_pick.get(package, []):
                    git_commands.append(
                        f"(git cherry-pick {cherry_pick_range} || git cherry-pick --abort)"
                    )
            else:
                if package == _WORKLOAD_VERSION_KEY:
                    continue
                # Another software package, `version` is probably a version number.
                # Installation of this version is delegated to an installPACKAGE.sh
                # script that is assumed to be available in `args.build_scripts_path`.
                assert self.args.build_scripts_path is not None
                assert package in self.packages_with_scripts, (
                    package,
                    self.packages_with_scripts,
                )
                extra_env = {
                    # Need the static part for the .bc library
                    "NVSHMEM": "DEVEL=1 STATIC=1",
                }.get(package, "DEVEL=1")  # Always need development headers to rebuild
                git_commands += [
                    f"{extra_env} ${{JAX_TOOLBOX_TRIAGE_PREFIX}}{self.args.build_scripts_path}/install{package}.sh {version}"
                ]
        # Keep the pathnames shorter by only including packages that actually have
        # multiple versions in the bisection range.
        brief_versions = {
            p: ver
            for p, ver in versions.items()
            if p in self.dynamic_packages and p != _WORKLOAD_VERSION_KEY
        }
        brief_versions[_REPETITION_KEY] = str(test_repetition)
        out_dir = self._test_output_directory(
            self.bisection_url,
            versions=brief_versions,
        )
        change_str = " ".join(changed) if len(changed) else "<nothing>"
        info_str = f"Checking out {change_str}"
        if len(skipped):
            info_str += f", leaving {' '.join(skipped)} unchanged"
        push_intermediate_containers = self.args.container_runtime == "plugin"
        with (
            contextlib.nullcontext()
            if push_intermediate_containers
            else self._make_container(self.bisection_url, test_output_directory=out_dir)
        ) as worker:
            self.logger.info(f"{info_str}")
            # Build JAX
            # TODO: include a dependency graph and do not re-build components if they and their dependencies did not change
            build_cmds = []
            if not push_intermediate_containers:
                # Unfortunately the build system does not always seem to handle incremental
                # rebuilds correctly, so clean the local cache and rely on the remote one.
                # Not needed if we are pushing a container, because the local cache is not
                # included in it.
                build_cmds.append("bazel clean --expunge")
            if self.args.bazel_cache:
                build_cmds.append(f"build-jax.sh --bazel-cache={self.args.bazel_cache}")
            else:
                build_cmds.append("build-jax.sh")
            if not self.args.exclude_transformer_engine:
                if len(self.args.transformer_engine_ccache_env):
                    build_cmds.append(
                        f"env {' '.join(self.args.transformer_engine_ccache_env)} build-te.sh --ccache"
                    )
                else:
                    build_cmds.append("build-te.sh")
            summary = {}
            if push_intermediate_containers:
                # Build a new layer on top of `bisection_url` that includes running `build_cmds` as a new layer.
                tag_suffix = self._version_slug(
                    self.bisection_url,
                    versions={
                        k: v for k, v in brief_versions.items() if k != _REPETITION_KEY
                    },
                ) + f"-{platform.machine()}"
                if self.args.container_registry is None:
                    # Do not push, everything local.
                    container_name = tag_suffix
                elif ":" in self.args.container_registry:
                    # gitlab.com/USER/containers:PREFIX-
                    container_name = f"{self.args.container_registry}{tag_suffix}"
                else:
                    # gitlab.com/USER/containers
                    container_name = f"{self.args.container_registry}:{tag_suffix}"
                data_files_dir = pathlib.Path(__file__).parent / "data_files"

                def _build():
                    if DockerContainer(
                        container_name, logger=self.logger, mounts=[]
                    ).exists():
                        command = [
                            "echo",
                            f"Skipping building {container_name} because it already exists",
                        ]
                    else:
                        # TODO: organise this better to encourage layer sharing
                        command = [
                            "docker",
                            "buildx",
                            "build",
                            f"--build-arg=BASE_IMAGE={self.bisection_url}",
                            f"--build-arg=GIT_CMD={' && '.join(git_commands)}",
                            f"--build-arg=BUILD_CMD={' && '.join(build_cmds)}",
                            "--tag",
                            container_name,
                            "--file",
                            str(data_files_dir / "Dockerfile.triage-tool"),
                            str(data_files_dir),
                        ]
                        if self.args.container_registry is not None:
                            command.append("--push")
                    return run_and_log(
                        command,
                        logger=self.logger,
                        stderr="interleaved",
                    )

                def _test():
                    return (
                        self._run_plugin(
                            container_url=container_name,
                            output_prefix=out_dir,
                            log_level=test_output_log_level,
                            workload_version=versions.get(_WORKLOAD_VERSION_KEY),
                        ),
                        out_dir,
                        container_name,
                    )
            else:
                container_name = self.bisection_url

                def _build():
                    worker.check_exec(
                        ["sh", "-c", " && ".join(git_commands)],
                        policy="once_per_container",
                    )
                    return worker.exec(
                        ["sh", "-c", " && ".join(build_cmds)],
                        policy="once_per_container",
                        workdir=self.package_dirs["jax"],
                    )

                def _test():
                    return (
                        worker.exec(
                            self.args.test_command, log_level=test_output_log_level
                        ),
                        out_dir,
                        container_name,
                    )

            before = time.monotonic()
            build_result = _build()
            build_pass = build_result.returncode == 0
            middle = time.monotonic()
            with open(out_dir / "build.log", "w") as log:
                log.write(build_result.stdout)
            build_time = middle - before
            self.logger.info(
                f"Container {container_name} build {'succeeded' if build_pass else 'failed'} in {build_time:.1f}s"
            )
            if build_pass:
                # Time and run the test, extract metrics
                test_result = self._run_test(_test)
                test_result.build_stdouterr = build_result.stdout
            else:
                test_result = TestResult(
                    build_stdouterr=build_result.stdout,
                    host_output_directory=out_dir,
                    result=TestExecutionOutcome.BUILD_FAILURE,
                    stdouterr=None,
                    time=None,
                    metrics={},
                )
        summary = {
            "build_time": build_time,
            "container": container_name,
            "output_directory": out_dir.as_posix(),
            "result": str(test_result.result),
            "test_repetition": test_repetition,
            "test_time": test_result.time,
        }
        summary.update(versions)
        add_summary_record(self.args.output_prefix, "versions", summary)
        return test_result

    def find_container_range(self) -> Tuple[str, str]:
        """
        Find the range from the passing and failing containers.
        Returns a tuple of the start and end container names.
        """
        if self.args.container_runtime == "local":
            self.logger.info(
                "Skipping container-level search because --container-runtime=local"
            )
            return "local", "local"

        container_url_func = functools.partial(
            container_url_base,
            container=self.args.container,
            template=self.args.container_url_template,
        )

        if self.args.passing_container is None and self.args.failing_container is None:
            self.logger.info("Launching container-level search")
            range_start, range_end = container_search(
                container_exists=lambda date: make_container(
                    self.args.container_runtime,
                    container_url_func(date),
                    [],
                    self.logger,
                ).exists(),
                container_passes=self._check_container_by_date,
                start_date=self.args.start_date,
                end_date=self.args.end_date,
                logger=self.logger,
                skip_precondition_checks=self.args.skip_precondition_checks,
                threshold_days=self.args.threshold_days,
            )

            return container_url_func(range_start), container_url_func(range_end)

        self.logger.info(
            "Skipping container-level search because "
            f"--passing-container={self.args.passing_container} "
            f"({self.args.passing_versions}) and "
            f"--failing-container={self.args.failing_container} "
            f"({self.args.failing_versions}) were passed"
        )
        return self.args.passing_container, self.args.failing_container

    def gather_version_info(self, passing_url: str, failing_url: str):
        """
        Gather version information from the passing and failing containers.

        Args:
            passing_url (str): The URL of the passing container.
            failing_url (str): The URL of the failing container.

        """
        self.logger.info("Gathering version information...")
        versions_from_env = self.args.build_scripts_path is not None
        # Get the versions from the endpoint containers (if they exist), overridden by any
        # explicitly passed versions.
        (
            passing_versions,
            original_passing_versions,
            passing_package_dirs,
            passing_env,
        ) = self._get_versions(
            passing_url, self.args.passing_versions, versions_from_env
        )
        (
            failing_versions,
            original_failing_versions,
            failing_package_dirs,
            failing_env,
        ) = self._get_versions(
            failing_url, self.args.failing_versions, versions_from_env
        )

        self._log_environment_differences(
            passing_url, failing_url, passing_env, failing_env
        )

        # Choose an environment to do the version-level bisection in; use directory names that
        # match it, and track what the initial versions of the different packages are
        if self.args.container_runtime == "local":
            self.bisection_url = "local"
            self.bisection_versions = original_failing_versions
            self.package_dirs = failing_package_dirs
        elif failing_url is not None:
            # Prefer to use the newer/failing container if both are available, as it will usually
            # already have the required git history and will, therefore, not need to git fetch.
            self.bisection_url = failing_url
            # If we are using the "bad" container for bisection, check we can reproduce failure
            # in it first -- before checking out the "good" versions in it
            self.check_success_before_failure = False
            self.bisection_versions = original_failing_versions
            self.package_dirs = failing_package_dirs
        else:
            assert passing_url is not None
            self.bisection_url = passing_url
            self.bisection_versions = original_passing_versions
            self.package_dirs = passing_package_dirs

        # We only know how to handle software packages that have versions defined at
        # both ends of the range.
        inconsistent_keys = passing_versions.keys() ^ failing_versions.keys()
        if len(inconsistent_keys):
            self.logger.warning(
                f"Ignoring packages that only have defined versions in one endpoint: {' '.join(inconsistent_keys)}"
            )
            for k in inconsistent_keys:
                for d in [passing_versions, failing_versions]:
                    d.pop(k, None)

        # Not sure how to handle a package that does not have a defined version in the
        # bisection environment but that is expected to be included in the bisection...
        assert passing_versions.keys() == failing_versions.keys()
        unknown_initial_packages = (
            passing_versions.keys() - self.bisection_versions.keys()
        )
        # With a build+test plugin, we can allow the plugin to handle this case.
        assert (
            len(unknown_initial_packages) == 0
            or self.args.container_runtime == "plugin"
        ), (
            passing_versions.keys(),
            self.bisection_versions.keys(),
        )

        # Which packages have versions that are not always the same? There are three
        # relevant sets of versions: the starting values in the bisection environment,
        # the start/passing value for the bisection, and the end/failing value for the
        # bisection.
        static_packages = {
            pkg
            for pkg, _ in set(passing_versions.items())
            & set(failing_versions.items())
            & set(self.bisection_versions.items())
        }
        self.dynamic_packages = passing_versions.keys() - static_packages
        self.logger.info(f"Using {self.bisection_url} for version-level bisection...")
        assert self.package_dirs is not None
        # This is the set of versions that are already installed
        assert self.bisection_versions is not None
        return passing_versions, failing_versions

    def run_version_bisection(
        self,
        passing_versions: Dict[str, str],
        failing_versions: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Run the version bisection process.

        Args:
            passing_versions (dict): The versions that passed.
            failing_versions (dict): The versions that failed.

        Returns:
            Tuple[dict, TestResult]: The final versions and the test result.
        """
        if self.args.metric_name:
            classifier = MetricClassifier(
                metric_name=self.args.metric_name,
                passing_values=self.args.passing_metric,
                failing_values=self.args.failing_metric,
            )
        else:
            classifier = ExitCodeClassifier()
        # Prepare the container for the bisection
        with self._make_container(self.bisection_url) as worker:
            package_versions = self._gather_histories(
                worker, passing_versions, failing_versions
            )
            packages_missing_scripts = self._check_installation_scripts(worker)
        if packages_missing_scripts:
            self.logger.warning(
                "No installation scripts found for: "
                f"{' '.join(packages_missing_scripts)}, whose version(s) change "
                "across the bisection range. These will be excluded from the "
                "bisection, which may cause it not to converge!"
            )
            self.dynamic_packages -= packages_missing_scripts
            for package in packages_missing_scripts:
                del package_versions[package]

        # Run the version-level bisection
        self.logger.info("Running version-level bisection...")
        result_cache = {
            key: result
            for (section, key), result in self.restart_cache.items()
            if section == VERSION_CACHE_SECTION
        }
        if self.args.restart:
            self.logger.info(
                f"Loaded {len(result_cache)} completed version-level result(s) "
                f"from {self.args.output_prefix / 'summary.json'}"
            )
        try:
            result, last_known_good, first_known_bad = version_search(
                versions=package_versions,
                build_and_test=self._build_and_test,
                logger=self.logger,
                skip_precondition_checks=self.args.skip_precondition_checks,
                check_success_before_failure=self.check_success_before_failure,
                confirmation_iterations=self.args.confirmation_iterations,
                result_cache=result_cache,
                classifier=classifier,
            )
        except CouldNotReproduceFailure as e:
            if (
                self.args.failing_container is not None
                and self.bisection_url == self.args.passing_container
            ):
                # Could not reproduce failing with 'bad' versions in the 'good' container. Triage
                # will not succeed, but before exiting we can check if we can even reproduce
                # failure in the 'bad' container.
                self.logger.fatal(
                    f"Checking if failure can be reproduced in {self.args.failing_container}..."
                )
                check_fail = self._check_container_by_url(
                    self.args.failing_container, test_output_log_level=logging.INFO
                )
                check_fail_outcome = classifier([check_fail])
                if check_fail_outcome == ClassifiedTestOutcome.FAIL:
                    self.logger.fatal(
                        f"Reproduced failure in {self.args.failing_container} after failing to "
                        f"reproduce in {self.bisection_url}. This may mean the failure is due to "
                        "a variable not visible to the bisection tool."
                    )
                    raise InconsistentResults(
                        f"Reproduced failure in {self.args.failing_container} but not {self.bisection_url}"
                    ) from e
                else:
                    assert check_fail_outcome == ClassifiedTestOutcome.PASS, (
                        check_fail_outcome
                    )
                    raise CouldNotReproduceFailure(
                        f"Could not reproduce failure with 'bad' container ({self.args.failing_container}, {check_fail.result})"
                    ) from e

            raise
        except CouldNotReproduceSuccess as e:
            if (
                self.args.passing_container is not None
                and self.bisection_url == self.args.failing_container
            ):
                # Could not reproduce pass with 'good' versions in the 'bad' container. Triage
                # will not succeed, but before exiting we can check if success is reproducible
                # in the 'good' container.
                self.logger.fatal(
                    f"Checking if success can be reproduced in {self.args.passing_container}..."
                )
                check_pass = self._check_container_by_url(
                    self.args.passing_container, test_output_log_level=logging.INFO
                )
                check_pass_outcome = classifier([check_pass])
                if check_pass_outcome == ClassifiedTestOutcome.PASS:
                    self.logger.fatal(
                        f"Reproduced success in {self.args.passing_container} after failing to "
                        f"reproduce in {self.bisection_url}. This may mean the failure is due to "
                        "a variable not visible to the bisection tool."
                    )
                    raise InconsistentResults(
                        f"Reproduced success in {self.args.passing_container} but not {self.bisection_url}"
                    ) from e
                else:
                    assert check_pass_outcome == ClassifiedTestOutcome.FAIL, (
                        check_pass_outcome
                    )
                    raise CouldNotReproduceSuccess(
                        f"Could not reproduce success with 'good' container ({self.args.passing_container}, {check_pass.result})"
                    ) from e
            raise
        # Write final summary
        create_output_symlinks(
            self.args.output_prefix, last_known_good, first_known_bad
        )
        result["container"] = self.bisection_url
        self.logger.info("Version-level bisection completed")
        return add_summary_record(
            self.args.output_prefix, "result", result, scalar=True
        )
