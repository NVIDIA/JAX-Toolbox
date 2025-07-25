import collections
import datetime
import functools
import hashlib
import logging
import pathlib
import time
from typing import Dict, Tuple, Union, Any, Optional

from .container import Container
from .logic import container_search, TestResult, version_search
from .versions import get_versions_dirs_env
from .summary import add_summary_record, create_output_symlinks
from .bisect import get_commit_history
from .utils import (
    container_url as container_url_base,
    prepare_bazel_cache_mounts,
)
from .container_factory import make_container


class TriageTool:
    """
    This is the main class that orchestrates the whole triage process.
    """

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.bisection_url = None
        self.bisection_versions = {}
        self.package_dirs = None
        self.dynamic_packages = set()
        self.packages_with_scripts = set()
        self.bazel_cache_mounts = prepare_bazel_cache_mounts(self.args.bazel_cache)
        # the cherry-pick gets populated only for non-linear cases
        self.cherry_pick_commits = {}

        self.logger.info("Arguments:")
        for k, v in vars(self.args).items():
            logger.info(f"  {k}: {v}")
        logger.info(
            "Verbose output, including stdout/err of triage commands, will be written "
            f"to {(self.args.output_prefix / 'debug.log').resolve()}"
        )

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
        hash_chars = 8
        urlhash = f"container-{hashlib.sha1(url.encode()).hexdigest()[:hash_chars]}"
        out_dirname = "-".join(
            [urlhash]
            + [f"{k}-{v[:hash_chars]}" for k, v in sorted((versions or {}).items())]
        )

        out_dir = self.args.output_prefix / out_dirname
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

        return make_container(self.args.container_runtime, url, mounts, self.logger)

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
            url_versions, dirs, env = get_versions_dirs_env(worker, versions_from_env)
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
            history, cherry_pick_range = get_commit_history(
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
            if cherry_pick_range is not None:
                self.cherry_pick_commits[package] = cherry_pick_range

            assert all(
                b[1] >= a[1]
                for a, b in zip(
                    package_versions[package], package_versions[package][1:]
                )
            )

            # this check works only for linear-case
            if package not in self.cherry_pick_commits:
                assert passing_versions[package] == package_versions[package][0][0]
                assert failing_versions[package] == package_versions[package][-1][0]

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

        before = time.monotonic()
        out_dir = self._test_output_directory(container_url, None)

        with self._make_container(
            container_url, test_output_directory=out_dir
        ) as worker:
            versions, _, _ = get_versions_dirs_env(
                worker, self.args.build_scripts_path is not None
            )
            result = worker.exec(
                self.args.test_command, log_level=test_output_log_level
            )
            test_time = time.monotonic() - before
            test_pass = result.returncode == 0
            self.logger.info(
                f"Ran test case in {worker} in {test_time:.1f}s, pass={test_pass}"
            )

        add_summary_record(
            self.args.output_prefix,
            "container",
            {
                **{
                    "container": container_url,
                    "output_directory": out_dir.as_posix(),
                    "result": test_pass,
                    "test_time": test_time,
                },
                **versions,
            },
        )
        return TestResult(
            host_output_directory=out_dir, result=test_pass, stdouterr=result.stdout
        )

    def _check_installation_scripts(self, worker: Container):
        """
        Check for special installation cases, like cuBLAS or cuDNN

        Args:
            worker (Container): The container in which to run the commands.
        """
        if self.args.build_scripts_path is None:
            return

        known_scripts_result = worker.exec(
            [
                "find",
                self.args.build_scripts_path,
                "-maxdepth",
                "1",
                "-executable",
                "-print0",
            ],
            policy="once",
            stderr="separate",
        )
        if known_scripts_result.returncode != 0:
            self.logger.warning(
                f"Failed to find known installation scripts in {self.args.build_scripts_path}: {known_scripts_result.stderr}"
            )
            known_scripts = []
        else:
            known_scripts = known_scripts_result.stdout.split("\0")

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
        if packages_missing_scripts:
            self.logger.warning(
                "No installation scripts found for: "
                f"{' '.join(packages_missing_scripts)}, whose version(s) change "
                "across the bisection range. These will be excluded from the "
                "bisection, which may cause it not to converge!"
            )
            self.dynamic_packages -= packages_missing_scripts

    def _build_and_test(
        self,
        *,
        versions: Dict[str, str],
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
            # Cache which version is now going to be checked out in the container
            self.bisection_versions[package] = version
            changed.append(f"{package}@{version}")
            if package in self.package_dirs:
                cherry_pick_range = self.cherry_pick_commits.get(package)
                git_commands.append(f"cd {self.package_dirs[package]}")
                git_commands.append("git stash")
                # this is a checkout on the main branch
                git_commands.append(f"git checkout {version}")
                if cherry_pick_range:
                    git_commands.append(f"git cherry-pick {cherry_pick_range}")

            else:
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
                    f"{extra_env} {self.args.build_scripts_path}/install{package}.sh {version}"
                ]
        # Keep the pathnames shorter by only including packages that actually have
        # multiple versions in the bisection range.
        brief_versions = {
            p: ver for p, ver in versions.items() if p in self.dynamic_packages
        }
        out_dir = self._test_output_directory(
            self.bisection_url, versions=brief_versions
        )

        with self._make_container(
            self.bisection_url, test_output_directory=out_dir
        ) as worker:
            change_str = " ".join(changed) if len(changed) else "<nothing>"
            info_str = f"Checking out {change_str} in {worker}"
            if len(skipped):
                info_str += f", leaving {' '.join(skipped)} unchanged"
            self.logger.info(info_str)
            worker.check_exec(
                ["sh", "-c", " && ".join(git_commands)],
                policy="once_per_container",
            )
            # Build JAX
            # TODO: teach the tool how to build TransformerEngine too
            # TODO: do not build JAX/XLA/TransformerEngine if we know their versions did not change?
            before = time.monotonic()
            # Unfortunately the build system does not always seem to handle incremental
            # rebuilds correctly, so clean the local cache and rely on the remote one.
            build_cmds = [
                "bazel clean --expunge",
                f"build-jax.sh --bazel-cache={self.args.bazel_cache}",
                "build-te.sh",
            ]
            worker.check_exec(
                ["sh", "-c", " && ".join(build_cmds)],
                policy="once_per_container",
                workdir=self.package_dirs["jax"],
            )
            middle = time.monotonic()
            self.logger.info(f"Build completed in {middle - before:.1f}s")
            # Run the test
            test_result = worker.exec(
                self.args.test_command, log_level=test_output_log_level
            )
            test_time = time.monotonic() - middle

        add_summary_record(
            self.args.output_prefix,
            "versions",
            {
                **{
                    "build_time": middle - before,
                    "container": self.bisection_url,
                    "output_directory": out_dir.as_posix(),
                    "result": test_result.returncode == 0,
                    "test_time": test_time,
                },
                **versions,
            },
        )
        result_str = "pass" if test_result.returncode == 0 else "fail"
        self.logger.info(f"Test completed in {test_time:.1f}s ({result_str})")
        return TestResult(
            host_output_directory=out_dir,
            result=test_result.returncode == 0,
            stdouterr=test_result.stdout,
        )

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

        # We should have versions for all the same software packages at both
        # ends of the range, one way or another. TODO: this could be relaxed.
        assert passing_versions.keys() == failing_versions.keys(), (
            passing_versions,
            failing_versions,
        )
        # Which packages have versions that are not always the same?
        self.dynamic_packages = {
            pkg
            for pkg, _ in set(passing_versions.items()) ^ set(failing_versions.items())
        }
        # Choose an environment to do the version-level bisection in; use directory names that
        # match it, and track what the initial versions of the different packages are
        if self.args.container_runtime == "local":
            self.bisection_url = "local"
            self.bisection_versions = original_failing_versions
            self.package_dirs = failing_package_dirs
        elif failing_url is not None:
            self.bisection_url = failing_url
            self.bisection_versions = original_failing_versions
            self.package_dirs = failing_package_dirs
        else:
            assert passing_url is not None
            self.bisection_url = passing_url
            self.bisection_versions = original_passing_versions
            self.package_dirs = passing_package_dirs
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
        # Prepare the container for the bisection
        with self._make_container(self.bisection_url) as worker:
            package_versions = self._gather_histories(
                worker, passing_versions, failing_versions
            )
            self._check_installation_scripts(worker)

        # Run the version-level bisection
        self.logger.info("Running version-level bisection...")
        result, last_known_good, first_known_bad = version_search(
            versions=package_versions,
            build_and_test=self._build_and_test,
            logger=self.logger,
            skip_precondition_checks=self.args.skip_precondition_checks,
        )
        # Write final summary
        create_output_symlinks(
            self.args.output_prefix, last_known_good, first_known_bad
        )
        result["container"] = self.bisection_url
        self.logger.info("Version-level bisection completed")
        return add_summary_record(
            self.args.output_prefix, "result", result, scalar=True
        )
