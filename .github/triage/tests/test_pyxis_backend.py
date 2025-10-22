import functools
import logging
import pathlib
import pytest
import shutil
import subprocess
import tempfile


from jax_toolbox_triage.args import compulsory_software, parse_args
from jax_toolbox_triage.triage_tool import TriageTool

mock_scripts_path = pathlib.Path(__file__).parent / "mock_scripts"


pyxis_args = [
    # Currently no way to use the pyxis backend without a cache
    "--bazel-cache=https://example.com/does-not-exist",
    "--container-runtime=pyxis",
]


def git_cmd(*args, cwd=None):
    return subprocess.run(
        ["git"] + list(args),
        capture_output=True,
        check=True,
        cwd=cwd,
        text=True,
    ).stdout.strip()


@pytest.fixture(params=["linear", "non-linear"])
def passing_container(request):
    scenario = request.param
    with tempfile.TemporaryDirectory(suffix=f"-passing-{scenario}") as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        (temp_path / "root").mkdir()
        metadata = {"prefix": temp_path, "scenario": scenario}
        for project in compulsory_software:
            repo_path = temp_path / "opt" / project
            repo_path.mkdir(parents=True)
            git = functools.partial(git_cmd, cwd=repo_path)
            git("init", "-b", "main")
            git("config", "user.name", "Test User")
            git("config", "user.email", "test@example.com")
            if scenario == "non-linear":
                # Regular commits underneath F0
                git("commit", "--allow-empty", "-m", "C0")
                git("commit", "--allow-empty", "-m", "C1")
                git("checkout", "-b", "feature-1")
            # This is needed for the mock build-jax.sh to succeed
            (repo_path / "feature_file.txt").write_text("feature")
            git("add", "feature_file.txt")
            git("commit", "-m", "F0")
            metadata[f"{project}_feature_commit"] = git("rev-parse", "HEAD")
            if scenario == "linear":
                # Regular commits on top of F0, which is always present
                git("commit", "--allow-empty", "-m", "C0")
                git("commit", "--allow-empty", "-m", "C1")
            metadata[f"{project}_passing_container"] = git("rev-parse", "HEAD")
        yield metadata


@pytest.fixture
def failing_container(passing_container):
    scenario = passing_container["scenario"]
    with tempfile.TemporaryDirectory(suffix=f"-failing-{scenario}") as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        metadata = {"prefix": temp_path}
        shutil.copytree(passing_container["prefix"], temp_path, dirs_exist_ok=True)
        for project in compulsory_software:
            git = functools.partial(git_cmd, cwd=temp_path / "opt" / project)
            if scenario == "non-linear":
                git("checkout", "main")
            git("commit", "--allow-empty", "-m", "C2")
            metadata[f"{project}_good"] = git("rev-parse", "HEAD")
            git("commit", "--allow-empty", "-m", "C3")
            metadata[f"{project}_bad"] = git("rev-parse", "HEAD")
            git("commit", "--allow-empty", "-m", "C4")
            git("commit", "--allow-empty", "-m", "C5")
            git("commit", "--allow-empty", "-m", "C6")
            c6 = git("rev-parse", "HEAD")
            if scenario == "non-linear":
                # cherry-pick the feature on top of the good commit
                git("checkout", metadata[f"{project}_good"])
                git("cherry-pick", passing_container[f"{project}_feature_commit"])
                metadata[f"{project}_good_with_feature"] = git("rev-parse", "HEAD")
                git("checkout", "-b", "feature-2")
                git("reset", "--hard", c6)
                git("cherry-pick", passing_container[f"{project}_feature_commit"])
            else:
                metadata[f"{project}_good_with_feature"] = metadata[f"{project}_good"]
            metadata[f"{project}_failing_container"] = git("rev-parse", "HEAD")
        yield metadata


@pytest.mark.parametrize("num_nodes", [1, 2])
@pytest.mark.parametrize("processes_per_node", [1, 2])
def test_mock_containers(
    monkeypatch,
    passing_container,
    failing_container,
    num_nodes,
    processes_per_node,
):
    # Doesn't really add anything to test both values
    bad_package = compulsory_software[0]
    # Tell the mock `srun` how to behave
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_NODES", str(num_nodes))
    monkeypatch.setenv(
        "JAX_TOOLBOX_TRIAGE_MOCK_SRUN_PROCS_PER_NODE", str(processes_per_node)
    )
    # Ensure bazel, build-jax.sh, srun etc. stubs can be found.
    monkeypatch.setenv("PATH", str(mock_scripts_path), prepend=":")
    with tempfile.TemporaryDirectory() as output_prefix:
        arg_list = pyxis_args + [
            "--output-prefix",
            output_prefix,
            "--passing-container",
            str(passing_container["prefix"]),
            "--failing-container",
            str(failing_container["prefix"]),
            "--",
            "test-case.sh",
            f"/opt/{bad_package}",
            failing_container[f"{bad_package}_bad"],
        ]
        args = parse_args(arg_list)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        tool = TriageTool(args, logger)
        # Check the correct versions are extracted from the two pseudocontainers
        passing_versions, failing_versions = tool.gather_version_info(
            args.passing_container, args.failing_container
        )
        for package in compulsory_software:
            assert (
                passing_versions[package]
                == passing_container[f"{package}_passing_container"]
            )
            assert (
                failing_versions[package]
                == failing_container[f"{package}_failing_container"]
            )
        # Run the bisection
        summary_data = tool.run_version_bisection(passing_versions, failing_versions)
        assert "result" in summary_data, summary_data
        assert (
            summary_data["result"][f"{bad_package}_good"]
            == failing_container[f"{bad_package}_good"]
        )
        assert (
            summary_data["result"][f"{bad_package}_bad"]
            == failing_container[f"{bad_package}_bad"]
        )


def test_mock_containers_with_explicit_version_override(
    monkeypatch,
    passing_container,
    failing_container,
):
    # The point of this test is that if you pass
    # --passing-container=<has X=a> --failing-container=<has X=c>,
    # --passing-versions=X=b        --failing-versions=X=b
    # then it is important to check out version b of package X in the triage
    # environment, even though it is not actually included in the triage

    # fixed_package is forced to its good value by --{passing,failing}-versions, and
    # the test command only passes if it has *exactly* that value -- which it doesn't
    # initially have in either container. `_with_feature` is needed to make sure the
    # fake build succeeds.
    triage_package, fixed_package = compulsory_software[:2]
    fixed_good_commit = failing_container[f"{fixed_package}_good_with_feature"]
    # Tell the mock `srun` how to behave
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_NODES", str(1))
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_PROCS_PER_NODE", str(1))
    # Ensure bazel, build-jax.sh, srun etc. stubs can be found.
    monkeypatch.setenv("PATH", str(mock_scripts_path), prepend=":")
    with tempfile.TemporaryDirectory() as output_prefix:
        arg_list = pyxis_args + [
            "--output-prefix",
            output_prefix,
            "--passing-container",
            str(passing_container["prefix"]),
            "--passing-versions",
            f"{fixed_package}:{fixed_good_commit}",
            "--failing-container",
            str(failing_container["prefix"]),
            "--failing-versions",
            f"{fixed_package}:{fixed_good_commit}",
            "--",
            "sh",
            "-c",
            " && ".join(
                [
                    f'[ $(cd ${{JAX_TOOLBOX_TRIAGE_PREFIX}}/opt/{fixed_package} && git rev-parse HEAD) = "{fixed_good_commit}" ]',
                    f"test-case.sh /opt/{triage_package} {failing_container[f'{triage_package}_bad']}",
                ]
            ),
        ]
        args = parse_args(arg_list)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        tool = TriageTool(args, logger)
        # Check the correct versions are extracted from the two pseudocontainers
        passing_versions, failing_versions = tool.gather_version_info(
            args.passing_container, args.failing_container
        )
        # These are not overridden, they are read from the containers
        assert (
            passing_versions[triage_package]
            == passing_container[f"{triage_package}_passing_container"]
        )
        assert (
            failing_versions[triage_package]
            == failing_container[f"{triage_package}_failing_container"]
        )
        # These are overridden by --passing-version and --failing-version
        assert passing_versions[fixed_package] == fixed_good_commit
        assert failing_versions[fixed_package] == fixed_good_commit
        # The starting value is not the value it is fixed to
        assert tool.bisection_versions[fixed_package] != fixed_good_commit
        assert tool.bisection_versions[fixed_package] in {
            passing_container[f"{fixed_package}_passing_container"],
            failing_container[f"{fixed_package}_failing_container"],
        }
        # fixed_package is dynamic, because its version needs to be changed from its starting value
        assert fixed_package in tool.dynamic_packages
        # triage_package is dynamic, because it is being triaged
        assert triage_package in tool.dynamic_packages
        # Run the bisection
        summary_data = tool.run_version_bisection(passing_versions, failing_versions)
        assert "result" in summary_data, summary_data
        assert (
            summary_data["result"][f"{triage_package}_good"]
            == failing_container[f"{triage_package}_good"]
        )
        assert (
            summary_data["result"][f"{triage_package}_bad"]
            == failing_container[f"{triage_package}_bad"]
        )


@pytest.fixture
def passing_container_with_bad_library(passing_container):
    scenario = passing_container["scenario"]
    with tempfile.TemporaryDirectory(suffix=f"-passing-{scenario}-bad-lib") as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        shutil.copytree(passing_container["prefix"], temp_path, dirs_exist_ok=True)
        scripts_dir = temp_path / "build-scripts"
        scripts_dir.mkdir()
        shutil.copy(mock_scripts_path / "installPACKAGE.sh", scripts_dir)
        with open(temp_path / ".env", "w") as env_file:
            env_file.write("PACKAGE_VERSION=bad\n")
            env_file.write("PACKAGE_WITHOUT_SCRIPT_VERSION=new\n")
            env_file.write("STATIC_PACKAGE_WITHOUT_SCRIPT_VERSION=fixed\n")
        yield {"prefix": temp_path}


@pytest.fixture
def passing_container_with_good_library(passing_container):
    scenario = passing_container["scenario"]
    with tempfile.TemporaryDirectory(
        suffix=f"-passing-{scenario}-good-lib"
    ) as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        shutil.copytree(passing_container["prefix"], temp_path, dirs_exist_ok=True)
        scripts_dir = temp_path / "build-scripts"
        scripts_dir.mkdir()
        shutil.copy(mock_scripts_path / "installPACKAGE.sh", scripts_dir)
        with open(temp_path / ".env", "w") as env_file:
            env_file.write("PACKAGE_VERSION=good\n")
            env_file.write("PACKAGE_WITHOUT_SCRIPT_VERSION=old\n")
            env_file.write("STATIC_PACKAGE_WITHOUT_SCRIPT_VERSION=fixed\n")
        yield {"prefix": temp_path}


def test_triage_with_missing_installation_script_dir(
    monkeypatch, passing_container, failing_container
):
    # Tell the mock `srun` how to behave
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_NODES", "1")
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_PROCS_PER_NODE", "1")
    # Ensure the srun stub can be found
    monkeypatch.setenv("PATH", str(mock_scripts_path), prepend=":")
    arg_list = pyxis_args + [
        "--build-scripts",
        "/path-does-not-exist",
        "--passing-container",
        str(passing_container["prefix"]),
        "--failing-container",
        str(failing_container["prefix"]),
        "false",
    ]
    args = parse_args(arg_list)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    tool = TriageTool(args, logger)
    passing_versions, failing_versions = tool.gather_version_info(
        args.passing_container, args.failing_container
    )
    with pytest.raises(
        Exception,
        match="Failed to find known installation scripts in /path-does-not-exist",
    ):
        tool.run_version_bisection(passing_versions, failing_versions)


def test_triage_with_installation_scripts(
    caplog,
    monkeypatch,
    passing_container_with_bad_library,
    passing_container_with_good_library,
):
    caplog.set_level(logging.DEBUG)
    # Tell the mock `srun` how to behave
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_NODES", "1")
    monkeypatch.setenv("JAX_TOOLBOX_TRIAGE_MOCK_SRUN_PROCS_PER_NODE", "1")
    # Ensure bazel, build-jax.sh, srun etc. stubs can be found.
    monkeypatch.setenv("PATH", str(mock_scripts_path), prepend=":")
    with tempfile.TemporaryDirectory() as output_prefix:
        arg_list = pyxis_args + [
            "--build-scripts",
            "/build-scripts",
            "--output-prefix",
            output_prefix,
            "--passing-container",
            str(passing_container_with_good_library["prefix"]),
            "--failing-container",
            str(passing_container_with_bad_library["prefix"]),
            "--",
            "sh",
            "-c",
            "[ $PACKAGE_VERSION = good ]",
        ]
        args = parse_args(arg_list)
        tool = TriageTool(args, logging.getLogger())
        # Check the correct versions are extracted from the two pseudocontainers
        passing_versions, failing_versions = tool.gather_version_info(
            args.passing_container, args.failing_container
        )
        assert passing_versions["PACKAGE"] == "good"
        assert failing_versions["PACKAGE"] == "bad"
        assert passing_versions["PACKAGE_WITHOUT_SCRIPT"] == "old"
        assert failing_versions["PACKAGE_WITHOUT_SCRIPT"] == "new"
        assert passing_versions["STATIC_PACKAGE_WITHOUT_SCRIPT"] == "fixed"
        assert failing_versions["STATIC_PACKAGE_WITHOUT_SCRIPT"] == "fixed"
        for package in compulsory_software:
            assert passing_versions[package] == failing_versions[package]
        # Run the bisection
        caplog.clear()
        summary_data = tool.run_version_bisection(passing_versions, failing_versions)

        # There should be a warning that PACKAGE_WITHOUT_SCRIPT is not triageable
        def my_warning(record):
            return (
                record.levelname == "WARNING"
                and record.message
                == "No installation scripts found for: PACKAGE_WITHOUT_SCRIPT, whose version(s) change across the bisection range. These will be excluded from the bisection, which may cause it not to converge!"
            )

        assert sum(map(my_warning, caplog.records)) == 1
        assert "result" in summary_data, summary_data
        assert summary_data["result"]["PACKAGE_good"] == "good"
        assert summary_data["result"]["PACKAGE_bad"] == "bad"
        # FIXME: don't completely drop version information in this case so
        #        that we can assert this value is in {"new", "old"} instead
        assert "PACKAGE_WITHOUT_SCRIPT_ref" not in summary_data["result"]
        assert summary_data["result"]["STATIC_PACKAGE_WITHOUT_SCRIPT_ref"] == "fixed"
