import pytest
from jax_toolbox_triage.args import parse_args

test_command = ["my-test-command"]
valid_start_end_container = [
    "--passing-container",
    "passing-url",
    "--failing-container",
    "failing-url",
]
valid_container_and_versions = [
    [
        "--passing-container",
        "passing-url",
        "--failing-versions",
        "jax:0123456789,xla:fedcba9876543210",
    ],
    [
        "--failing-container",
        "failing-url",
        "--passing-versions",
        "xla:fedcba9876543210,jax:0123456789",
    ],
    [
        "--passing-container",
        "passing-url",
        "--passing-versions",
        "jax:123",  # xla not needed, the value from passing-container can be read
        "--failing-container",
        "failing-url",
    ],
    [
        "--failing-container",
        "failing-url",
        "--failing-versions",
        "xla:456",  # jax not needed, the value from failing-container can be read
        "--passing-container",
        "passing-url",
    ],
]
valid_start_end_date_args = [
    ["--container", "jax"],
    ["--container", "jax", "--start-date", "2024-10-02"],
    ["--container", "jax", "--end-date", "2024-10-02"],
    ["--container", "jax", "--start-date", "2024-10-01", "--end-date", "2024-10-02"],
]
valid_local_args = [
    "--container-runtime",
    "local",
    "--passing-versions",
    "jax:0123,xla:4567",
    "--failing-versions",
    "jax:89ab,xla:cdef",
]


@pytest.mark.parametrize(
    "good_args",
    [valid_start_end_container, valid_local_args]
    + valid_start_end_date_args
    + valid_container_and_versions,
)
def test_good_container_args(good_args):
    args = parse_args(good_args + test_command)
    assert args.test_command == test_command


def test_good_local_args():
    args = parse_args(valid_local_args + test_command)
    assert args.test_command == test_command
    assert args.container_runtime == "local"
    assert "jax" in args.passing_versions
    assert "xla" in args.failing_versions


@pytest.mark.parametrize("date_args", valid_start_end_date_args)
def test_bad_container_arg_combinations_across_groups(date_args):
    # Can't combine --{start,end}-container with --container/--{start,end}-date
    with pytest.raises(Exception):
        parse_args(valid_start_end_container + date_args + test_command)


@pytest.mark.parametrize(
    "container_args",
    [
        # Need --container
        [],
        ["--start-date", "2024-10-01"],
        ["--end-date", "2024-10-02"],
        ["--start-date", "2024-10-01", "--end-date", "2024-10-02"],
        # If --passing-container (or --passing-versions) is passed then
        # --failing-container (or --failing-versions) must be too
        ["--passing-container", "passing-url"],
        ["--failing-container", "failing-url"],
        # Need at least one container
        [
            "--passing-versions",
            "jax:0123456789,xla:fedcba9876543210",
            "--failing-versions",
            "xla:fedcba9876543210,jax:0123456789",
        ],
        # --{passing,failing}-versions must be formatted correctly
        [
            "--passing-container",
            "passing-url",
            "--failing-versions",
            # no xla
            "jax:123",
        ],
        ["--passing-container", "passing-url", "--failing-versions", "jax:123,jax:456"],
        [
            "--passing-container",
            "passing-url",
            "--failing-versions",
            # no jax
            "xla:123",
        ],
        [
            "--passing-container",
            "passing-url",
            "--failing-versions",
            # neither jax nor xla
            "bob:123",
        ],
    ],
)
def test_bad_container_arg_combinations_within_groups(container_args):
    with pytest.raises(Exception):
        parse_args(container_args + test_command)


@pytest.mark.parametrize(
    "container_args",
    [
        # Need valid ISO dates
        ["--container", "jax", "--start-date", "a-blue-moon-ago"],
        ["--container", "jax", "--end-date", "a-year-ago-last-thursday"],
    ],
)
def test_unparsable_container_args(container_args):
    with pytest.raises(SystemExit):
        parse_args(container_args + test_command)


def test_invalid_container_runtime():
    with pytest.raises(Exception):
        parse_args(["--container-runtime=magic-beans"] + test_command)


@pytest.mark.parametrize(
    "bad_local_args",
    [
        ["--container-runtime", "local"],
        ["--container-runtime", "local", "--passing-versions", "jax:1,xla:2"],
        ["--container-runtime", "local", "--failing-versions", "jax:1,xla:2"],
        valid_local_args + ["--container", "jax"],
        valid_local_args + ["--start-date", "2024-01-01"],
        valid_local_args + ["--passing-container", "url"],
        valid_local_args + ["--failing-container", "url"],
    ],
)
def test_bad_local_arg_combinations(bad_local_args):
    with pytest.raises(Exception):
        parse_args(bad_local_args + test_command)


@pytest.mark.parametrize(
    "args",
    [
        ["--passing-commits", "jax:1,xla:2", "--passing-versions", "jax:2,xla:2"],
        ["--failing-commits", "jax:1,xla:2", "--failing-versions", "jax:2,xla:2"],
        [
            "--failing-commits",
            "jax:1,xla:2",
            "--failing-versions",
            "jax:2,xla:2",
            "--passing-commits",
            "jax:1,xla:2",
            "--passing-versions",
            "jax:2,xla:2",
        ],
    ],
)
def test_combining_deprecated_args_with_their_replacements(args):
    with pytest.raises(Exception):
        parse_args(args + test_command)


@pytest.mark.parametrize(
    "args",
    [
        ["--passing-commits", "jax:1,xla:2", "--failing-container", "url"],
        ["--failing-commits", "jax:1,xla:2", "--passing-container", "url"],
        [
            "--failing-commits",
            "jax:1,xla:2",
            "--passing-commits",
            "jax:1,xla:2",
            "--passing-container",
            "url",
        ],
    ],
)
def test_warning_on_deprecated_args(args):
    with pytest.deprecated_call():
        parse_args(args + test_command)
