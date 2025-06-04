import pytest
from jax_toolbox_triage.args import parse_args

test_command = ["my-test-command"]
valid_start_end_container = [
    "--passing-container",
    "passing-url",
    "--failing-container",
    "failing-url",
]
valid_container_and_commits = [
    [
        "--passing-container",
        "passing-url",
        "--failing-commits",
        "jax:0123456789,xla:fedcba9876543210",
    ],
    [
        "--failing-container",
        "failing-url",
        "--passing-commits",
        "xla:fedcba9876543210,jax:0123456789",
    ],
    [
        "--passing-container",
        "passing-url",
        "--passing-commits",
        "jax:123",  # xla not needed, the value from passing-container can be read
        "--failing-container",
        "failing-url",
    ],
    [
        "--failing-container",
        "failing-url",
        "--failing-commits",
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


@pytest.mark.parametrize(
    "good_args",
    [valid_start_end_container]
    + valid_start_end_date_args
    + valid_container_and_commits,
)
def test_good_container_args(good_args):
    args = parse_args(good_args + test_command)
    assert args.test_command == test_command


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
        # If --passing-container (or --passing-commits) is passed then
        # --failing-container (or --failing-commits) must be too
        ["--passing-container", "passing-url"],
        ["--failing-container", "failing-url"],
        # Need at least one container
        [
            "--passing-commits",
            "jax:0123456789,xla:fedcba9876543210",
            "--failing-commits",
            "xla:fedcba9876543210,jax:0123456789",
        ],
        # --{passing,failing}-commits must be formatted correctly
        [
            "--passing-container",
            "passing-url",
            "--failing-commits",
            # no xla
            "jax:123",
        ],
        ["--passing-container", "passing-url", "--failing-commits", "jax:123,jax:456"],
        [
            "--passing-container",
            "passing-url",
            "--failing-commits",
            # no jax
            "xla:123",
        ],
        [
            "--passing-container",
            "passing-url",
            "--failing-commits",
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
