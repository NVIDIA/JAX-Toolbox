import pytest
from jax_toolbox_triage.args import parse_args

test_command = ["my-test-command"]
valid_start_end_container = [
    "--passing-container",
    "passing-url",
    "--failing-container",
    "failing-url",
]
valid_start_end_date_args = [
    ["--container", "jax"],
    ["--container", "jax", "--start-date", "2024-10-02"],
    ["--container", "jax", "--end-date", "2024-10-02"],
    ["--container", "jax", "--start-date", "2024-10-01", "--end-date", "2024-10-02"],
]


@pytest.mark.parametrize(
    "good_args", [valid_start_end_container] + valid_start_end_date_args
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
        # Need both if either is passed
        ["--passing-container", "passing-url"],
        ["--failing-container", "failing-url"],
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
