from jax_toolbox_triage.triage_tool import (
    _git_fetch_refs,
    _remote_without_credentials,
)


def test_remote_credentials_are_removed_from_build_url():
    remote = (
        "https://gitlab-ci-token:secret-token@"
        "gitlab-master.nvidia.com/dl/jax/jax.git"
    )

    sanitized, credentials = _remote_without_credentials(remote)

    assert sanitized == "https://gitlab-master.nvidia.com/dl/jax/jax.git"
    assert credentials == (
        "gitlab-master.nvidia.com",
        "gitlab-ci-token",
        "secret-token",
    )
    assert "secret-token" not in sanitized


def test_remote_without_credentials_is_unchanged():
    remote = "origin"

    assert _remote_without_credentials(remote) == (remote, None)


def test_fetch_refs_include_checkout_and_cherry_pick_endpoints():
    assert _git_fetch_refs(
        "selected",
        ["passing-main..passing-container", "explicit-fix"],
    ) == [
        "selected",
        "passing-main",
        "passing-container",
        "explicit-fix",
    ]
