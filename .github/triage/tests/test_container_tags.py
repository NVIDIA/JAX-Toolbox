from jax_toolbox_triage.triage_tool import _bounded_tag_suffix


def test_container_tag_suffix_is_bounded_and_stable():
    registry = "gitlab-master.nvidia.com/dl/dgx/jax:triage-"
    suffix = (
        "container-194a08c5-xla-569d0d78_569d0d78_089e80dd-"
        "jax-5b7d26a9_5b7d26a9_66588223-maxtext-df1b3598-"
        "transformer-engine-38a82f5d-x86_64"
    )

    bounded = _bounded_tag_suffix(registry, suffix)

    assert len("triage-" + bounded) == 128
    assert bounded == _bounded_tag_suffix(registry, suffix)
    assert bounded != _bounded_tag_suffix(registry, suffix + "-different")


def test_short_container_tag_suffix_is_unchanged():
    suffix = "container-12345678-jax-abcdef01-x86_64"

    assert _bounded_tag_suffix("registry.example/repo:triage-", suffix) == suffix
    assert _bounded_tag_suffix("registry.example:5005/repo", suffix) == suffix
