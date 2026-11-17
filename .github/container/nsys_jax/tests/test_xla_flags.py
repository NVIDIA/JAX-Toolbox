import os
import stat
import subprocess

import pytest  # type: ignore


@pytest.mark.parametrize(
    "xla_flags",
    [
        "--xla_dump_hlo_as_proto",
        "--xla_dump_hlo_as_proto=true",
        "--xla_dump_hlo_as_text --xla_dump_hlo_as_proto",
    ],
)
def test_bare_boolean_xla_dump_flag(xla_flags, tmp_path):
    """
    XLA accepts a bare --flag as meaning --flag=true, and its HLO dump documentation
    uses XLA_FLAGS="--xla_dump_hlo_as_proto --xla_dump_to=...". nsys-jax should get
    as far as launching nsys with such XLA_FLAGS.

    A stand-in `nsys` records the XLA_FLAGS it was launched with and exits without
    writing a report, so nsys-jax itself is expected to fail afterwards.
    """
    marker = tmp_path / "nsys-xla-flags.txt"
    fake_nsys = tmp_path / "bin" / "nsys"
    fake_nsys.parent.mkdir()
    fake_nsys.write_text(f'#!/bin/sh\nprintf "%s" "$XLA_FLAGS" > "{marker}"\nexit 1\n')
    fake_nsys.chmod(fake_nsys.stat().st_mode | stat.S_IXUSR)
    env = dict(
        os.environ,
        PATH=f"{fake_nsys.parent}{os.pathsep}{os.environ['PATH']}",
        XLA_FLAGS=xla_flags,
    )
    subprocess.run(
        ["nsys-jax", "--force-overwrite", "--output", str(tmp_path / "out"), "true"],
        cwd=tmp_path,
        env=env,
        check=False,
    )
    assert marker.exists(), "nsys-jax failed before launching nsys"
    flags = marker.read_text().split()
    proto_flags = {"--xla_dump_hlo_as_proto", "--xla_dump_hlo_as_proto=true"}
    assert proto_flags & set(flags), flags
    assert any(f.startswith("--xla_dump_to=") for f in flags), flags
