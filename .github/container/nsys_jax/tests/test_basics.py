import os
import subprocess
import sys
import tempfile
import zipfile

helper_dir = os.path.join(os.path.dirname(__file__), "nsys_jax_test_helpers")
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)
from nsys_jax_test_helpers import nsys_jax  # noqa: E402


def test_program_without_gpu_activity():
    """
    Profiling a program that doesn't do anything should succeed.
    """
    nsys_jax([sys.executable, "-c", "print('Hello world!')"])


def test_stacktrace_entry_with_file():
    """
    Test that if a source file appears in the traceback of a JITed JAX function then
    the source file is bundled into the nsys-jax output archive.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = f"{tmpdir}/out.zip"
        src_file = f"{tmpdir}/test.py"
        assert os.path.isabs(src_file), src_file
        src_code = "import jax\njax.jit(lambda x: x*2)(4)\n"
        with open(src_file, "w") as f:
            f.write(src_code)
        subprocess.run(
            ["nsys-jax", "--output", archive, sys.executable, src_file], check=True
        )
        with zipfile.ZipFile(archive) as ifile:
            src_file_in_archive = f"sources{src_file}"
            assert src_file_in_archive in ifile.namelist()
            with ifile.open(src_file_in_archive, "r") as archived_file:
                assert archived_file.read().decode() == src_code


def test_stacktrace_entry_without_file():
    """
    Test that tracing code that does not come from a named file works (bug 4931958).
    """
    archive = nsys_jax(["python", "-c", "import jax; jax.jit(lambda x: x*2)(4)"])
    with zipfile.ZipFile(archive.name) as ifile:
        # The combination of -c and JAX suppressing references to its own source code
        # should mean that no source code files are gathered.
        assert not any(x.startswith("sources/") for x in ifile.namelist())
