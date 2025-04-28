from nsys_jax import ensure_compiled_protos_are_importable
import os
import pathlib
import subprocess
import tempfile
import zipfile


def nsys_jax_with_result(command):
    """
    Helper to run nsys-jax with a unique output file that will be automatically
    cleaned up on destruction. Explicitly returns the `subprocess.CompletedProcess`
    instance.
    """
    output = tempfile.NamedTemporaryFile(suffix=".zip")
    result = subprocess.run(
        ["nsys-jax", "--force-overwrite", "--output", output.name] + command,
    )
    return output, result


def nsys_jax(command):
    """
    Helper to run nsys-jax with a unique output file that will be automatically
    cleaned up on destruction. Throws if running `nsys-jax` does not succeed.
    """
    output, result = nsys_jax_with_result(command)
    result.check_returncode()
    return output


def nsys_jax_archive(command):
    """
    Helper to run nsys-jax and automatically extract the output, yielding a directory.
    """
    tmpdir = tempfile.TemporaryDirectory()
    archive = nsys_jax(command)
    old_dir = os.getcwd()
    os.chdir(tmpdir.name)
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall()
    finally:
        os.chdir(old_dir)
    # Make sure the protobuf bindings can be imported, the generated .py will go into
    # a temporary directory that is not currently cleaned up. The bindings cannot be
    # un-imported from the test process, so there is a tacit assumption that in a given
    # test session there will only be one set of .proto files and it doesn't matter
    # which ones are picked up.
    ensure_compiled_protos_are_importable(prefix=pathlib.Path(tmpdir.name))
    return tmpdir
