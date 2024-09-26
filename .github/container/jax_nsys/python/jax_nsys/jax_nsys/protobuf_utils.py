# WARNING: it is tacitly assumed that the protobuf compiler (protoc) is
# compatible with the google.protobuf version.
import glob
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Optional


def which(executable: str) -> pathlib.Path:
    """
    Wrap shutil.which, making sure that if we are inside a virtual environment
    then the bin/ directory is searched first.
    """
    # If we are running in a virtual environment, make sure the bin/ directory
    # of it is in the PATH.
    path = os.environ.get("PATH", os.defpath).split(":")
    if sys.prefix != sys.base_prefix:
        # Running in a virtual environment
        venv_bin = os.path.join(sys.prefix, "bin")
        if venv_bin not in path:
            path.insert(0, venv_bin)
    exe = shutil.which(executable, path=":".join(path))
    if exe is None:
        raise Exception(f"Did not find {executable} in PATH")
    return pathlib.Path(exe)


def compile_protos(
    proto_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    output_stub_dir: Optional[str | pathlib.Path] = None,
):
    if not os.path.isdir(proto_dir):
        raise Exception(f"Input: {proto_dir} is not a directory")
    if not os.path.isdir(output_dir):
        raise Exception(f"Output: {output_dir} is not a directory")
    # Find the .proto files
    proto_files = glob.glob("**/*.proto", recursive=True, root_dir=proto_dir)
    if len(proto_files) == 0:
        raise Exception(f"Did not find any .proto files under {proto_dir}")
    protoc = which("protoc")
    # Generate code to load the protobuf files
    args: list[str | pathlib.Path] = [
        protoc,
        f"-I={proto_dir}",
        f"--python_out={output_dir}",
    ]
    if output_stub_dir is not None:
        args.append(f"--pyi_out={output_stub_dir}")
    args += proto_files
    subprocess.run(args, check=True)


def ensure_compiled_protos_are_importable(*, prefix: pathlib.Path = pathlib.Path(".")):
    """
    See if the Python bindings generated from .proto are importable, and if not then
    generate them in a temporary directory and prepend it to sys.path.
    """

    def do_import():
        # Use this as a proxy for everything being importable
        from xla.service import hlo_pb2  # noqa: F401

    try:
        do_import()
        return
    except ImportError:
        tmp_dir = tempfile.mkdtemp()
        compile_protos(prefix / "protos", tmp_dir)
        sys.path.insert(0, tmp_dir)
        do_import()
        return tmp_dir
