# WARNING: it is tacitly assumed that the protobuf compiler (protoc) is
# compatible with the google.protobuf version.
import glob
import google.protobuf
import os
import pathlib
import shutil
import subprocess
import sys


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


def compile_protos(proto_dir: str | pathlib.Path, output_dir: str | pathlib.Path):
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
    args: list[str | pathlib.Path] = [protoc, f"-I={proto_dir}", f"--python_out={output_dir}"]
    args += proto_files
    subprocess.run(args, check=True)
