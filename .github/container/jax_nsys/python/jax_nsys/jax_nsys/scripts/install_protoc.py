import argparse
import google.protobuf
import io
import os
import platform
import requests
import zipfile

def main():
    # TODO: add a default to (with confirmation) install in the same prefix as this script is installed to
    parser = argparse.ArgumentParser(
        "Install a version of the protoc compiler that is compatible with the google.protobuf runtime"
    )
    parser.add_argument(
        "prefix", help="Output prefix under which to install protoc", type=str
    )
    args = parser.parse_args()

    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=5))

    # protobuf versioning is complicated, see protocolbuffers/protobuf#11123 for more
    # discussion. For older versions, when the versioning scheme was aligned, try and
    # install a protoc with the same version as google.protobuf. For newer versions, given
    # google.protobuf version X.Y.Z install protoc version Y.Z as described in
    # https://protobuf.dev/support/version-support
    runtime_version = tuple(map(int, google.protobuf.__version__.split(".")))
    if runtime_version < (3, 21):
        # old versioning scheme, try and install a matching protoc version
        protoc_version = runtime_version
    else:
        # new versioning scheme, runtime minor.patch should be the protoc version
        protoc_version = runtime_version[1:]

    # Install the given protobuf version
    ver = ".".join(map(str, protoc_version))
    system = platform.system().lower()
    machine = platform.machine()
    system = {"darwin": "osx"}.get(system, system)
    machine = {
        "aarch64": "aarch_64",
        "arm64": "aarch_64",
    }.get(machine, machine)
    # Apple Silicon can handle universal and x86_64 if it needs to.
    machines = {
        ("osx", "aarch_64"): ["aarch_64", "universal_binary", "x86_64"],
    }.get((system, machine), [machine])
    for machine in machines:
        r = s.get(
            f"https://github.com/protocolbuffers/protobuf/releases/download/v{ver}/protoc-{ver}-{system}-{machine}.zip"
        )
        if r.status_code == 404:
            # assume this means the architecture is not available
            continue
    else:
        r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in z.namelist():
            if ".." in name:
                continue
            if name.startswith("bin/") or name.startswith("include/"):
                z.extract(name, path=args.prefix)

    # Make sure the protoc binary is executable
    os.chmod(os.path.join(args.prefix, "bin", "protoc"), 0o755)
