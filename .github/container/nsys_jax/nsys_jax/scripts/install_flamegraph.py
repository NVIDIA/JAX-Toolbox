import argparse
import os
import requests


def main():
    """
    Ideally flamegraph.pl could just be declared as a dependency in pyproject.toml, but
    it's not packaged for that. It could probably be worked around, but for now we just
    distribute this script to install it.
    """
    # TODO: add a default to (with confirmation) install in the same prefix as this script is installed to
    parser = argparse.ArgumentParser("Fetch the flamegraph.pl script")
    parser.add_argument(
        "prefix", help="Output prefix under which to install protoc", type=str
    )
    args = parser.parse_args()
    install_dir = os.path.join(args.prefix, "bin")
    install_path = os.path.join(install_dir, "flamegraph.pl")
    assert not os.path.exists(install_path), f"{install_path} already exists"
    os.makedirs(install_dir, exist_ok=True)

    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=5))
    r = s.get(
        "https://raw.githubusercontent.com/brendangregg/FlameGraph/master/flamegraph.pl"
    )
    r.raise_for_status()
    with open(install_path, "w") as ofile:
        ofile.write(r.text)
    os.chmod(install_path, 0o755)
