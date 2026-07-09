#!/usr/bin/env python
import argparse
import json
import pathlib
import subprocess
import sys

parser = argparse.ArgumentParser()
# This is the interface expected by the triage tool
parser.add_argument("--container", required=True, type=str, help="Container to test.")
parser.add_argument(
    "--output-prefix",
    required=True,
    type=pathlib.Path,
    help="Directory to download output to.",
)
# These are for use in test cases
parser.add_argument("--exit-code", type=int)
args = parser.parse_args()
result = subprocess.run(
    ["docker", "run", args.container, "cat", "/opt/jax/pass.txt"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
with open(args.output_prefix / "stdout.txt", "wb") as ofile:
    ofile.write(result.stdout)
with open(args.output_prefix / "stderr.txt", "wb") as ofile:
    ofile.write(result.stderr)
if result.returncode == 0:
    metric_value = 1.0
else:
    metric_value = 0.0
with open(args.output_prefix / "metrics.json", "w") as ofile:
    json.dump({"test_metric": metric_value}, ofile)
if args.exit_code is None:
    result.check_returncode()
else:
    sys.exit(args.exit_code)
