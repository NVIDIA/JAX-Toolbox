#!/usr/bin/env python3
"""Explicit, one-time AF3 setup using the helpers shipped in the AF3 container.

Never called automatically by benchmark/profile launchers. Refuses existing
model directories to avoid replacing real or previously generated parameters.
"""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="NEW directory for random parameters")
    parser.add_argument("--schema", type=Path, default=Path("/opt/alphafold_input_data/model_parameters.md"))
    parser.add_argument("--generator", type=Path, default=Path("/usr/local/bin/generate_af_parameters.py"))
    args = parser.parse_args()
    if args.model_dir.exists():
        parser.error("Model directory already exists; choose a new directory (nothing overwritten)")
    if not args.schema.is_file() or not args.generator.is_file():
        parser.error("The AF3 container must provide the parameter schema and generator; set their paths")
    # Reserve the target before running any helper; preserve it on failure for diagnosis.
    args.model_dir.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, "-c", "from alphafold3.build_data import build_data; build_data()"], check=True)
    subprocess.run([sys.executable, str(args.generator), "--input-parameters-file", str(args.schema),
                    "--out-model-dir", str(args.model_dir), "--seed", "42"], check=True)
    print(f"Prepared random parameters in {args.model_dir}. These do not produce meaningful predictions.")


if __name__ == "__main__":
    main()
