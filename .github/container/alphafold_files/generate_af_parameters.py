import argparse
import pathlib
import re
import numpy as np
import zstandard
import ml_dtypes
from alphafold3.model import params


LINE_RE = re.compile(r"^\s*name=(.*?)\s+dtype=([A-Za-z0-9_]+)\s+shape=\((.*?)\)\s*$")


def parse_dtype(dtype_name: str):
    return ml_dtypes.bfloat16 if dtype_name == "bfloat16" else np.dtype(dtype_name)


def parse_shape(shape_text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in shape_text.split(",") if p.strip())


def iter_schema(md_path: pathlib.Path):
    """Parses the model_parameters.md file and yields (full_name, scope, name, dtype, shape) for each parameter."""
    for line in md_path.read_text().splitlines():
        m = LINE_RE.match(line.rstrip())
        full_name, dtype_name, shape_text = m.groups()
        # The AF3 doc schema stores "scope:name".
        scope, name = full_name.rsplit(":", 1)
        yield full_name, scope, name, parse_dtype(dtype_name), parse_shape(shape_text)


def make_array(full_name: str, shape: tuple[int, ...], dtype, rng: np.random.Generator):
    if full_name == "__meta__:__identifier__":
        return np.zeros(shape=shape, dtype=dtype)

    # This mirrors the public doc guidance: avoid all-zero tensors.
    vals = rng.uniform(low=-1.0, high=1.0, size=shape if shape else None)
    return np.asarray(vals, dtype=dtype)


def main():
    """Main entry point for generating random AF3 weights.
    The output is a single zstd-compressed file containing the encoded parameter."""
    ap = argparse.ArgumentParser()
    # keep the `--repo` in case we want to test this locally
    ap.add_argument(
        "--repo", required=True, help="Path to local alphafold3 repo checkout"
    )
    ap.add_argument(
        "--out-model-dir",
        required=True,
        help="Directory that will contain exactly one model file",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo = pathlib.Path(args.repo)
    model_params_md = repo / "model_parameters.md"
    out_model_dir = pathlib.Path(args.out_model_dir)
    out_model_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_model_dir / "random_weights.bin.zst"

    rng = np.random.default_rng(args.seed)

    with zstandard.open(out_path, "wb") as f:
        for full_name, scope, name, dtype, shape in iter_schema(model_params_md):
            arr = make_array(full_name, shape, dtype, rng)
            f.write(params.encode_record(scope, name, arr))

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
