import argparse
import json
import pathlib

AA_MOTIF = "AGLYGLYGAFACFADFACDA"


def make_sequence(length: int) -> str:
    reps = (length + len(AA_MOTIF) - 1) // len(AA_MOTIF)
    return (AA_MOTIF * reps)[:length]


def make_input(length: int, seed: int) -> dict:
    return {
        "name": f"bench_L{length:05d}",
        "modelSeeds": [seed],
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": make_sequence(length),
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates": [],
                }
            }
        ],
        "dialect": "alphafold3",
        "version": 4,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lengths", type=int, nargs="+", required=True)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for length in args.lengths:
        payload = make_input(length, args.seed)
        path = out_dir / f"bench_L{length:05d}.json"
        path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
