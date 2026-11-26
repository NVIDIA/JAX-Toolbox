#!/usr/bin/env bash
# Fresh-process CUDA-graphs-ON / PDL-off-vs-on experiment; see ../README.md.
set -euo pipefail
PDL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$PDL_SCRIPT_DIR/../run_experiment.py" timesfm xprof "$@"
