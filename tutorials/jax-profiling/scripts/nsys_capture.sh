#!/usr/bin/env bash
# Capture a steady-state Nsight Systems timeline of a python target.
#
# The target must call cudaProfilerStart()/Stop() around its steady loop (compile
# + warm-up happen OUTSIDE that range), so only steady-state work is profiled.
#
# Usage:  scripts/nsys_capture.sh [TARGET_PY] [OUT] [GRAPH] [TRACE]
# Examples:
#   scripts/nsys_capture.sh                                            # defaults below
#   scripts/nsys_capture.sh profile_artifacts/target.py profile_artifacts/step_nodes node cuda,nvtx
set -euo pipefail

TARGET="${1:-profile_artifacts/target.py}"   # python script to profile (calls cudaProfilerStart/Stop)
OUT="${2:-profile_artifacts/step}"           # report path ('.nsys-rep' is appended)
GRAPH="${3:-graph}"                          # CUDA-graph tracing: 'graph' (low overhead) or 'node' (individual graph nodes, more overhead)
TRACE="${4:-cuda,nvtx,osrt}"                 # APIs to trace

nsys profile --force-overwrite true -o "${OUT}" \
    --trace="${TRACE}" \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace="${GRAPH}" \
    python "${TARGET}"

echo "Report: ${OUT}.nsys-rep"
echo "  rank kernels: nsys stats --force-export=true --report cuda_gpu_kern_sum ${OUT}.nsys-rep"
