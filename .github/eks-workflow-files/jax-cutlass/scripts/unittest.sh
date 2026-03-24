         set -xu -o pipefail

         LOG_DIR=${LOG_DIR:-/opt/output}

         SRC_ROOT=${SRC_ROOT:-$PWD/jax-cutlass-src}
         SRC_ROOT=$(realpath $SRC_ROOT)

         # Clone CUTLASS examples
         CUTLASS_ROOT="${SRC_ROOT}/cutlass"
         CUTLASS_EXAMPLES_ROOT="${CUTLASS_ROOT}/examples/python/CuTeDSL"
         git clone https://github.com/NVIDIA/cutlass.git ${CUTLASS_ROOT}

         NGPUS=$(nvidia-smi --list-gpus | wc -l)

         # Start MPS daemon
         nvidia-cuda-mps-control -d

         export PYTHONPATH=${CUTLASS_EXAMPLES_ROOT}

         # Run the examples
         for f in ${CUTLASS_ROOT}/examples/python/CuTeDSL/jax/*.py; do
             echo "[Executing] $f"
             python $f && outcome="passed" || outcome="failed"
             echo "{\"outcome\": \"${outcome}\", \"nodeid\": \"${f}\"}" >> ${LOG_DIR}/pytest-report.jsonl
         done

         touch ${LOG_DIR}/done
