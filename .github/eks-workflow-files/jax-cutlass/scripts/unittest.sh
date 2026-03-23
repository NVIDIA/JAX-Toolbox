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

          # Run the examples; if they complete then everything is working.
		  for f in ${CUTLASS_EXAMPLES_ROOT}/jax/*.py; do
		    echo "[Executing] $f"
		  	python $f
		  done

          touch ${LOG_DIR}/done
