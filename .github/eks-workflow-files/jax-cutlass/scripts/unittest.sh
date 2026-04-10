         set -xu -o pipefail

         LOG_DIR=${LOG_DIR:-/opt/output}
         mkdir -p ${LOG_DIR}

         SRC_ROOT=${SRC_ROOT:-$PWD/jax-cutlass-src}
         SRC_ROOT=$(realpath $SRC_ROOT)
         pip install pytest-reportlog pytest-xdist flatbuffers
          
         # Clone CUTLASS examples (use branch)
         # TODO: Support dev release which likely use main branch
         CUTLASS_ROOT="${SRC_ROOT}/cutlass"
         CUTLASS_EXAMPLES_ROOT="${CUTLASS_ROOT}/examples/python/CuTeDSL"
         [[ -d ${CUTLASS_ROOT} ]] && rm -rf ${CUTLASS_ROOT}
         CUTEDSL_VERSION=$(python -c 'import cutlass; print(cutlass.__version__);')
         git clone -b v${CUTEDSL_VERSION} --single-branch --depth 1 https://github.com/NVIDIA/cutlass.git ${CUTLASS_ROOT}

         NGPUS=$(nvidia-smi --list-gpus | wc -l)

         # Start MPS daemon
         nvidia-cuda-mps-control -d

         # Run the examples
         for f in ${CUTLASS_ROOT}/examples/python/CuTeDSL/jax/*.py; do
             echo "[Executing] $f"
             log_output=$(python $f 2>&1)
             exit_code=$?
             outcome=$( [ $exit_code -eq 0 ] && echo "passed" || echo "failed" )
             echo "=== ${f} ===" | tee -a ${LOG_DIR}/pytest_stdout.log
             echo "${log_output}" | tee -a ${LOG_DIR}/pytest_stdout.log
             python3 -c "import json,sys; print(json.dumps({'outcome': sys.argv[1], 'nodeid': sys.argv[2], 'longrepr': sys.argv[3]}))" \
                 "${outcome}" "${f}" "${log_output}" >> ${LOG_DIR}/pytest-report.jsonl
         done

         touch ${LOG_DIR}/done
