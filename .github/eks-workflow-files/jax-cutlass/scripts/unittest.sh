          set -xu -o pipefail

          LOG_DIR=/opt/output

          pip install pytest-reportlog pytest-xdist

          # Install here as we don't want to install into the container directly yet.
          pip install git+https://github.com/NVIDIA/JAX-Toolbox.git@${JAX_TOOLBOX_REF}#subdirectory=.github/container/cutlass_dsl_jax

          # Start MPS daemon
          nvidia-cuda-mps-control -d
          
          # 1 GPU per worker, 4 workers per GPU
          pytest-xdist.sh 1 4 ${LOG_DIR}/pytest-report-L0-unittest.jsonl ${SRC_PATH_JAX_CUTLASS}/tests/ | tee -a ${LOG_DIR}/pytest_stdout.log
          
          # 8 GPUs per worker, 1 worker per GPU. pytest-xdist.sh allows aggregation
          # into a single .jsonl file of results from multiple pytest invocations
          # inside the test.sh script, so it's useful even with a single worker per
          # device.
          pytest-xdist.sh 8 1 ${LOG_DIR}/pytest-report-L0-distributed-unittest.jsonl bash  ${SRC_PATH_JAX_CUTLASS}/tests/ | tee -a ${LOG_DIR}/pytest_stdout.log

          # merge the log files
          cat \
            ${LOG_DIR}/pytest-report-L0-unittest.jsonl \
            ${LOG_DIR}/pytest-report-L0-distributed-unittest.jsonl \
            > ${LOG_DIR}/pytest-report.jsonl

          touch ${LOG_DIR}/done
