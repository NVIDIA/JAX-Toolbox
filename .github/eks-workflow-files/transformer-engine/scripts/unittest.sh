          LOG_DIR=/opt/output
          pip install pytest-reportlog pytest-xdist
          
          # Start MPS daemon
          nvidia-cuda-mps-control -d
          
          # TE's default is slightly different, without the hyphen
          export TE_PATH=${SRC_PATH_TRANSFORMER_ENGINE}
          
          # 1 GPU per worker, 4 workers per GPU
          pytest-xdist.sh 1 4 ${LOG_DIR}/pytest-report-L0-unittest.jsonl bash ${TE_PATH}/qa/L0_jax_unittest/test.sh | tee -a ${LOG_DIR}/pytest_stdout.log
          
          # 4 GPUs per worker, 1 worker per GPU. pytest-xdist.sh allows aggregation
          # into a single .jsonl file of results from multiple pytest invocations
          # inside the test.sh script, so it's useful even with a single worker per
          # device.
          pytest-xdist.sh 4 1 ${LOG_DIR}/pytest-report-L0-distributed-unittest.jsonl bash ${TE_PATH}/qa/L0_jax_distributed_unittest/test.sh | tee -a ${LOG_DIR}/pytest_stdout.log

          # merge the log files
          cat \
            ${LOG_DIR}/pytest-report-L0-unittest.jsonl \
            ${LOG_DIR}/pytest-report-L0-distributed-unittest.jsonl \
            > ${LOG_DIR}/pytest-report.jsonl

          touch ${LOG_DIR}/done
