#!/bin/bash
pip install pytest-reportlog pytest-xdist

# Start MPS daemon
nvidia-cuda-mps-control -d

# TE's default is slightly different, without the hyphen
export TE_PATH=${SRC_PATH_TRANSFORMER_ENGINE}

# 1 GPU per worker, 6 workers per GPU
pytest-xdist.sh 1 6 pytest-report-L0-unittest.jsonl bash ${TE_PATH}/qa/L0_jax_unittest/test.sh | tee -a ${LOG_DIR}/tests.log

# 8 GPUs per worker, 1 worker per GPU. pytest-xdist.sh allows aggregation
# into a single .jsonl file of results from multiple pytest invocations
# inside the test.sh script, so it's useful even with a single worker per
# device.
pytest-xdist.sh 8 1 pytest-report-L0-distributed-unittest.jsonl bash ${TE_PATH}/qa/L0_jax_distributed_unittest/test.sh | tee -a ${LOG_DIR}/tests.log
pytest-xdist.sh 8 1 pytest-report-L1-distributed-unittest.jsonl bash ${TE_PATH}/qa/L1_jax_distributed_unittest/test.sh | tee -a ${LOG_DIR}/tests.log

touch /opt/output/done
