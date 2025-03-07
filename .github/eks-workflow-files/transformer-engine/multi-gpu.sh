        nvidia-smi
        pip install pytest \
              pytest-reportlog \
              cuda-python \
              -r ${SRC_PATH_TRANSFORMER_ENGINE}/examples/jax/encoder/requirements.txt
        pip install pytest-reportlog pytest-xdist
        
        set -ex
        cd ${SRC_PATH_TRANSFORMER_ENGINE}/examples/jax/encoder
        pytest --report-log=/output/pytest-report.jsonl \
        test_single_gpu_encoder.py \
        test_multigpu_encoder.py \
        test_model_parallel_encoder.py | tee -a ${LOG_DIR}/tests.log
        
        touch /opt/output/done
