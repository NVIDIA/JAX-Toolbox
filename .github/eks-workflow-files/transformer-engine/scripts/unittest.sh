          set -e

          apt update
          apt install -y tmux

          pip install pytest-reportlog pytest-xdist
          # Start MPS daemon
          nvidia-cuda-mps-control -d
          # TE's default is slightly different, without the hyphen
          export TE_PATH=${SRC_PATH_TRANSFORMER_ENGINE}

          JAX_UNITTEST_CMD="pytest-xdist.sh --START_GPU_IDX=0 --END_GPU_IDX=3 1 3 ${LOG_DIR}/pytest-report-L0-unittest.jsonl bash $TE_PATH/qa/L0_jax_unittest/test.sh  | tee -a ${LOG_DIR}/unittest_pytest_stdout.log"
          JAX_DISTRIBUTED_UNITTEST_CMD="pytest-xdist.sh --START_GPU_IDX=4 --END_GPU_IDX=7 4 1 ${LOG_DIR}/pytest-report-L0-distributed-unittest.jsonl bash $TE_PATH/qa/L0_jax_distributed_unittest/test.sh"

          tmux_run() {
            local session_name=$1
            shift
            local cmd="$@"
            tmux new-session -d -s $session_name
            tmux send-keys -t $session_name "export TE_PATH=${SRC_PATH_TRANSFORMER_ENGINE}; $cmd; touch ${LOG_DIR}/$session_name.done" C-m
          }
          
          tmux_run unittest $JAX_UNITTEST_CMD
          tmux_run distributed-unittest $JAX_DISTRIBUTED_UNITTEST_CMD

          while true; do
            if  [[ -f ${LOG_DIR}/unittest.done && -f ${LOG_DIR}/distributed-unittest.done ]]; then
              echo "TransformerEngine unittest and distributed unittest both completed"
              break
            elif  [[ -f ${LOG_DIR}/unittest.done && ! -f ${LOG_DIR}/distributed-unittest.done ]]; then
              echo "TransformerEngine unittest done, waiting for distributed unittest to complete"
            elif  [[ ! -f ${LOG_DIR}/unittest.done && -f ${LOG_DIR}/distributed-unittest.done ]]; then
              echo "TransformerEngine distributed unittest done, waiting for unittest to complete"
            else
              echo "Waiting for TransformerEngine unittest and distributed unittest to complete"
            fi
            sleep 10
          done

          # merge the log files
          cat \
            ${LOG_DIR}/pytest-report-L0-unittest.jsonl
            ${LOG_DIR}/pytest-report-L0-distributed-unittest.jsonl
            > ${LOG_DIR}/pytest-report.jsonl

          cat ${LOG_DIR}/pytest-jsonl
          touch ${LOG_DIR}/done
