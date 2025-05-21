          set -x

          apt update
          apt install -y tmux
          mkdir -p /opt/output
          mkdir -p /log

          pip install pytest-reportlog pytest-xdist
          # Start MPS daemon
          nvidia-cuda-mps-control -d
          # TE's default is slightly different, without the hyphen
          export TE_PATH=${SRC_PATH_TRANSFORMER_ENGINE}

          JAX_UNITTEST_CMD="pytest-xdist.sh --START_GPU_IDX=0 --END_GPU_IDX=3 1 3 /log/pytest-report-L0-unittest.jsonl bash $TE_PATH/qa/L0_jax_unittest/test.sh  | tee -a ${LOG_DIR}/unittest_pytest_stdout.log"
          JAX_DISTRIBUTED_UNITTEST_CMD="pytest-xdist.sh --START_GPU_IDX=4 --END_GPU_IDX=7 4 1 /log/pytest-report-L0-distributed-unittest.jsonl bash $TE_PATH/qa/L0_jax_distributed_unittest/test.sh"

          tmux_run() {
            local session_name=$1
            shift
            local cmd="$@"
            tmux new-session -d -s $session_name
            tmux send-keys -t $session_name "export TE_PATH=${SRC_PATH_TRANSFORMER_ENGINE}; $cmd; touch /opt/output/$session_name.done" C-m
          }
          
          tmux_run unittest $JAX_UNITTEST_CMD
          tmux_run distributed-unittest $JAX_DISTRIBUTED_UNITTEST_CMD

          while true; do
            if  [[ -f /opt/output/unittest.done && -f /opt/output/distributed-unittest.done ]]; then
              echo "TransformerEngine unittest and distributed unittest both completed"
              break
            elif  [[ -f /opt/output/unittest.done && ! -f /opt/output/distributed-unittest.done ]]; then
              echo "TransformerEngine unittest done, waiting for distributed unittest to complete"
            elif  [[ ! -f /opt/output/unittest.done && -f /opt/output/distributed-unittest.done ]]; then
              echo "TransformerEngine unittest done, waiting for distributed unittest to complete"
            else
              echo "Waiting for TransformerEngine unittest and distributed unittest to complete"
            fi
            sleep 5
          done

          # merge the log files
          cat \
            /log/pytest-report-L0-unittest.jsonl
            /log/pytest-report-L0-distributed-unittest.jsonl
            > /log/pytest-report.jsonl

          touch ${LOG_DIR}/done
