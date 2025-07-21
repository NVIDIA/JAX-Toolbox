          set -xu -o pipefail

          LOG_DIR=${LOG_DIR:-/opt/output}

          SRC_ROOT=${SRC_ROOT:-$PWD/jax-cutlass-src}
          SRC_ROOT=$(realpath $SRC_ROOT)

          pip install pytest-reportlog pytest-xdist

          # nvidia-cutlass-dsl-jax is not yet installed to the container by default.
          # Clone if not already present locally as indicated by SRC_ROOT
          if [[ ! -d ${SRC_ROOT} ]]; then
            git clone https://github.com/NVIDIA/JAX-Toolbox.git --branch ${JAX_TOOLBOX_REF} ${SRC_ROOT}
            PIP_SRC=${SRC_ROOT}/.github/container/cutlass_dsl_jax
          else
            PIP_SRC=${SRC_ROOT}
          fi

          pip install ${PIP_SRC}

          NGPUS=$(nvidia-smi --list-gpus | wc -l)

          # Start MPS daemon
          nvidia-cuda-mps-control -d

          pytest-xdist.sh ${NGPUS} 1 ${LOG_DIR}/pytest-report.jsonl pytest -xsv --log-file=${LOG_DIR}/pytest_log.log --log-file-level=INFO ${PIP_SRC}/tests/ | tee -a ${LOG_DIR}/pytest_stdout_dist.log

          touch ${LOG_DIR}/done
