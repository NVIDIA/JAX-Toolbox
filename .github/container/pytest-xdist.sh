#!/bin/bash
# Launcher script using pytest-xdist to parallelise test execution across multiple GPUs
# on the same machine. This should be combined with a higher-level decomposition of the
# tests into single/multi-GPU cases. For example:
#
# nvidia-cuda-mps-control -d
# pytest-xdist.sh 1 10 1gpu-logfile.jsonl tests/single_gpu
# pytest-xdist.sh 2 5 2gpu-logfile.jsonl tests/two_gpu
set -e
# How many GPUs will be visible to each worker process.
GPUS_PER_WORKER=$1
shift
# How many workers will submit work to each GPU. If this is more than 1 then MPS should
# be used to allow fine-grained interleaving of work submitted from different processes
WORKERS_PER_GPU=$1
shift
# pytest-reportlog logfile
LOGFILE=$1
shift
# How many GPUs are attached to the system
GPU_COUNT=$(nvidia-smi -L | wc -l)
# Create a FIFO that multiple child pytest invocations (nested inside $@) can write to
FIFO=$(mktemp -u)
PYTEST_ADDOPTS+=" --report-log=${FIFO} --dist=load"
for worker in $(seq 0 $((WORKERS_PER_GPU-1))); do
  for first_device in $(seq 0 $GPUS_PER_WORKER $((GPU_COUNT-1))); do
    last_device=$((first_device+GPUS_PER_WORKER-1))
    devices=$(seq -s , $first_device $last_device)
    device_range="${first_device}"
    if [[ $first_device != $last_device ]]; then
      device_range+="-${last_device}"
    fi
    PYTEST_ADDOPTS+=" --tx popen//id=gpu${device_range}_worker${worker}//env:CUDA_VISIBLE_DEVICES=${devices}"
  done
done
export PYTEST_ADDOPTS
export XLA_PYTHON_CLIENT_MEM_FRACTION=$(printf '0.%02d' $((75/WORKERS_PER_GPU)))
mkfifo ${FIFO}
# Handle errors explicitly below here
set +e
# Open input RW so cat goes not receive EOF when the first child pytest invocation exits
cat 0<> "${FIFO}" > "${LOGFILE}" &
cat_pid=$!
"$@"
exit_code=$?
# Cleanup
rm "${FIFO}"
kill $cat_pid
wait
exit $exit_code
