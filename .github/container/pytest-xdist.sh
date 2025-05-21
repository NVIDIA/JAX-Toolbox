#!/bin/bash
# Launcher script using pytest-xdist to parallelise test execution across multiple GPUs
# on the same machine. This should be combined with a higher-level decomposition of the
# tests into single/multi-GPU cases. For example:
#
# nvidia-cuda-mps-control -d
# pytest-xdist.sh 1 10 1gpu-logfile.jsonl tests/single_gpu
# pytest-xdist.sh 2 5 2gpu-logfile.jsonl tests/two_gpu
set -e

# Parse named args.
while [[ "$1" == --* ]]; do
case $1 in
  --*=*)
      pair=($(echo $1 | awk -F'=' '{ print $1, $2 }'))
      key=${pair[0]#--}
      value=${pair[1]}
      declare "$key=$value"
      ;;
esac
shift
done

# Set partition of GPUs by index from START_GPU_IDX to END_GPU_IDX to use in tests
# By default, use all detected GPUs - override this by providing named arguments for GPU indexes
#
# # Use a partition of GPUs 0,1,2,3 for tests
# pytest-xdist.sh --START_GPU_IDX=0 --END_GPU_IDX=3 4 1 3gpu-logfile.jsonl bash tests/four_gpu
if [[ ! -n "$START_GPU_IDX" && -n "$END_GPU_IDX" ]] || [[ ! -z "$START_GPU_IDX" && -z "$END_GPU_IDX" ]]; then
    echo "Must set both START_GPU_IDX and END_GPU_IDX or neither"
    exit 1
fi
if [ -z ${START_GPU_IDX+x} ] && [ -z ${END_GPU_IDX+x} ] ; then
    # default, use all detected GPUs
    START_GPU_IDX=$(nvidia-smi -L | awk  '{ print $2 }' | tr -d ':' | sort | head -n 1)
    END_GPU_IDX=$(nvidia-smi -L | awk  '{ print $2 }' | tr -d ':' | sort -rn | head -n 1)
fi

# Check requested GPU partition is valid
PARTITION_GPU_IDXS=$(seq -s , $START_GPU_IDX $END_GPU_IDX)
ALL_GPU_IDXS=$(nvidia-smi -L | awk '{ print $2 }' | tr -d ':' | paste -s -d,)
if [[ ! $ALL_GPU_IDXS =~ $PARTITION_GPU_IDXS ]]; then
   echo "Requested partition GPU index sequence \"$PARTITION_GPU_IDXS\" is not valid"
   exit 1
fi

# How many GPUs are attached to the system
GPU_COUNT=$((END_GPU_IDX - START_GPU_IDX + 1))
export NUM_GPUS=$GPU_COUNT # variable used in $TE_PATH/examples/jax/encoder/run_test_multiprocessing_encoder.sh

# How many GPUs will be visible to each worker process.
GPUS_PER_WORKER=$1
shift
if [[ ! $GPUS_PER_WORKER -le $GPU_COUNT ]]; then
   echo "Requested GPUs per worker ($GPUS_PER_WORKER) is greater than GPUs available in partition ($GPU_COUNT)"
   exit 1
fi

# How many workers will submit work to each GPU. If this is more than 1 then MPS should
# be used to allow fine-grained interleaving of work submitted from different processes
WORKERS_PER_GPU=$1
shift
# pytest-reportlog logfile
LOGFILE=$1
shift
# Create a FIFO that multiple child pytest invocations (nested inside $@) can write to
FIFO=$(mktemp -u)
PYTEST_ADDOPTS+=" --report-log=${FIFO} --dist=load"
for worker in $(seq 0 $((WORKERS_PER_GPU-1))); do
  for first_device in $(seq $START_GPU_IDX $GPUS_PER_WORKER $((START_GPU_IDX + GPU_COUNT-1))); do
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
export XLA_PYTHON_CLIENT_MEM_FRACTION=$(printf '0.%02d' $((85/WORKERS_PER_GPU)))
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
