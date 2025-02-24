#! /bin/bash
BASEDIR="/opt/host/"
CONFIG="fuji-1B-v3-flash"
POSTFIX=${POSTFIX:=""}


export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
                 --xla_gpu_graph_level=0
                 --xla_gpu_enable_highest_priority_async_stream=true
                 --xla_gpu_all_reduce_combine_threshold_bytes=1073741824
                 --xla_gpu_all_gather_combine_threshold_bytes=1073741824
                 --xla_gpu_reduce_scatter_combine_threshold_bytes=1073741824
                 --xla_gpu_enable_pipelined_all_gather=true
                 --xla_gpu_enable_pipelined_reduce_scatter=true
                 --xla_gpu_enable_pipelined_all_reduce=true
                 --xla_gpu_enable_while_loop_double_buffering=true
                 --xla_gpu_enable_triton_gemm=false
                 --xla_gpu_enable_all_gather_combine_by_dim=false
                 --xla_gpu_enable_reduce_scatter_combine_by_dim=false
                 --xla_disable_hlo_passes=rematerialization"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export NCCL_BUFFSIZE=8388608 
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_LAUNCH_MODE=GROUP

LOG_DIF=${BASEDIR}/logs
TRAINER_DIR=${LOG_DIF}/${CONFIG}_N${SLURM_JOB_NUM_NODES}_n${SLURM_NTASKS}/trainer-logs
mkdir -p ${TRAINER_DIR}

#test "${WITH_MP}" == 1 && export MP_ARGS="--num_processes=${SLURM_NTASKS} --distributed_coordinator=${SLURM_LAUNCH_NODE_IPADDR}:12345 --process_id=${SLURM_PROCID}"

python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer \
    --config=${CONFIG} \
    --trainer_dir=${TRAINER_DIR} \
    --data_dir=gs://axlearn-public/tensorflow_datasets \
    --jax_backend=gpu 
