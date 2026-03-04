#!/bin/bash
set -x

ici_DP=1
dcn_DP=1
ici_FSDP=4
dcn_FSDP=1

HLO_DUMP_PATH="xla_dump"
BASE_THRESHOLD=8589934592 # 8 GB
RS_MULTIPLE=4
AR_MULTIPLE=4
AG_MULTIPLE=4

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export NVTE_FUSED_ATTN=1

export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
                --xla_gpu_enable_triton_gemm=false
                --xla_gpu_enable_command_buffer=
                --xla_gpu_all_reduce_combine_threshold_bytes=$((BASE_THRESHOLD/AR_MULTIPLE))
                --xla_gpu_all_gather_combine_threshold_bytes=$((BASE_THRESHOLD/AG_MULTIPLE))
                --xla_gpu_reduce_scatter_combine_threshold_bytes=$((BASE_THRESHOLD/RS_MULTIPLE))
                --xla_gpu_enable_pipelined_all_gather=true
                --xla_gpu_enable_pipelined_reduce_scatter=true
                --xla_gpu_enable_pipelined_all_reduce=true
                --xla_gpu_enable_while_loop_double_buffering=false
                --xla_gpu_enable_all_gather_combine_by_dim=false
                --xla_gpu_enable_reduce_scatter_combine_by_dim=false
                --xla_dump_hlo_as_text
                --xla_dump_to=$HLO_DUMP_PATH
                --xla_disable_hlo_passes=rematerialization"

echo "XLA_FLAGS = ${XLA_FLAGS}"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION = ${XLA_PYTHON_CLIENT_MEM_FRACTION}"

RUN_SETTINGS="-m MaxText.train maxtext/configs/base.yml run_name=debug_run base_output_directory=./debug_logs hardware=gpu dataset_type=synthetic  model_name=llama3-8b remat_policy='minimal_with_context_and_quantization' scan_layers=False attention='cudnn_flash_te' steps=50 dtype=bfloat16 max_target_length=8192 per_device_batch_size=4 ici_data_parallelism=${ici_DP} dcn_data_parallelism=${dcn_DP} ici_fsdp_parallelism=${ici_FSDP} dcn_fsdp_parallelism=${dcn_FSDP} profiler=nsys enable_checkpointing=false override_model_config=True gradient_accumulation_steps=1 quantization=te_nvfp4_no_rht max_segments_per_seq=32"

python3 ${RUN_SETTINGS}

set +x