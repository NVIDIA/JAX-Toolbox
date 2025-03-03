import os

from absl import app, flags
from axlearn.common.launch_trainer import run_trainer
from axlearn.common.config import config_for_function
from axlearn.experiments.text.gpt import c4_trainer
from axlearn.common.trainer import SpmdTrainer

FLAGS = flags.FLAGS
FLAGS.set_default("module", "text.gpt.c4_trainer") 
FLAGS.set_default("config", "fuji-7B-v2-flash")  # Set the model 
FLAGS.set_default("trainer_dir", "/opt/host/axlearn-checkpoints")  # Set the trainer directory

def main(_):
    axlearn_path = "/opt/axlearn"  
    os.environ["PYTHONPATH"] = f"{axlearn_path}:{os.environ.get('PYTHONPATH', '')}"  

    n_gpus = 16 # This can be also an env variable
    # Base XLA flags
    base_flags = [
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_enable_command_buffer=",
        "--xla_gpu_enable_highest_priority_async_stream=true",
        "--xla_gpu_all_reduce_combine_threshold_bytes=1073741824",
        "--xla_gpu_all_gather_combine_threshold_bytes=1073741824",
        "--xla_gpu_reduce_scatter_combine_threshold_bytes=1073741824",
        "--xla_gpu_enable_pipelined_all_gather=true",
        "--xla_gpu_enable_pipelined_reduce_scatter=true",
        "--xla_gpu_enable_pipelined_all_reduce=true",
        "--xla_gpu_enable_while_loop_double_buffering=true",
        "--xla_gpu_enable_triton_gemm=false",
        "--xla_gpu_enable_all_gather_combine_by_dim=false",
        "--xla_gpu_enable_reduce_scatter_combine_by_dim=false",
        "--xla_disable_hlo_passes=rematerialization",
    ]
    # Get existing flags from environment with proper fallback.
    existing_xla_flags = os.environ.get("XLA_FLAGS", "").split()
    # XLA flags
    os.environ.update({
        "XLA_FLAGS": " ".join([
            *base_flags,
            *existing_xla_flags
        ])})

    os.environ.update({
        "DATA_DIR":"gs://axlearn-public/tensorflow_datasets", # Set up your input dataset
        "NUM_PROCESSES":f"{n_gpus}", 
        "DISTRIBUTED_COORDINATOR":"127.0.0.1:8080", 
        "PROCESS_ID":"0",
    })

    # Raw config
    config_fn = c4_trainer.named_trainer_configs()[FLAGS.config]
    trainer_config: SpmdTrainer.Config = config_for_function(config_fn).fn()

    trainer_config.max_step = 100 # Set the max number of steps to run
    trainer_config.dir = "/opt/host/axlearn-checkpoints"  # Use 'dir' instead of 'model_dir'
    trainer_config.input.input_dispatcher.global_logical_batch_size = 8 # Tune the batch size for training
    #trainer_config.input.source.max_sequence_length = 2048 # Tune the max sequence length if running in OOM
    trainer_config.checkpointer.save_policy.n = 500  # Save every 500 steps
    trainer_config.checkpointer.keep_every_n_steps = 500  # Keep checkpoints
    trainer_config.summary_writer.write_every_n_steps = 100  # Log every 100 steps

    run_trainer(
        trainer_config=trainer_config,
    )


if __name__ == "__main__":
    from absl import app
    app.run(main)
