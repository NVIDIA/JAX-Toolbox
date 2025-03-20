import sys
import argparse

from absl import app, flags
from axlearn.common.config import config_for_function
from axlearn.experiments.text.gpt import c4_trainer
from axlearn.common.trainer import SpmdTrainer
from axlearn.common import launch, launch_trainer, measurement

from axlearn.common.trainer_config_modifier import (
    ChainConfigModifier,
    GradientAccumulationModifier,
    MeshShapeModifier,
)

from axlearn.experiments.text.gpt.common import mesh_shape_from_axes

from axlearn.common.utils import (
    HybridMeshShape,
)

FLAGS = flags.FLAGS

parser = argparse.ArgumentParser(description="Custom parallelism args")
parser.add_argument(
    "--ici_pp", type=int, default=1, help="Intranode pipeline parallelism size."
)

parser.add_argument(
    "--ici_fsdp",
    type=int,
    default=1,
    help="Intranode FSDP (fully sharded data parallel) size.",
)

parser.add_argument(
    "--ici_sqp", type=int, default=1, help="Intranode sequence parallelism size."
)

parser.add_argument(
    "--ici_dp", type=int, default=1, help="Intranode data parallelism size"
)

parser.add_argument(
    "--dcn_pp", type=int, default=1, help="Internode pipeline parallelism size."
)

parser.add_argument(
    "--dcn_fsdp",
    type=int,
    default=1,
    help="Interanode FSDP (fully sharded data parallel) size.",
)

parser.add_argument(
    "--dcn_sqp", type=int, default=1, help="Internode sequence parallelism size."
)

parser.add_argument(
    "--dcn_dp", type=int, default=1, help="Internode data parallelism size"
)

parser.add_argument("--ga", type=int, default=1, help="Gradient accumulation size.")


parser.add_argument("--gbs", type=int, default=1, help="Global batch size per GPU")

parser.add_argument("--seq_len", type=int, default=4096, help="Max sequence length.")

parser.add_argument(
    "--max_step", type=int, default=100, help="Max number of steps to execute."
)

parser.add_argument(
    "--save_checkpoint_steps",
    type=int,
    default=500,
    help="Save checkpoints every X steps.",
)

parser.add_argument(
    "--write_summary_steps", type=int, default=500, help="Save summary every X steps."
)


def run_main():
    """Run main function

    This function parses the input args and runs the main code.
    """
    # retrieve the parse args, while absl flags are recognized
    parsed_args, other_argv = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + other_argv
    app.run(lambda _: main(parsed_args))


def main(parsed_args):
    """Main function
    This function runs the Fuji model defined in the config and set custom parameters.

    Args:
        parsed_args:  argparse.Namespace object, input parsed arguments
    """
    # Retrieve absl flags
    config_name = FLAGS.config
    trainer_dir = FLAGS.trainer_dir
    # Parsed args
    ici_pp_size = parsed_args.ici_pp
    ici_fsdp_size = parsed_args.ici_fsdp
    ici_sqp_size = parsed_args.ici_sqp
    ici_dp_size = parsed_args.ici_dp
    dcn_pp_size = parsed_args.dcn_pp
    dcn_fsdp_size = parsed_args.dcn_fsdp
    dcn_sqp_size = parsed_args.dcn_sqp
    dcn_dp_size = parsed_args.dcn_dp
    ga_size = parsed_args.ga
    gbs_size = parsed_args.gbs
    seq_len = parsed_args.seq_len
    max_step = parsed_args.max_step
    save_checkpoint_steps = parsed_args.save_checkpoint_steps
    write_summary_steps = parsed_args.write_summary_steps

    print(
        f"=== Parameter Check ===\n"
        f"ici_pp_size: {ici_pp_size}\n"
        f"ici_fsdp_size: {ici_fsdp_size}\n"
        f"ici_sqp_size: {ici_sqp_size}\n"
        f"ici_dp_size: {ici_dp_size}\n"
        f"dcn_pp_size: {dcn_pp_size}\n"
        f"dcn_fsdp_size: {dcn_fsdp_size}\n"
        f"dcn_sqp_size: {dcn_sqp_size}\n"
        f"dcn_dp_size: {dcn_dp_size}\n"
        f"ga_size: {ga_size}\n"
        f"seq_len: {seq_len}\n"
        f"max_step: {max_step}\n"
        f"save_checkpoint_steps: {save_checkpoint_steps}\n"
        f"write_summary_steps: {write_summary_steps}\n"
        f"global_batch_size: {gbs_size}\n"
        f"======================\n"
    )

    # Build the model config
    config_fn = c4_trainer.named_trainer_configs()[config_name]
    trainer_config: SpmdTrainer.Config = config_for_function(config_fn).fn()
    # Intra-node parallelism --> multinode, fsdp = num of gpus in single node
    ici_mesh_shape = mesh_shape_from_axes(
        pipeline=ici_pp_size, data=ici_dp_size, fsdp=ici_fsdp_size, seq=ici_sqp_size
    )
    # Inter-node parallelism --> multinode, dp_size = numb of nodes
    dcn_mesh_shape = mesh_shape_from_axes(
        pipeline=dcn_pp_size, data=dcn_dp_size, fsdp=dcn_fsdp_size, seq=dcn_sqp_size
    )
    # Create a mesh
    mesh_shape = HybridMeshShape(
        ici_mesh_shape=ici_mesh_shape, dcn_mesh_shape=dcn_mesh_shape
    )
    # GA & FSDP setup
    mesh_rule = (
        "custom",
        ChainConfigModifier.default_config().set(
            config_modifiers=[
                GradientAccumulationModifier.default_config().set(
                    grad_acc_steps=ga_size
                ),
                MeshShapeModifier.default_config().set(mesh_shape=mesh_shape),
            ]
        ),
    )
    trainer_config.mesh_rules = mesh_rule
    trainer_config.mesh_shape = mesh_shape
    # Max step
    trainer_config.max_step = max_step
    # Checkpoint directory
    trainer_config.dir = trainer_dir
    trainer_config.input.input_dispatcher.global_logical_batch_size = gbs_size
    trainer_config.input.source.max_sequence_length = seq_len
    trainer_config.checkpointer.save_policy.n = save_checkpoint_steps
    trainer_config.checkpointer.keep_every_n_steps = save_checkpoint_steps
    trainer_config.summary_writer.write_every_n_steps = write_summary_steps
    # Call AXLearn Jax setup
    launch.setup()
    # Setup the config
    trainer_config.set(
        recorder=config_for_function(lambda: measurement.global_recorder)
    )
    # Launch training
    launch_trainer.run_trainer(
        trainer_config=trainer_config,
    )


if __name__ == "__main__":
    run_main()
