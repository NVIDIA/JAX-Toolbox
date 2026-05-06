from __future__ import annotations
import argparse
import json
import pathlib
import statistics
import sys
import time
from typing import Any
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
import tokamax

from alphafold3.common import folding_input  # pylint: disable=import-error
from alphafold3.constants import chemical_components  # pylint: disable=import-error
from alphafold3.data import featurisation  # pylint: disable=import-error
from alphafold3.model import model as af3model  # pylint: disable=import-error
from alphafold3.model import params as af3params  # pylint: disable=import-error
from alphafold3.model.components import utils  # pylint: disable=import-error


def _load_nominal_sequence_length(json_path: pathlib.Path) -> int:
    """Load the sequence from the input file
    Args:
        json_path: path to the input JSON file containing the sequence
    Returns:
        nominal_tokens_proxy: the total sequence length of the input.
    """
    payload = json.loads(json_path.read_text())
    total = 0
    for entry in payload.get("sequences", []):
        for kind in ("protein", "rna", "dna"):
            if kind in entry and "sequence" in entry[kind]:
                total += len(entry[kind]["sequence"])
    return total


def _bucket_for_tokens(num_tokens: int, buckets: list[int]) -> int:
    """Given the number of tokens, return the bucket it falls into.
    If it exceeds all buckets, return the original number of tokens.

    Args:
        num_tokens: the total sequence length of the input
        buckets: the list of bucket sizes to compare against

    Returns:
        bucket_used: the bucket that num_tokens falls into,
            or num_tokens if it exceeds all buckets
    """
    for bucket in buckets:
        if num_tokens <= bucket:
            return bucket
    return num_tokens


def _extract_flops(cost_analysis: Any) -> float | None:
    if cost_analysis is None:
        return None
    if isinstance(cost_analysis, dict):
        for key in ("flops", "FLOPs", "floating_point_operations"):
            if key in cost_analysis:
                try:
                    return float(cost_analysis[key])
                except Exception:
                    pass
    return None


def make_model_config(
    *,
    flash_attention_implementation: tokamax.DotProductAttentionImplementation = "triton",
    num_diffusion_samples: int = 5,
    num_recycles: int = 10,
    return_embeddings: bool = False,
    return_distogram: bool = False,
) -> af3model.Model.Config:
    """Returns a model config with some defaults overridden."""
    config = af3model.Model.Config()
    config.global_config.flash_attention_implementation = flash_attention_implementation
    config.heads.diffusion.eval.num_samples = num_diffusion_samples
    config.num_recycles = num_recycles
    config.return_embeddings = return_embeddings
    config.return_distogram = return_distogram
    return config


def main() -> None:
    """Main entrypoint for running AF3 inference benchmark on a single input JSON file, and writing a summary of results to an output JSON file."""

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json-path", required=True)  # Input JSON files
    ap.add_argument("--model-dir", required=True)  # Input random model
    ap.add_argument(
        "--output-json", required=True
    )  # Output JSON file with benchmark results
    # the inputs have exactly the needed number of tokens = buckets
    ap.add_argument(
        "--num-recycles", type=int, default=10
    )  # how many times to run Evoformer on the input, default 10 to match AF3 doc examples
    ap.add_argument(
        "--num-diffusion-samples", type=int, default=1
    )  # this is the number of samples AF3 generates from teh diffusion head. For benchmarking 1 is a good starting point
    ap.add_argument(
        "--flash-attention-implementation",
        default="triton",
        choices=["triton", "cudnn", "xla"],
    )  # here we can chose what FA to use, ideally Triton > CuDNN > XLA
    ap.add_argument(
        "--warm-runs", type=int, default=3
    )  # we are going to run 3 times the main benchmark and report stats from there. There's an additional run on top to avoid compilation time impacting the bench.
    ap.add_argument("--jax-compilation-cache-dir", default=None)
    ap.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="Index of the GPU to use for benchmarking (default: 0)",
    )  # AF3 runs on process per GPU. We can run multiple different jobs on different gpu-indexes
    ap.add_argument(
        "--extract-results",
        action="store_true",
        help="Run AF3 output extraction once to recover true num_tokens",
    )
    args = ap.parse_args()
    # Tokamax uses absl.flags internally and may parse sys.argv lazily during
    # JAX lowering/compilation. Remove this script's argparse flags so absl does
    # not see unknown flags like --repo-dir or --json-path.
    sys.argv = [sys.argv[0]]

    input_json_path = pathlib.Path(args.input_json_path)
    model_dir = pathlib.Path(args.model_dir)
    output_json = pathlib.Path(args.output_json)
    # should we move this as an int argument to take, and test just as part of JET with multiple arguments?
    buckets = [512, 1024, 2048, 4096, 5120]

    if args.jax_compilation_cache_dir:
        jax.config.update("jax_compilation_cache_dir", args.jax_compilation_cache_dir)

    devices = jax.local_devices(backend="gpu")
    if not devices:
        raise RuntimeError(
            "No GPU devices visible to JAX. This benchmark expects a GPU."
        )
    if args.gpu_index < 0 or args.gpu_index >= len(devices):
        raise ValueError(
            f"gpu-index {args.gpu_index} out of range for {len(devices)} visible GPU(s)"
        )
    # pick one single GPU per job
    device = devices[args.gpu_index]

    fold_inputs = list(folding_input.load_fold_inputs_from_path(input_json_path))
    if len(fold_inputs) != 1:
        raise ValueError("Expected exactly one fold input in the JSON file")
    fold_input = fold_inputs[0]
    if not fold_input.rng_seeds:
        raise ValueError("Input JSON must contain at least one model seed")
    rng_seed = int(fold_input.rng_seeds[0])

    # the number of tokens = the sequence length of input_json_path
    nominal_tokens_proxy = _load_nominal_sequence_length(input_json_path)

    # Create features out of the input sequence
    ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input,
        buckets=buckets,
        ccd=ccd,
        verbose=True,
        ref_max_modified_date=None,
        conformer_max_iterations=None,
        resolve_msa_overlaps=True,
    )
    # For benchmark purposes we can keep only the first featurised example (as they have the same shape)
    example = featurised_examples[0]
    # now call the main function for AF3 inference and create the model
    config = make_model_config(
        flash_attention_implementation=args.flash_attention_implementation,
        num_diffusion_samples=args.num_diffusion_samples,
        num_recycles=args.num_recycles,
        return_embeddings=False,
        return_distogram=False,
    )
    model_params = af3params.get_model_haiku_params(model_dir=model_dir)
    model_params_dev = jax.device_put(model_params, device)
    batch_dev = jax.device_put(
        jax.tree_util.tree_map(
            jnp.asarray, utils.remove_invalidly_typed_feats(example)
        ),
        device,
    )
    rng_key = jax.random.PRNGKey(rng_seed)

    # this comes directly from AF running script
    # https://github.com/google-deepmind/alphafold3/blob/main/run_alphafold.py#L425-L431
    @hk.transform
    def forward_fn(batch):
        """Define AF3 forward pass as a Haiku function"""
        return af3model.Model(config)(batch)

    apply_jit = jax.jit(forward_fn.apply, device=device)

    lower_t0 = time.perf_counter()
    lowered = apply_jit.lower(model_params_dev, rng_key, batch_dev)
    # here we're measuring:
    # - specialization to input shapes/dtypes
    # - creation of JAX intermediate repr
    # - lowering to XLA/Stable HLO input form
    lowering_time_s = time.perf_counter() - lower_t0

    compile_t0 = time.perf_counter()
    # and here we are compiling the model based on the hardware
    compiled = lowered.compile()
    compile_time_s = time.perf_counter() - compile_t0
    # we can then measure the JAX compilation time
    jax_compilation_time = lowering_time_s + compile_time_s
    cost_analysis = None
    for obj in (compiled, lowered):
        try:
            cost_analysis = obj.cost_analysis()
            if cost_analysis:
                break
        except Exception:
            continue
    # here we estimate the TFLOPs
    estimated_flops = _extract_flops(cost_analysis)
    # now we can run the model once to make sure compilation is done
    # and we are not including it in the benchmark time,
    # and also to have a warm start for the following runs
    first_result = compiled(model_params_dev, rng_key, batch_dev)
    jax.block_until_ready(first_result)
    # compilation is done, now run 3 times
    warm_times: list[float] = []
    warm_result = None
    for i in range(args.warm_runs):
        warm_key = jax.random.PRNGKey(rng_seed + 1000 + i)
        t0 = time.perf_counter()
        warm_result = compiled(model_params_dev, warm_key, batch_dev)
        jax.block_until_ready(warm_result)
        warm_times.append(time.perf_counter() - t0)

    warm_mean_s = statistics.mean(warm_times)
    warm_std_s = statistics.pstdev(warm_times) if len(warm_times) > 1 else 0.0

    true_num_tokens = None
    extraction_time_s = None
    num_samples = None
    if args.extract_results and warm_result is not None:
        extract_t0 = time.perf_counter()
        # here is the run inference
        # https://github.com/google-deepmind/alphafold3/blob/main/run_alphafold.py#L433-L453
        host_result = jax.tree.map(np.asarray, warm_result)
        host_result = jax.tree.map(
            lambda x: x.astype(np.float32)
            if getattr(x, "dtype", None) == jnp.bfloat16
            else x,
            host_result,
        )
        host_result = dict(host_result)
        identifier = np.asarray(model_params["__meta__"]["__identifier__"]).tobytes()
        host_result["__identifier__"] = identifier
        inference_results = list(
            af3model.Model.get_inference_result(
                batch=example,
                result=host_result,
                target_name=fold_input.name,
            )
        )
        extraction_time_s = time.perf_counter() - extract_t0
        if inference_results:
            true_num_tokens = int(len(inference_results[0].metadata["token_chain_ids"]))
            num_samples = int(len(inference_results))

    if true_num_tokens is None:
        true_num_tokens = nominal_tokens_proxy
    bucket_used = _bucket_for_tokens(true_num_tokens, buckets)

    xla_estimated_tflops_per_s = None
    if estimated_flops is not None and warm_mean_s > 0:
        xla_estimated_tflops_per_s = estimated_flops / warm_mean_s / 1.0e12

    compile_free_gpu_seconds = warm_mean_s

    summary = {
        "input_json": str(input_json_path),
        "job_name": fold_input.name,
        "seed_used_for_benchmark": rng_seed,
        "device_platform": device.platform,
        "flash_attention_implementation": args.flash_attention_implementation,
        "num_recycles": args.num_recycles,
        "num_diffusion_samples": args.num_diffusion_samples,
        "warm_runs": args.warm_runs,
        "jax_compilation_cache_dir": args.jax_compilation_cache_dir,
        "tokens_used": bucket_used,  # in our case tokens used = the bucket size
        "jax_compilation_time_s": jax_compilation_time,
        "warm_execute_times_s": warm_times,
        "warm_execute_mean_s": warm_mean_s,  # this is the compile-free gpu seconds, since we're using 1 GPU
        "warm_execute_std_s": warm_std_s,
        "compile_free_inference_time_s": warm_mean_s,
        "compile_free_gpu_seconds": compile_free_gpu_seconds,
        "xla_estimated_flops": estimated_flops,
        "xla_estimated_tflops_per_s": xla_estimated_tflops_per_s,
        "extraction_time_s": extraction_time_s,
        "num_output_samples": num_samples,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
