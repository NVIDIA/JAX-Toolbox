# Reconstructed PDL benchmark and profile runners

Here are the scripts for replicating the PDL study.

The default experiment is **CUDA graphs ON, PDL OFF versus ON**. Each arm is
a fresh process. Use unprofiled runs for latency; use XProf and Nsight in
separate runs to explain kernel scheduling. No benchmark is run just by
reading this directory or calling `--dry-run`.

## Entry points

Run these from the `blog/xla-pdl` directory of your JAX Toolbox checkout,
inside the appropriate GPU container (or `/workspace` with the mount below):

| Workload | Latency benchmark | XProf / xplane capture | Nsight Systems capture |
|---|---|---|---|
| Gemma 4 / MaxText | [gemma/run_benchmark.sh](gemma/run_benchmark.sh) | [gemma/run_xprof.sh](gemma/run_xprof.sh) | [gemma/run_nsys.sh](gemma/run_nsys.sh) |
| TimesFM 2.5 / Flax | [timesfm/run_benchmark.sh](timesfm/run_benchmark.sh) | [timesfm/run_xprof.sh](timesfm/run_xprof.sh) | [timesfm/run_nsys.sh](timesfm/run_nsys.sh) |
| AlphaFold 3 | [alphafold/run_benchmark.sh](alphafold/run_benchmark.sh) | [alphafold/run_xprof.sh](alphafold/run_xprof.sh) | [alphafold/run_nsys.sh](alphafold/run_nsys.sh) |

All nine wrappers use [run_experiment.py](run_experiment.py). TimesFM and AF3
model drivers are included locally; Gemma uses the MaxText entry point installed
in its container. There is no dependency on the author's original home directory.

## Container and input prerequisites

Use a Linux GPU host with a compatible driver and a Hopper-or-newer GPU. The
launcher/comparator require Python 3.10+ and only the standard library; the
inference processes require their model packages and compatible JAX/XLA builds.

Choose the [JAX Toolbox image](https://github.com/NVIDIA/JAX-Toolbox#container-images)
for the model: `ghcr.io/nvidia/jax:maxtext`, `ghcr.io/nvidia/jax:alphafold`, or
`ghcr.io/nvidia/jax:jax` plus the TimesFM Flax dependencies. Pin the image digest
and package versions. These moving tags are examples, **not the archived builds
from the historical measurements**. MaxText/TimesFM/AF3 internal APIs can change.

Example host-side setup for Gemma, starting at the JAX Toolbox repository root:

```bash
cd blog/xla-pdl
PDL_IMAGE=ghcr.io/nvidia/jax:maxtext
docker pull "$PDL_IMAGE"

docker run --rm -it --gpus 'device=0' --shm-size=2g \
  -e PDL_IMAGE_PINNED \
  --mount "type=bind,source=$PWD,target=/workspace" \
  --workdir /workspace "$PDL_IMAGE_PINNED" bash
```

Use the AF3 image for AlphaFold. 
For TimesFM we suggest to install the package directly from source: 
```bash 
git clone https://github.com/google-research/timesfm.git
cd timsefm
pip install -e . --no-deps
```

Gemma needs the base config and tokenizer paths below. With `RANDOM_WEIGHTS=1`,
the launcher omits the checkpoint argument as the original did: use a base
config whose checkpoint-loading defaults are empty. With real weights, set
`RANDOM_WEIGHTS=0 CKPT=/path/to/checkpoint`. The checkpoint must be compatible
with the selected MaxText model/configuration.

AF3 needs the preprocessed input JSON and a prepared parameter directory. These
are **not bundled**, nor are model weights, MSAs, templates, or databases. Use
your original `bench_L02048.json`; a raw sequence alone is not equivalent to
the prepared input. Real parameters remain subject to their access terms.
For the original synthetic-parameter setup, the AF3 container must provide
the parameter-schema and generator helpers. Prepare them explicitly once:

```bash
python benchmark_scripts/alphafold/prepare_random_weights.py \
  --model-dir /workspace/af3_random_parameters
export MODEL_DIR=/workspace/af3_random_parameters
export INPUT_JSON=/opt/alphafold_input_data/input_sequences/bench_L02048.json
```

The helper refuses an existing model directory. It also runs AF3's data-build
helper, which writes package data inside the container. The benchmark/profile
scripts themselves **never regenerate or overwrite parameters**. Random weights
exercise workload structure, not prediction quality or numerical equivalence.

## Run the three experiments

Every command creates a **new** directory and runs both PDL arms. Reusing an
existing output path is rejected. Omit `--out-dir` to get a timestamped directory
under `benchmark_scripts/results/<model>/<mode>/`.

### Gemma: one GB300, TP=1, batch setting 4

```bash
export TP=1 FSDP=1 PER_DEVICE_BS=4 PREFILL_LEN=1024 TARGET_LEN=2048
export CUDA_VISIBLE_DEVICES=0
bash benchmark_scripts/gemma/run_benchmark.sh --repeats 3 --out-dir results/gemma_latency
bash benchmark_scripts/gemma/run_xprof.sh --out-dir results/gemma_xprof
bash benchmark_scripts/gemma/run_nsys.sh --out-dir results/gemma_nsys
```

For the H100/four-GB300 TP=4 configuration, instead use:

```bash
export TP=4 FSDP=1 PER_DEVICE_BS=1 CUDA_VISIBLE_DEVICES=0,1,2,3
# Run the same three scripts with NEW output directories.
```

`PER_DEVICE_BS` is MaxText's `per_device_batch_size` config knob; do not infer
per-rank GEMM shapes or global batch by dividing/multiplying it by TP alone.
Inspect the native JSON's `autoregressive.global_batch_size` and compiled shapes.
The launcher rejects a device count inconsistent with TP × FSDP. TP=4 does not
by itself promise a larger PDL gain.

### TimesFM: one GPU, batch 1, context 512, horizon 512

```bash
export CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=1 CONTEXT_LEN=512 HORIZON_LEN=512
bash benchmark_scripts/timesfm/run_benchmark.sh --repeats 3 --out-dir results/timesfm_latency
bash benchmark_scripts/timesfm/run_xprof.sh --out-dir results/timesfm_xprof
bash benchmark_scripts/timesfm/run_nsys.sh --out-dir results/timesfm_nsys
```

This is the pure compiled decode core, not the public forecast wrapper. Horizon
512 exercises three autoregressive decode steps after prefill; horizon 128
does not exercise that loop. The native dtype is float32. `DTYPE=bfloat16` is
an experimental mixed-precision parameter cast, not the historical default.

### AlphaFold 3: one GPU, prepared 2048-token input

```bash
export CUDA_VISIBLE_DEVICES=0 WARM_RUNS=15
export INPUT_JSON=/path/to/bench_L02048.json MODEL_DIR=/path/to/prepared_parameters
bash benchmark_scripts/alphafold/run_benchmark.sh --out-dir results/af3_latency
bash benchmark_scripts/alphafold/run_xprof.sh --out-dir results/af3_xprof
bash benchmark_scripts/alphafold/run_nsys.sh --out-dir results/af3_nsys
```

The legacy name `WARM_RUNS=15` means **15 measured, compile-free iterations**,
not 15 discarded warmups. One initial execution is discarded. AF3 is expensive:
start with one pair, and use `--repeats 3` when a more reliable small-effect
estimate is needed. Inspect samples and medians; do not silently remove stalls.

## Settings shared by all three modes

| Model | Environment overrides and defaults |
|---|---|
| Gemma | `TP=1`, `FSDP=1`, `PER_DEVICE_BS=4`, `PREFILL_LEN=1024`, `TARGET_LEN=2048`, `ITERS=20`, `PROFILE_ITERS=20`, `ATTENTION=dot_product`, `WEIGHT_DTYPE=bfloat16`, `MODEL_NAME=gemma4-31b`, `RANDOM_WEIGHTS=1`, optional `CKPT` |
| TimesFM | `BATCH_SIZE=1`, `CONTEXT_LEN=512`, `HORIZON_LEN=512`, `ITERS=20`, `PROFILE_ITERS=5`, `WARMUP=2`, `LEVEL=core`, `DTYPE=float32`, `RANDOM_WEIGHTS=1`, optional `CHECKPOINT_PATH` or `MODEL_ID` |
| AlphaFold | `WARM_RUNS=15`, `PROFILE_ITERS=1` for XProf / `3` for Nsight, `NUM_RECYCLES=10`, `NUM_DIFFUSION_SAMPLES=1`, `FLASH_ATTENTION_IMPLEMENTATION=triton`, `INPUT_JSON`, `MODEL_DIR` |

Gemma always sets `scan_layers=false` and `inference_microbenchmark_stages=generate`.
`BASE_CONFIG` defaults to `/opt/maxtext/src/maxtext/configs/base.yml`;
`TOKENIZER_PATH` to `/opt/maxtext/src/maxtext/assets/tokenizers/tokenizer_gemma4.model`.
AF3 defaults to the container input/parameter paths shown above. All defaults
are visible in `--dry-run`; it neither imports JAX nor requires those files to
exist locally. For example, inspect the exact four-GPU Nsight command:

```bash
TP=4 PER_DEVICE_BS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash benchmark_scripts/gemma/run_nsys.sh --dry-run
```

Common switches: `--repeats N`, `--order off-on|on-off`, `--out-dir PATH`.
Successive repeats reverse the order. `--matrix` adds CUDA-graphs-OFF arms
**only in benchmark mode**, for secondary attribution. The headline summary
still compares `cg_on_pdl_off` with `cg_on_pdl_on`.

## Flags, timing, and interpretation

The launcher preserves common `XLA_FLAGS` and optional `XLA_FLAGS_EXTRA`, removes
conflicting PDL settings, then sets both switches explicitly:

```text
PDL off: --xla_gpu_enable_pdl=false --xla_gpu_enable_pdl_launch=false
PDL on:  --xla_gpu_enable_pdl=true  --xla_gpu_enable_pdl_launch=true
```

The master flag gates PDL; setting both is explicit experiment hygiene, not a
claim that master=false leaves PDL active. Graphs-ON uses the build's enabled
default (or a retained nonempty command-buffer configuration). An inherited
empty `--xla_gpu_enable_command_buffer=` is rejected. The secondary graphs-OFF
arms add that empty flag explicitly. Verify graph execution in the traces;
the launcher's intended settings alone do not prove capture occurred.

AF3 additionally sets `--xla_gpu_enable_latency_hiding_scheduler=true` as in the
original runs. Persistent JAX compilation caching is disabled for all arms;
this does not disable within-process executable/graph reuse. Autotuning is a
separate issue: the scripts **do not automatically freeze autotuning**. Supply
the appropriate [persisted-autotuning flags](https://openxla.org/xla/persisted_autotuning)
through the common flags for your pinned XLA build, and verify cache hits and
matching kernel choices. Do not reuse another GPU architecture's tuning cache.

Timing semantics intentionally match the original model paths:

- **TimesFM:** AOT compilation/validation, discarded warmups, then each decode
  call timed through `jax.block_until_ready`; raw `samples_ms` are retained.
  The core now fails if its AOT API is unavailable instead of silently switching
  to a different execution path.
- **AF3:** lowering/compilation, one discarded execution, then each forward pass
  timed through `block_until_ready`. Raw `warm_execute_times_s` are retained.
  Featurization and host-side structure extraction are outside this timer.
- **Gemma:** the container's native MaxText microbenchmark performs warmup and
  synchronizes at the **end of the measured AR loop**, dividing by its iteration
  count. `autoregressive.step_in_ms` is a loop average, not a distribution of
  individually synchronized requests. This bundle does not fabricate per-step
  Gemma samples or medians. Check the installed MaxText implementation when
  changing container versions.

The comparator reports `(T_off - T_on) / T_off × 100`, using equal-weight process
means and also displaying paired results. For AF3/TimesFM it shows per-pair
median comparisons. It makes **no significance claim**, and refuses profiled,
incomplete, or shape/path-mismatched data. Pin clocks separately if appropriate
for your node; no script changes clocks, invokes sudo, or checks model accuracy.

## Saved files and how to view them

Each experiment has `manifest.json` (settings, exact argument arrays, flags,
order, status, script/input hashes), `python_packages.txt`, and `nvidia-smi.txt`.
Set/export `PDL_IMAGE_PINNED` inside the container to record its digest; it cannot
be inferred reliably from the model process. Nsight also records its version.
Each `run01/cg_on_pdl_off/` and `run01/cg_on_pdl_on/` contains:

- `raw.json`: the model's native output. In a profiling experiment this is
  **not** a latency benchmark; the manifest is authoritative even if MaxText's
  JSON has no `profiled` field.
- `run.log`: stdout/stderr, including model initialization/configuration.
- XProf mode: an `xprof/` subtree containing `plugins/profile/.../*.xplane.pb`.
- Nsight mode: `trace.nsys-rep`, or `trace.zip` with `--nsys-tool nsys-jax`.

Successful unprofiled runs also write `summary.json`. Reprint the summary with:

```bash
python benchmark_scripts/compare_results.py results/timesfm_latency/manifest.json
```

For XProf, use a compatible [XProf/TensorBoard installation](https://docs.jax.dev/en/latest/profiling.html)
and point its log directory at the **individual arm's** `xprof/` directory (for
Gemma, locate the log root immediately above the nested `plugins/` directory).
AF3 labels iterations `warm_step`; TimesFM labels them `forecast_step`.
MaxText owns Gemma's profiler: its microbenchmark can bracket the whole AR loop,
so `skip_first_n_steps_for_profiler`/`profiler_steps` do **not** guarantee a
five-step capture there. Select steady repeated generate modules/sentinels;
do not assume a `generate` StepTraceAnnotation exists.

The Nsight scripts capture the **whole process**, including initialization and
compilation; this matches the working original capture method. `PROFILE_ITERS`
limits model iterations, not process-wide trace duration. AF3 defaults to three
measured iterations for Nsight so a following invocation exists for
start-to-next-start selection. Profiles can be large. No unverified NVTX
capture-range setting is used. Graph-node tracing is enabled for graph coverage
and kernel analysis; sampling/context-switch collection is disabled.

For JAX/XLA attribution via JAX Toolbox's wrapper:

```bash
bash benchmark_scripts/gemma/run_nsys.sh --nsys-tool nsys-jax \
  --out-dir results/gemma_nsys_jax
```

[`nsys-jax`](https://github.com/NVIDIA/JAX-Toolbox/blob/main/docs/nsys-jax.md) adds
metadata-dump flags. The model drivers leave these flags intact. Extract its
ZIP to find the report and metadata. With plain Nsight, export on the capture
node using the **matching Nsight version**, then copy both report and SQLite
locally if your local GUI cannot open the report:

```bash
nsys export --type sqlite \
  --output results/gemma_nsys/run01/cg_on_pdl_on/trace.sqlite \
  results/gemma_nsys/run01/cg_on_pdl_on/trace.nsys-rep
nsys stats --report cuda_gpu_kern_sum \
  results/gemma_nsys/run01/cg_on_pdl_on/trace.nsys-rep
```

Compare corresponding steady kernels on the same GPU/context/stream, graph
membership, and `dependent_start - producer_end` (negative means overlap).
Look for `xla.pdl_launch` markers when available and the programmatic launch
attribute/graph dependency, not just the presence of `cuLaunchKernelEx`.
An API name alone is not proof of PDL, and graph replay need not show an
individual host launch for each kernel. This publication bundle contains the
collection scripts, not the original study's full gap/overlap analyzers or raw
profiles. Use Nsight Systems or the notebook workflow in the linked `nsys-jax`
documentation for downstream analysis, selecting fresh steady-step boundaries
for each new trace.

## Local validation and limitations

```bash
python3 -m unittest discover -s benchmark_scripts/tests -v
```

These tests validate flag construction, all entry points, output safety,
comparison logic, and mocked process/capture flows without a GPU. GPU inference
and capture must still be smoke-tested in the selected containers. Historical
results are not regenerated by these local tests, and this bundle makes no
claim that a moving package/image version reproduces an earlier percentage.
