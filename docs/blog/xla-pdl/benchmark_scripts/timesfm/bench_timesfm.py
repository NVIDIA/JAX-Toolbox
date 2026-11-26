#!/usr/bin/env python3
# Copyright 2026. Standalone JAX inference-latency benchmark for TimesFM 2.5 (Flax).
#
# Measures steady-state forecast-step latency so that CUDA graphs (XLA command
# buffers) and PDL (Programmatic Dependent Launch) can be A/B'd via XLA_FLAGS.
# Timing discipline for this driver (MaxText instead reports a loop average):
#     jax.jit -> .lower(...) -> .compile()   (AOT: compile time separated from run time)
#     warm call discarded, then timed loop with jax.block_until_ready() inside.
#
# ---------------------------------------------------------------------------
# MODEL/API NOTES FOR THE VERSION USED IN THE ORIGINAL EXPERIMENTS
# ---------------------------------------------------------------------------
# 1. Model / API. There is no longer a PAX "timesfm-2.0-500m-jax" path in the
#    maintained package. TimesFM 2.5 (package `timesfm` 2.0.x) ships a *Flax*
#    backend: `timesfm.TimesFM_2p5_200M_flax`, weights `google/timesfm-2.5-200m-flax`.
#    The 2.0/1.x PAX code is archived under the repo's `v1/` subdir
#    (`pip install timesfm==1.3.0`). We target the Flax 2.5 path.
#      Install:  pip install 'timesfm[flax]'
#
# 2. What we time ("--level"). The public `model.forecast()` and even the
#    `model.compiled_decode` wrapper are NOT clean latency probes:
#      - `compiled_decode` ends with `np.array(full_forecast)` (a blocking D2H
#        copy) and calls `try_gc()`, which can invoke `gc.collect()`. Both add
#        host-side noise that swamps the few-percent kernel-launch effect that
#        CUDA graphs / PDL produce.
#      - `forecast()` additionally does a per-series Python/NumPy
#        interpolation+padding loop.
#    So the default `--level core` times the pure-JAX jitted core,
#    `TimesFM_2p5_200M_flax_module.decode(horizon, inputs, masks)`, which is
#    async and therefore correctly measured with block_until_ready().
#    `--level e2e` is provided to time the official wrapper for reference.
#
# 3. dtype. This Flax path is float32 end-to-end: the wrapper casts inputs with
#    `jnp.array(inputs, dtype=jnp.float32)` and the decode cache is allocated as
#    float32. There is no supported bfloat16 knob, so the default here is
#    float32 (NOT bfloat16 as the spec suggested). `--dtype bfloat16` is offered
#    as an EXPERIMENTAL param cast; it produces a mixed-precision graph and is
#    only meaningful if used identically in both A/B arms.
#
# 4. Horizon and AR steps. Inside decode: num_decode_steps = (horizon-1)//128.
#    So HORIZON_LEN=128 yields ZERO autoregressive steps -- prefill only. The
#    launch-bound autoregressive loop (where CUDA graphs help most) only appears
#    at horizon > 128. Sweep horizon in {128, 256, 512} to see it.
#
# Shape constraints (enforced below, mirroring the library):
#    context must be a multiple of input_patch_len  = 32
#    horizon must be a multiple of output_patch_len = 128
#    context + horizon <= context_limit = 16384

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time

# ---------------------------------------------------------------------------
# CLI. Parsed BEFORE importing jax so we can fail fast on bad args without
# paying JAX/CUDA init cost. XLA_FLAGS is read by XLA once at backend init, so
# it must already be set in the environment by the caller (see run_benchmark.sh).
# ---------------------------------------------------------------------------
def parse_args(argv=None):
  p = argparse.ArgumentParser(
      description="TimesFM 2.5 (Flax) forecast-step latency benchmark.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  p.add_argument("--batch-size", type=int, default=1,
                 help="Number of series forecast per call.")
  p.add_argument("--context-len", type=int, default=512,
                 help="Past timesteps per series. Rounded up to a multiple of 32.")
  p.add_argument("--horizon-len", type=int, default=512,
                 help="Forecast steps. Rounded up to a multiple of 128. "
                      "NOTE: 128 => prefill only, no autoregressive steps.")
  p.add_argument("--iters", type=int, default=20,
                 help="Timed iterations. Use >= 20; CUDA-graph gains are often only a few %%.")
  p.add_argument("--warmup", type=int, default=2,
                 help="Discarded warm calls before timing (absorbs autotune / cache warmup).")
  p.add_argument("--level", choices=("core", "e2e"), default="core",
                 help="'core' = jitted decode kernel (recommended, clean signal). "
                      "'e2e' = official compiled_decode wrapper (includes D2H copy + GC).")
  p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32",
                 help="float32 is the model's native/supported dtype here. "
                      "bfloat16 is an EXPERIMENTAL param cast.")
  # Weights
  p.add_argument("--random-weights", action="store_true",
                 help="Skip the checkpoint entirely (no network/HF access). "
                      "Preserves workload shapes, but does not test model quality.")
  p.add_argument("--model-id", default="google/timesfm-2.5-200m-flax",
                 help="HuggingFace repo id used when not --random-weights.")
  p.add_argument("--checkpoint-path", default=None,
                 help="Local Orbax checkpoint dir. Fails fast if set but missing.")
  # e2e-only forecast flags. Defaults chosen to be CHEAP and DETERMINISTIC.
  p.add_argument("--force-flip-invariance", action="store_true",
                 help="[e2e] Runs decode TWICE (on x and -x). Doubles latency. Off by default.")
  p.add_argument("--use-continuous-quantile-head", action="store_true",
                 help="[e2e] Requires horizon <= 1024.")
  p.add_argument("--normalize-inputs", action="store_true", help="[e2e]")
  # Output
  p.add_argument("--output-json", default=None, help="Path to write the JSON summary.")
  p.add_argument("--label", default=None, help="Free-form label recorded in the JSON (e.g. 'ON').")
  # Profiling. When --profile-dir is set, the timed loop is wrapped in a
  # JAX/XLA trace (open in TensorBoard -> Profile -> Trace Viewer). Use a
  # SEPARATE run for this; the tracer adds overhead, so do NOT quote the
  # latency from a profiled run.
  p.add_argument("--profile-dir", default=None,
                 help="If set, capture a JAX/XLA trace of the timed loop into this dir.")
  p.add_argument("--profile-runs", type=int, default=5,
                 help="Iterations included inside the trace (keep small; traces grow fast).")
  return p.parse_args(argv)


def round_up(value: int, multiple: int, name: str) -> int:
  """Round `value` up to a multiple of `multiple`, mirroring library behaviour."""
  if value % multiple:
    new = math.ceil(value / multiple) * multiple
    print(f"[bench] {name}={value} is not a multiple of {multiple}; using {new}.",
          file=sys.stderr)
    return new
  return value


def percentile(sorted_vals, q: float) -> float:
  """Linear-interpolation percentile; avoids a hard NumPy-version dependency."""
  if not sorted_vals:
    return float("nan")
  if len(sorted_vals) == 1:
    return sorted_vals[0]
  pos = (len(sorted_vals) - 1) * q
  lo = math.floor(pos)
  hi = math.ceil(pos)
  if lo == hi:
    return sorted_vals[int(pos)]
  return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def main(argv=None) -> int:
  args = parse_args(argv)

  for name in ("batch_size", "context_len", "horizon_len", "iters", "profile_runs"):
    if getattr(args, name) < 1:
      raise ValueError(f"{name} must be positive")
  if args.warmup < 0:
    raise ValueError("warmup must be nonnegative")
  if args.level == "e2e" and args.dtype != "float32":
    raise ValueError("The e2e wrapper does not implement a bfloat16 switch")
  if args.output_json and os.path.exists(args.output_json):
    raise FileExistsError(f"Refusing to overwrite {args.output_json}")

  # Fail fast on a bad checkpoint path BEFORE importing jax / touching the GPU.
  if args.checkpoint_path is not None and not os.path.exists(args.checkpoint_path):
    print(f"[bench] FATAL: --checkpoint-path '{args.checkpoint_path}' does not exist.",
          file=sys.stderr)
    return 2
  if args.random_weights and args.checkpoint_path:
    print("[bench] FATAL: --random-weights and --checkpoint-path are mutually exclusive.",
          file=sys.stderr)
    return 2
  if args.iters < 1:
    print("[bench] FATAL: --iters must be >= 1.", file=sys.stderr)
    return 2
  if args.iters < 20:
    print(f"[bench] WARNING: --iters={args.iters} < 20; stats will be noisy relative "
          "to the few-percent effect being measured.", file=sys.stderr)

  # Echo the flags that actually govern the measurement, for provenance.
  print("=" * 74)
  print("TimesFM 2.5 (Flax) forecast-step latency benchmark")
  print("=" * 74)
  print(f"[env] XLA_FLAGS            = {os.environ.get('XLA_FLAGS', '<unset>')}")
  print(f"[env] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
  print(f"[env] JAX_COMPILATION_CACHE_DIR = "
        f"{os.environ.get('JAX_COMPILATION_CACHE_DIR', '<unset>')!r}")

  # --- imports (after arg validation) --------------------------------------
  try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
  except ImportError as exc:
    print(f"[bench] FATAL: JAX/Flax import failed ({exc}).\n"
          "        Install with: pip install 'timesfm[flax]'", file=sys.stderr)
    return 2

  try:
    import timesfm
  except ImportError as exc:
    print(f"[bench] FATAL: `import timesfm` failed ({exc}).\n"
          "        Install with: pip install 'timesfm[flax]'", file=sys.stderr)
    return 2

  if not hasattr(timesfm, "TimesFM_2p5_200M_flax"):
    # timesfm/__init__.py swallows the ImportError for a missing backend, so an
    # absent Flax extra shows up as a missing attribute rather than a traceback.
    print("[bench] FATAL: timesfm.TimesFM_2p5_200M_flax is unavailable. The Flax\n"
          "        extra is not installed (timesfm/__init__.py silently skips a\n"
          "        backend whose imports fail). Run: pip install 'timesfm[flax]'\n"
          "        and ensure flax/optax/einshape/orbax-checkpoint/jaxtyping import.",
          file=sys.stderr)
    return 2

  devices = jax.devices()
  print(f"[env] jax={jax.__version__} jaxlib={getattr(jax, 'lib', None) and jax.lib.__version__}")
  print(f"[env] timesfm={getattr(timesfm, '__version__', 'unknown')} "
        f"flax={getattr(__import__('flax'), '__version__', 'unknown')}")
  print(f"[env] devices={devices}")
  if len(devices) != 1 or devices[0].platform != "gpu":
    raise RuntimeError("This experiment requires exactly one visible GPU")

  # --- shape normalisation (identical in both A/B arms) --------------------
  P_IN, P_OUT, CTX_LIMIT = 32, 128, 16384
  context_len = round_up(args.context_len, P_IN, "context-len")
  horizon_len = round_up(args.horizon_len, P_OUT, "horizon-len")
  if context_len + horizon_len > CTX_LIMIT:
    print(f"[bench] FATAL: context+horizon = {context_len}+{horizon_len} > {CTX_LIMIT}.",
          file=sys.stderr)
    return 2
  num_ar_steps = (horizon_len - 1) // P_OUT
  print(f"[cfg] batch={args.batch_size} context={context_len} horizon={horizon_len} "
        f"dtype={args.dtype} level={args.level}")
  print(f"[cfg] autoregressive decode steps = {num_ar_steps}"
        + ("  (PREFILL ONLY -- raise --horizon-len above 128 to exercise the "
           "launch-bound AR loop)" if num_ar_steps == 0 else ""))
  print("[cfg] TIP: pin GPU clocks to cut variance:  sudo nvidia-smi -lgc <max>")

  # --- model ---------------------------------------------------------------
  t_load0 = time.perf_counter()
  if args.random_weights:
    # The module constructor initialises params from jax.random.key(42); no
    # checkpoint, no network. These weights are for workload structure, not quality.
    print("[model] --random-weights: constructing with random init (no checkpoint).")
    model = timesfm.TimesFM_2p5_200M_flax()
  elif args.checkpoint_path:
    print(f"[model] loading local checkpoint: {args.checkpoint_path}")
    model = timesfm.TimesFM_2p5_200M_flax()
    model.load_checkpoint(args.checkpoint_path)
  else:
    print(f"[model] from_pretrained({args.model_id})")
    try:
      model = timesfm.TimesFM_2p5_200M_flax.from_pretrained(args.model_id)
    except Exception as exc:
      print(f"[bench] FATAL: checkpoint load failed: {exc}\n"
            "        For a pure launch-overhead A/B use --random-weights "
            "(no network needed).", file=sys.stderr)
      return 2
  load_s = time.perf_counter() - t_load0
  print(f"[model] ready in {load_s:.2f}s")

  target_dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
  if args.dtype == "bfloat16":
    print("[bench] WARNING: bfloat16 is EXPERIMENTAL here. The decode cache and "
          "running stats are allocated float32 by the library, so the graph will "
          "be mixed precision. Only comparable if both arms use it.", file=sys.stderr)

  # --- dummy inputs (fixed across arms so the compiled graphs match) -------
  rng = np.random.default_rng(0)
  inputs_np = rng.standard_normal((args.batch_size, context_len), dtype=np.float32)
  masks_np = np.zeros((args.batch_size, context_len), dtype=bool)  # False = observed

  compile_s = float("nan")
  aot = False

  if args.level == "core":
    # -------------------------------------------------------------------
    # Build an AOT-compiled executable for the pure decode kernel.
    #
    # module.decode is already wrapped in nnx.jit. Rather than nest jits, we
    # recover the undecorated function (flax's JitWrapped calls
    # functools.update_wrapper, so __wrapped__ is the raw method) and drive it
    # through the standard nnx functional split/merge pattern under our own
    # jax.jit. That gives a genuine lower()/compile() split.
    # -------------------------------------------------------------------
    module = model.model
    inputs = jnp.asarray(inputs_np, dtype=target_dtype)
    masks = jnp.asarray(masks_np)

    graphdef, state = nnx.split(module)
    if target_dtype is not jnp.float32:
      state = jax.tree.map(
          lambda x: x.astype(target_dtype)
          if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x,
          state)

    raw_decode = getattr(type(module).decode, "__wrapped__", None)

    if raw_decode is None:
      raise RuntimeError("decode.__wrapped__ is missing: use a compatible TimesFM/Flax build. "
                         "Refusing to silently change the benchmark's execution path.")
    def pure_decode(state_, inputs_, masks_):
      merged = nnx.merge(graphdef, state_)
      return raw_decode(merged, horizon_len, inputs_, masks_)
    t0 = time.perf_counter()
    compiled = jax.jit(pure_decode).lower(state, inputs, masks).compile()
    compile_s = time.perf_counter() - t0
    fn = lambda: compiled(state, inputs, masks)
    jax.block_until_ready(fn())
    aot = True
    print(f"[compile] AOT lower+compile: {compile_s:.2f}s")

  else:
    # ---------------------- e2e: official wrapper ----------------------
    fc = timesfm.ForecastConfig(
        max_context=context_len,
        max_horizon=horizon_len,
        normalize_inputs=args.normalize_inputs,
        per_core_batch_size=args.batch_size,
        use_continuous_quantile_head=args.use_continuous_quantile_head,
        force_flip_invariance=args.force_flip_invariance,
        infer_is_positive=False,   # cheap + deterministic
        fix_quantile_crossing=False,
    )
    t0 = time.perf_counter()
    model.compile(fc, dryrun=True)  # dryrun performs the compilation
    compile_s = time.perf_counter() - t0
    print(f"[compile] model.compile(): {compile_s:.2f}s")
    print("[bench] NOTE: e2e includes a blocking np.array() D2H copy and try_gc(); "
          "expect higher variance and a diluted CUDA-graph/PDL delta.", file=sys.stderr)
    fn = lambda: model.compiled_decode(horizon_len, inputs_np, masks_np)

  # --- warmup (discarded) --------------------------------------------------
  for i in range(max(0, args.warmup)):
    jax.block_until_ready(fn())
  print(f"[bench] discarded {max(0, args.warmup)} warm call(s); timing {args.iters} iters")

  # --- timed loop ----------------------------------------------------------
  # block_until_ready() INSIDE the timed region is mandatory: JAX dispatch is
  # async, so without it we would be timing dispatch, not execution.
  #
  # Optional profiling: when --profile-dir is set we wrap ONLY this steady-state
  # loop in a JAX/XLA trace (compile + warmup already happened above), and label
  # each iteration with a StepTraceAnnotation so a single clean 'forecast_step'
  # can be isolated in the TensorBoard Trace Viewer.
  profiling = args.profile_dir is not None
  if profiling:
    os.makedirs(args.profile_dir, exist_ok=True)
    jax.profiler.start_trace(args.profile_dir)
    print(f"[profile] tracing {args.profile_runs} iters into {args.profile_dir}",
          file=sys.stderr)

  n_iters = args.profile_runs if profiling else args.iters
  samples_ms = []
  out = None
  try:
    for i in range(n_iters):
      with jax.profiler.StepTraceAnnotation("forecast_step", step_num=i):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        samples_ms.append((time.perf_counter() - t0) * 1e3)
  finally:
    if profiling:
      jax.profiler.stop_trace()
      print(f"[profile] trace closed in {args.profile_dir}", file=sys.stderr)
  del out

  s = sorted(samples_ms)
  mean_ms = statistics.fmean(samples_ms)
  std_ms = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
  result = {
      "label": args.label,
      "level": args.level,
      "aot_compiled": aot,
      "batch_size": args.batch_size,
      "context_len": context_len,
      "horizon_len": horizon_len,
      "num_ar_steps": num_ar_steps,
      "dtype": args.dtype,
      "iters": n_iters,
      "warmup": args.warmup,
      "random_weights": args.random_weights,
      "profiled": profiling or os.environ.get("PDL_CAPTURE_MODE") == "nsys",
      "capture_mode": os.environ.get("PDL_CAPTURE_MODE", "xprof" if profiling else "benchmark"),
      "profile_dir": args.profile_dir,
      "mean_ms": mean_ms,
      "std_ms": std_ms,
      "min_ms": s[0],
      "p50_ms": percentile(s, 0.50),
      "p90_ms": percentile(s, 0.90),
      "max_ms": s[-1],
      "throughput_forecasts_per_s": (args.batch_size / (mean_ms / 1e3)) if mean_ms else None,
      "compile_s": compile_s,
      "model_load_s": load_s,
      "samples_ms": samples_ms,
      "xla_flags": os.environ.get("XLA_FLAGS", ""),
      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
      "jax_version": jax.__version__,
      "timesfm_version": getattr(timesfm, "__version__", "unknown"),
      "device_kind": devices[0].device_kind if devices else None,
      "platform": devices[0].platform if devices else None,
      "hostname": platform.node(),
  }

  print("-" * 74)
  if result["profiled"]:
    print(f"[profile] NOTE: this was a PROFILED run ({n_iters} iters); latency below "
          "includes tracer overhead -- do not quote it.")
  print(f"mean   {mean_ms:9.3f} ms   (std {std_ms:.3f})")
  print(f"p50    {result['p50_ms']:9.3f} ms")
  print(f"p90    {result['p90_ms']:9.3f} ms")
  print(f"min    {s[0]:9.3f} ms      max {s[-1]:9.3f} ms")
  print(f"thrpt  {result['throughput_forecasts_per_s']:9.1f} forecasts/s")
  print(f"compile{compile_s:9.2f} s")
  print("-" * 74)

  if args.output_json:
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "x") as fh:
      json.dump(result, fh, indent=2)
    print(f"[bench] wrote {args.output_json}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
