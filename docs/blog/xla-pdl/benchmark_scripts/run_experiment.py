#!/usr/bin/env python3
"""Reconstructed PDL experiments. Uses only stdlib; models run in child processes.

See README.md for container dependencies, timing semantics, and provenance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PDL_KEYS = {"--xla_gpu_enable_pdl", "--xla_gpu_enable_pdl_launch"}
CG_KEY = "--xla_gpu_enable_command_buffer"
LHS_KEY = "--xla_gpu_enable_latency_hiding_scheduler"


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("gemma", "timesfm", "alphafold"))
    parser.add_argument("mode", choices=("benchmark", "xprof", "nsys"))
    parser.add_argument("--out-dir", help="New directory; never overwrite an existing run")
    parser.add_argument("--repeats", type=positive, default=1,
                        help="Independent process pairs; alternate order between repeats")
    parser.add_argument("--order", choices=("off-on", "on-off"), default="off-on")
    parser.add_argument("--matrix", action="store_true",
                        help="Also run graphs-OFF arms (benchmark mode only)")
    parser.add_argument("--nsys-tool", choices=("nsys", "nsys-jax"), default="nsys")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the exact plan, without creating files or running models")
    args = parser.parse_args(argv)
    if args.matrix and args.mode != "benchmark":
        parser.error("--matrix is secondary attribution for unprofiled benchmarks only")
    return args


def config(model, env):
    defaults = {
        "gemma": dict(TP="1", FSDP="1", PER_DEVICE_BS="4", PREFILL_LEN="1024",
                      TARGET_LEN="2048", ITERS="20", PROFILE_ITERS="20",
                      MODEL_NAME="gemma4-31b", WEIGHT_DTYPE="bfloat16",
                      ATTENTION="dot_product", RANDOM_WEIGHTS="1", CKPT="",
                      BASE_CONFIG="/opt/maxtext/src/maxtext/configs/base.yml",
                      TOKENIZER_PATH="/opt/maxtext/src/maxtext/assets/tokenizers/tokenizer_gemma4.model"),
        "timesfm": dict(BATCH_SIZE="1", CONTEXT_LEN="512", HORIZON_LEN="512",
                        ITERS="20", PROFILE_ITERS="5", WARMUP="2", LEVEL="core",
                        DTYPE="float32", RANDOM_WEIGHTS="1", CHECKPOINT_PATH="",
                        MODEL_ID="google/timesfm-2.5-200m-flax"),
        "alphafold": dict(INPUT_JSON="/opt/alphafold_input_data/input_sequences/bench_L02048.json",
                         MODEL_DIR="/opt/alphafold/model", WARM_RUNS="15",
                         PROFILE_ITERS="1", NUM_RECYCLES="10", NUM_DIFFUSION_SAMPLES="1",
                         FLASH_ATTENTION_IMPLEMENTATION="triton"),
    }[model]
    cfg = {key: env.get(key, value) for key, value in defaults.items()}
    integer_keys = {"TP", "FSDP", "PER_DEVICE_BS", "PREFILL_LEN", "TARGET_LEN", "ITERS",
                    "PROFILE_ITERS", "BATCH_SIZE", "CONTEXT_LEN", "HORIZON_LEN", "WARMUP",
                    "WARM_RUNS", "NUM_RECYCLES", "NUM_DIFFUSION_SAMPLES"}
    for key in integer_keys & cfg.keys():
        cfg[key] = int(cfg[key])
        if cfg[key] < (0 if key in {"WARMUP", "NUM_RECYCLES"} else 1):
            raise ValueError(f"Invalid {key}={cfg[key]}")
    if "RANDOM_WEIGHTS" in cfg and cfg["RANDOM_WEIGHTS"] not in {"0", "1"}:
        raise ValueError("RANDOM_WEIGHTS must be 0 or 1")
    if model == "gemma":
        if cfg["TARGET_LEN"] <= cfg["PREFILL_LEN"]:
            raise ValueError("TARGET_LEN must exceed PREFILL_LEN")
        if cfg["RANDOM_WEIGHTS"] == "0" and not cfg["CKPT"]:
            raise ValueError("Set CKPT when RANDOM_WEIGHTS=0")
        if cfg["RANDOM_WEIGHTS"] == "1" and cfg["CKPT"]:
            raise ValueError("Unset CKPT or set RANDOM_WEIGHTS=0")
    if model == "timesfm":
        if cfg["LEVEL"] not in {"core", "e2e"} or cfg["DTYPE"] not in {"float32", "bfloat16"}:
            raise ValueError("TimesFM requires LEVEL=core|e2e and DTYPE=float32|bfloat16")
        if cfg["LEVEL"] == "e2e" and cfg["DTYPE"] != "float32":
            raise ValueError("The e2e wrapper does not implement a bfloat16 switch")
        if cfg["CONTEXT_LEN"] % 32 or cfg["HORIZON_LEN"] % 128:
            raise ValueError("Use exact shapes: CONTEXT_LEN multiple of 32; HORIZON_LEN multiple of 128")
        if cfg["CONTEXT_LEN"] + cfg["HORIZON_LEN"] > 16384:
            raise ValueError("TimesFM context + horizon exceeds 16384")
        if cfg["RANDOM_WEIGHTS"] == "1" and cfg["CHECKPOINT_PATH"]:
            raise ValueError("Unset CHECKPOINT_PATH or set RANDOM_WEIGHTS=0")
    if model == "alphafold" and cfg["FLASH_ATTENTION_IMPLEMENTATION"] not in {"triton", "cudnn", "xla"}:
        raise ValueError("Invalid FLASH_ATTENTION_IMPLEMENTATION")
    return cfg


def flag_tokens(env):
    tokens = shlex.split(env.get("XLA_FLAGS", "")) + shlex.split(env.get("XLA_FLAGS_EXTRA", ""))
    # XLA flags normally use --name=value. Normalize the space-separated spelling too.
    normalized = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected XLA_FLAGS token: {token!r}")
        if "=" not in token and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            i += 1
            token += "=" + tokens[i]
        normalized.append(token)
        i += 1
    return normalized


def arm_flags(env, model, graphs, pdl):
    tokens = flag_tokens(env)
    if graphs and any(t == CG_KEY + "=" for t in tokens):
        raise ValueError("Inherited empty --xla_gpu_enable_command_buffer= disables graphs. "
                         "Remove it from XLA_FLAGS/XLA_FLAGS_EXTRA for the headline comparison.")
    removed = set(PDL_KEYS)
    if not graphs:
        removed.add(CG_KEY)
    if model == "alphafold":
        removed.add(LHS_KEY)
    tokens = [t for t in tokens if t.split("=", 1)[0] not in removed]
    if not graphs:
        tokens.append(CG_KEY + "=")
    if model == "alphafold":
        tokens.append(LHS_KEY + "=true")
    tokens.extend(f"{key}={'true' if pdl else 'false'}" for key in sorted(PDL_KEYS))
    # XLA's environment parser understands --key="value with spaces", but not
    # shell quoting of the WHOLE argument ('--key=value with spaces').
    encoded = []
    for token in tokens:
        key, equals, value = token.partition("=")
        if equals and any(c.isspace() or c in "\"'\\" for c in value):
            if "\n" in value or "\r" in value:
                raise ValueError("Newlines are not supported in XLA flag values")
            encoded.append(key + "=" + json.dumps(value, ensure_ascii=False))
        else:
            encoded.append(token)
    return " ".join(encoded)


def visible_devices(model, cfg, env):
    count = cfg["TP"] * cfg["FSDP"] if model == "gemma" else 1
    default = ",".join(map(str, range(count)))
    devices = env.get("CUDA_VISIBLE_DEVICES", env.get("GPU", default)).split(",")
    devices = [item.strip() for item in devices]
    if len(devices) != count or any(not x or x == "-1" for x in devices) or len(set(devices)) != count:
        raise ValueError(f"Expected exactly {count} distinct visible GPU IDs; got {devices}")
    return ",".join(devices)


def model_command(model, mode, cfg, directory, python):
    raw = directory / "raw.json"
    tracing = mode == "xprof"
    iterations = cfg["PROFILE_ITERS"] if mode != "benchmark" else cfg.get("ITERS", cfg.get("WARM_RUNS"))
    if model == "gemma":
        # Quote the YAML empty string: a bare profiler= can become None.
        settings = {
            "hardware": "gpu", "model_name": cfg["MODEL_NAME"], "tokenizer_path": cfg["TOKENIZER_PATH"],
            "weight_dtype": cfg["WEIGHT_DTYPE"], "per_device_batch_size": cfg["PER_DEVICE_BS"],
            "max_prefill_predict_length": cfg["PREFILL_LEN"], "max_target_length": cfg["TARGET_LEN"],
            "ici_tensor_parallelism": cfg["TP"], "ici_fsdp_parallelism": cfg["FSDP"],
            "scan_layers": "false", "attention": cfg["ATTENTION"],
            "inference_microbenchmark_stages": "generate",
            "inference_microbenchmark_prefill_lengths": f"{cfg['PREFILL_LEN']},{cfg['PREFILL_LEN']}",
            "inference_microbenchmark_loop_iters": iterations,
            "inference_microbenchmark_log_file_path": str(raw),
            "run_name": directory.name, "base_output_directory": str(directory / "xprof"),
            "profiler": "xplane" if tracing else "''",
        }
        if cfg["RANDOM_WEIGHTS"] == "0":
            settings["load_parameters_path"] = cfg["CKPT"]
        if tracing:
            settings.update(skip_first_n_steps_for_profiler=3, profiler_steps=5)
        return [python, "-m", "maxtext.inference.inference_microbenchmark", cfg["BASE_CONFIG"],
                *[f"{key}={value}" for key, value in settings.items()]]
    if model == "timesfm":
        command = [python, str(ROOT / "timesfm/bench_timesfm.py"),
                   "--batch-size", str(cfg["BATCH_SIZE"]), "--context-len", str(cfg["CONTEXT_LEN"]),
                   "--horizon-len", str(cfg["HORIZON_LEN"]), "--iters", str(iterations),
                   "--warmup", str(cfg["WARMUP"]), "--level", cfg["LEVEL"], "--dtype", cfg["DTYPE"],
                   "--output-json", str(raw), "--label", directory.name]
        if cfg["RANDOM_WEIGHTS"] == "1":
            command.append("--random-weights")
        elif cfg["CHECKPOINT_PATH"]:
            command.extend(["--checkpoint-path", cfg["CHECKPOINT_PATH"]])
        else:
            command.extend(["--model-id", cfg["MODEL_ID"]])
    else:
        command = [python, str(ROOT / "alphafold/af3_inference.py"),
                   "--input-json-path", cfg["INPUT_JSON"], "--model-dir", cfg["MODEL_DIR"],
                   "--warm-runs", str(iterations), "--num-recycles", str(cfg["NUM_RECYCLES"]),
                   "--num-diffusion-samples", str(cfg["NUM_DIFFUSION_SAMPLES"]),
                   "--flash-attention-implementation", cfg["FLASH_ATTENTION_IMPLEMENTATION"],
                   "--output-json", str(raw)]
    if tracing:
        command.extend(["--profile-dir", str(directory / "xprof"), "--profile-runs", str(iterations)])
    return command


def make_plan(args, env):
    cfg = config(args.model, env)
    if args.model == "alphafold" and args.mode == "nsys" and "PROFILE_ITERS" not in env:
        # Keep a following invocation for start-to-next-start window selection.
        cfg["PROFILE_ITERS"] = 3
    devices = visible_devices(args.model, cfg, env)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = Path(args.out_dir or env.get("OUT_DIR") or env.get("OUTDIR") or
               ROOT / "results" / args.model / args.mode / stamp).expanduser().resolve()
    python = env.get("PYTHON", sys.executable)
    jobs = []
    for repeat in range(1, args.repeats + 1):
        pdl_order = [False, True] if args.order == "off-on" else [True, False]
        if repeat % 2 == 0:
            pdl_order.reverse()
        for graphs in ([True, False] if args.matrix else [True]):
            for pdl in pdl_order:
                arm = f"cg_{'on' if graphs else 'off'}_pdl_{'on' if pdl else 'off'}"
                directory = out / f"run{repeat:02d}" / arm
                command = model_command(args.model, args.mode, cfg, directory, python)
                if args.mode == "nsys":
                    options = ["--trace=cuda,nvtx", "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none"]
                    if args.nsys_tool == "nsys":
                        command = ["nsys", "profile", *options, "--output=" + str(directory / "trace"), *command]
                    else:
                        command = ["nsys-jax", *options, "--python-sampling=false", "-o",
                                   str(directory / "trace"), "--", *command]
                jobs.append(dict(repeat=repeat, arm=arm, graphs=graphs, pdl=pdl,
                                 directory=str(directory), raw_json=str(directory / "raw.json"),
                                 command=command, status="pending", env={
                                     "XLA_FLAGS": arm_flags(env, args.model, graphs, pdl),
                                     "CUDA_VISIBLE_DEVICES": devices,
                                     "JAX_ENABLE_COMPILATION_CACHE": "false",
                                     "JAX_COMPILATION_CACHE_DIR": "",
                                     "PDL_CAPTURE_MODE": args.mode,
                                     "PYTHONUNBUFFERED": "1",
                                 }))
    return dict(schema_version=1, created_utc=stamp, model=args.model, mode=args.mode,
                profiled=args.mode != "benchmark", out_dir=str(out), settings=cfg,
                python=python, hostname=platform.node(), container_image=env.get("PDL_IMAGE_PINNED"),
                inherited_xla_flags=env.get("XLA_FLAGS", ""), extra_xla_flags=env.get("XLA_FLAGS_EXTRA", ""),
                jobs=jobs)


def preflight(plan):
    if Path(plan["out_dir"]).exists():
        raise ValueError("Output directory already exists; choose a new --out-dir (no files overwritten)")
    for executable in {plan["python"], *[job["command"][0] for job in plan["jobs"]]}:
        if not shutil.which(executable):
            raise ValueError(f"Executable not found: {executable}")
    keys = {"gemma": ("BASE_CONFIG", "TOKENIZER_PATH"), "timesfm": ("CHECKPOINT_PATH",),
            "alphafold": ("INPUT_JSON", "MODEL_DIR")}[plan["model"]]
    for key in keys:
        value = plan["settings"][key]
        if value and not Path(value).exists():
            raise ValueError(f"{key} does not exist: {value}. See README.md for container prerequisites.")
    # MaxText's M_* environment overrides can silently replace CLI settings.
    overrides = sorted(k for k in os.environ if k.startswith("M_"))
    if plan["model"] == "gemma" and overrides:
        raise ValueError(f"Unset MaxText environment overrides before A/B: {', '.join(overrides)}")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def record_setup(plan):
    directory = Path(plan["out_dir"])
    source_paths = sorted(ROOT.glob("*.py")) + sorted(ROOT.glob("*/*.py")) + sorted(ROOT.glob("*/*.sh"))
    plan["script_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    plan["input_sha256"] = {}
    for key in ("BASE_CONFIG", "TOKENIZER_PATH", "INPUT_JSON"):
        path = plan["settings"].get(key)
        if path and Path(path).is_file():
            plan["input_sha256"][key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    info_code = (
        "import importlib.metadata as m, json, sys; "
        "names={'jax','jaxlib','flax','timesfm','maxtext','dm-haiku','tokamax'}; "
        "print(json.dumps({'python':sys.version, 'packages': "
        "{d.metadata['Name']:d.version for d in m.distributions() "
        "if d.metadata['Name'].lower() in names}}, indent=2))"
    )
    checks = [("python_packages.txt", [plan["python"], "-c", info_code]),
              ("nvidia-smi.txt", ["nvidia-smi", "-q"])]
    if plan["mode"] == "nsys":
        checks.append(("nsys_version.txt", ["nsys", "--version"]))
    for filename, command in checks:
        try:
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=30, check=False)
            output = f"exit_code={result.returncode}\n{result.stdout}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            output = f"Unavailable: {exc}\n"
        (directory / filename).write_text(output)


def run_logged(command, env, log_path):
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def execute(plan, env):
    preflight(plan)
    out = Path(plan["out_dir"])
    out.mkdir(parents=True, exist_ok=False)
    manifest = out / "manifest.json"
    record_setup(plan)
    write_json(manifest, plan)
    for job in plan["jobs"]:
        directory = Path(job["directory"])
        directory.mkdir(parents=True, exist_ok=False)
        job["status"] = "running"
        write_json(manifest, plan)
        print(f"\n[{job['arm']}, repeat {job['repeat']}] {shlex.join(job['command'])}", flush=True)
        try:
            job["exit_code"] = run_logged(job["command"], {**env, **job["env"]}, directory / "run.log")
            if job["exit_code"]:
                raise RuntimeError(f"Model process exited with {job['exit_code']}; see {directory / 'run.log'}")
            if not Path(job["raw_json"]).is_file():
                raise RuntimeError("Model did not write raw.json; check compatibility with this container")
            if plan["mode"] == "xprof" and not any(directory.rglob("*.xplane.pb")):
                raise RuntimeError("No .xplane.pb found; check model profiler support and run.log")
            if plan["mode"] == "nsys" and not any(
                directory.glob("trace.nsys-rep" if job["command"][0] == "nsys" else "trace.zip")
            ):
                raise RuntimeError("Nsight did not produce its report/archive; inspect run.log")
            job["status"] = "complete"
        except BaseException as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
            raise
        finally:
            write_json(manifest, plan)
    if plan["mode"] == "benchmark":
        from compare_results import summarize, render
        summary = summarize(manifest)
        write_json(out / "summary.json", summary)
        print(render(summary))
    else:
        print("Profile capture complete. Profiled timings are NOT benchmark results.")
    print(f"Saved experiment: {out}")


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = make_plan(args, os.environ)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
        else:
            execute(plan, os.environ)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
