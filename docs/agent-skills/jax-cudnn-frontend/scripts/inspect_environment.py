#!/usr/bin/env python3
"""Environment fingerprint for cuDNN Frontend / CuTe DSL + JAX work.

Prints a paste-ready report: package versions, CUDA toolkit, GPU
architecture, JAX backend, the cudnn-frontend surface, and the JAX-bridge
surface. Everything is best-effort: missing components are reported, never
fatal.

Usage:
    python inspect_environment.py            # full report (initializes CUDA via jax)
    python inspect_environment.py --no-jax   # skip jax import (no CUDA context)
"""
import importlib
import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"


def pkg_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def section(title):
    print(f"\n== {title} " + "=" * max(0, 60 - len(title)))


def main():
    no_jax = "--no-jax" in sys.argv

    section("host")
    print(f"python   : {platform.python_version()}  ({sys.executable})")
    print(f"platform : {platform.platform()}  arch={platform.machine()}")

    section("packages (pip metadata)")
    interesting = [
        "jax", "jaxlib",
        "jax-cuda12-plugin", "jax-cuda12-pjrt",
        "jax-cuda13-plugin", "jax-cuda13-pjrt",
        "nvidia-cudnn-frontend", "nvidia-cutlass-dsl",
        "nvidia-cublas", "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12", "nvidia-cudnn-cu13",
    ]
    for name in interesting:
        v = pkg_version(name)
        if v:
            print(f"{name:24s} {v}")
    missing = [n for n in ("jax", "nvidia-cudnn-frontend", "nvidia-cutlass-dsl")
               if not pkg_version(n)]
    if missing:
        print(f"MISSING (required for this workflow): {missing}")

    section("CUDA toolkit / driver / GPU")
    if shutil.which("nvcc"):
        print("nvcc     :", sh("nvcc --version | tail -1"))
    else:
        print("nvcc     : not on PATH")
    if shutil.which("nvidia-smi"):
        out = sh("nvidia-smi --query-gpu=name,compute_cap,driver_version"
                 " --format=csv,noheader")
        for i, line in enumerate(out.splitlines()):
            print(f"GPU[{i}]   : {line}")
        caps = {l.split(",")[1].strip() for l in out.splitlines() if "," in l}
        if caps:
            major = sorted(int(c.split(".")[0]) for c in caps)[0]
            print(f"compute capability major: {major}  "
                  f"(kernel classes are usually arch-gated: look for sm{major}0_* dirs)")
    else:
        print("nvidia-smi: not on PATH")

    section("jax runtime")
    if no_jax:
        print("(skipped: --no-jax)")
    else:
        try:
            import jax  # noqa: PLC0415
            print(f"jax.__version__ : {jax.__version__}")
            print(f"  NOTE: the '+<hash>' suffix is the only stable identifier for")
            print(f"        dev/nightly builds; cite it, not container tags.")
            print(f"jax.__file__    : {jax.__file__}")
            try:
                print(f"devices         : {jax.devices()}")
            except Exception as e:  # noqa: BLE001
                print(f"devices         : <error: {e}>")
        except Exception as e:  # noqa: BLE001
            print(f"import jax failed: {e}")

    section("cudnn-frontend surface")
    try:
        cudnn = importlib.import_module("cudnn")
        print(f"cudnn.__file__  : {getattr(cudnn, '__file__', '?')}")
        names = [a for a in dir(cudnn) if not a.startswith("_")]
        print(f"top-level names : {names}")
        pkg_dir = os.path.dirname(getattr(cudnn, "__file__", "") or "")
        if pkg_dir:
            subs = sorted(d for d in os.listdir(pkg_dir)
                          if os.path.isdir(os.path.join(pkg_dir, d))
                          and not d.startswith("_"))
            print(f"subpackages     : {subs}")
            # arch-gated kernel dirs are a strong signal of per-arch contracts
            hits = sh(f"find {pkg_dir} -maxdepth 4 -type d -name 'sm*' | head -20")
            if hits:
                print("arch-gated dirs :")
                for line in hits.splitlines():
                    print(f"  {line}")
    except Exception as e:  # noqa: BLE001
        print(f"import cudnn failed: {e}")

    section("JAX bridge surface (cutlass.jax)")
    try:
        cjax = importlib.import_module("cutlass.jax")
        names = [a for a in dir(cjax) if not a.startswith("_")]
        print(f"exports         : {names}")
        import inspect  # noqa: PLC0415
        for cand in ("cutlass_call",):
            fn = getattr(cjax, cand, None)
            if fn is not None:
                try:
                    print(f"{cand}{inspect.signature(fn)}")
                except (TypeError, ValueError):
                    print(f"{cand}: <signature unavailable>")
        for mod in ("cutlass.jax.primitive", "cutlass.jax.compile",
                    "cutlass.jax.types"):
            try:
                m = importlib.import_module(mod)
                print(f"{mod} -> {m.__file__}")
            except Exception:  # noqa: BLE001
                pass
        print("READ those source files: launcher convention, output allocation,")
        print("aliasing semantics, and spec types are version-specific.")
    except Exception as e:  # noqa: BLE001
        print(f"import cutlass.jax failed: {e}")

    section("advisories seen at import time")
    print("Watch process stderr on first GPU use for:")
    print(" - cuDNN runtime-vs-compiled version messages (record, usually benign)")
    print(" - cuBLAS known-issue warnings (e.g. TMEM concurrency <13.2 on Blackwell)")
    print(" - anything printing 'capture'/'graph' during faults (command buffers)")


if __name__ == "__main__":
    main()
