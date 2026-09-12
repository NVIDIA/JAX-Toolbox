# Environment discovery

Goal: an unambiguous fingerprint of the environment, sufficient to (a) select
the right kernel class for the hardware, (b) reproduce any finding, and (c)
detect when the environment shifts underneath you.

Run `scripts/inspect_environment.py` first; it automates most of this. What
follows explains what each field is *for* and the pitfalls.

## What to record and why

| Field | Command | Why it matters |
|---|---|---|
| JAX version incl. local suffix | `python -c "import jax; print(jax.__version__)"` | Dev builds like `0.11.1.dev20260803+c6ab31b9bf` — the `+hash` is the only stable identifier; two containers can differ by days of API churn |
| jaxlib / cuda plugins | `pip list \| grep -Ei "jaxlib\|jax-cuda"` | Plugin CUDA major (cu12/cu13) must match the stack |
| cudnn-frontend | `pip list \| grep cudnn-frontend` | Kernel APIs restructure between minors (e.g. class-based → functional between 1.26 and 1.27) |
| cutlass-dsl | `pip list \| grep cutlass-dsl` | The JAX bridge lives here; bridge bugs are version-specific (e.g. aliasing broken in 4.6, fixed 4.7.1) |
| CUDA toolkit | `nvcc --version \| tail -1` | Compile-time toolchain for DSL kernels |
| cuDNN runtime | `pip list \| grep nvidia-cudnn-cu` + watch startup logs | Runtime-vs-compiled mismatches print E-level noise; usually benign for DSL kernels but record it |
| GPU + capability + driver | `nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader` | **Kernel availability and class selection are gated on compute capability.** |
| GPU count | same command (one line per GPU) | Multi-GPU nodes: pin with `CUDA_VISIBLE_DEVICES` during debugging |

## Kernel presence check

Presence is not implied by the package being installed:

```python
import cudnn
print([a for a in dir(cudnn) if not a.startswith('_')])       # top-level surface
import cudnn.<subpackage>                                       # target module
print(dir(cudnn.<subpackage>))                                  # what it exports
```

If the expected name is missing, do **not** conclude it doesn't exist — the
API may have moved (search the installed tree, Phase 2) or be lazily loaded
behind `__getattr__`. Conversely, a module *existing* does not mean your GPU
supports it: look for arch-suffixed directories (`sm90_*`, `sm100_*`, …) and
arch assertions in the code.

## Container identity (the recurring trap)

- **Rolling tags** (`ghcr.io/nvidia/jax:jax`) move daily. Two pulls days
  apart are different environments; **arm64 builds can lag x86** by several
  days, so an x86 node and a Grace node pulled the same hour can carry
  different JAX commits. Multiple debugging sessions were wasted on this.
- Cite environments by the JAX `+commit` suffix, not the tag.
- Once an environment validates, freeze it:
  `--container-save=/path/name.sqsh` (enroot/pyxis) and use the `.sqsh` path
  thereafter. Docker: pin by digest.
- Env vars worth setting for work sessions:
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` (shared GPUs),
  `TF_CPP_MIN_LOG_LEVEL=3` (suppress XLA C++ log spam — set **before**
  importing jax).

## When two machines disagree

If the same code passes on machine A and fails on machine B, resist the two
easy stories ("broken platform", "broken container") until you have:

1. Fingerprinted both environments with the script and diffed them.
2. Reproduced the failure in a fresh process on B (session contamination
   mimics platform bugs).
3. Checked whether the failure is *size-dependent* rather than
   machine-dependent (different default sizes/occupancy across your runs).

In the source experience, an entire "preprod platform memory corruption"
narrative — garbage tensors, cross-machine flakiness, apparent container
regressions — was ultimately one caller-side buffer under-allocation. The
platform was innocent. Machines differing in *symptom* does not mean the
machine is the cause; buffer overflows land differently in different
allocator states.
