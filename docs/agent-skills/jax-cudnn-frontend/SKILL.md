---
name: jax-cudnn-frontend
description: Implement, integrate, debug, and validate cuDNN Frontend / CuTe DSL kernels from pure JAX (no PyTorch). Use when asked to wrap, call, port, or autotune a cudnn-frontend kernel — attention variants, GEMM fusions, or experimental csrc kernel classes — from JAX. Discovery-first — derives APIs and tensor contracts from the installed environment instead of trusting memory or online examples.
---

# cuDNN Frontend kernels from pure JAX

You are integrating experimental GPU kernels whose APIs, tensor contracts, and
supported configurations change between package versions. **Treat everything
you think you know about these APIs as stale.** Every claim about a signature,
shape, dtype, layout, or flag must be re-derived from the installed
environment before you build on it. This skill is the procedure for doing
that, plus the failure modes that cost days when the procedure was skipped.

## Non-negotiable rules

1. **Never call a kernel whose contract you have not read from installed
   source.** Kernel names, docs pages, GitHub examples, and your prior
   knowledge describe *some* version — not necessarily the installed one.
2. **Buffer shapes come from the vendor's own allocations**, not from
   signature comments or docs. Find where the installed package allocates
   each buffer before calling the kernel itself (see rule of the Rosetta
   stone, Phase 2). An under-allocated output is a silent buffer overflow
   that corrupts *neighboring* tensors and surfaces as unrelated crashes
   later — memcheck cannot see it.
3. **Compilation and execution are not correctness.** A kernel can launch,
   return, and pass `block_until_ready()` while producing garbage, writing
   nothing, or corrupting memory that only a later op trips over. Only value
   checks against an independent reference count as passing.
4. **No PyTorch** in the implementation path unless the user asks for it.
   PyTorch is permitted only as an optional cross-check oracle, clearly
   labeled.
5. **Fresh process per meaningful measurement.** After any illegal memory
   access or suspect result, the CUDA context and all session state are
   untrustworthy — including results that *look* fine.
6. **One variable per experiment.** Build an env-var-toggled repro script
   early (Phase 5) and bisect with it; never change two things between runs.

## Workflow

### Phase 1 — Environment discovery
Run `scripts/inspect_environment.py`. Record: JAX/jaxlib/plugin versions and
the source-commit suffix, cudnn-frontend and cutlass-dsl versions, CUDA
toolkit, GPU name + compute capability + driver, and whether the target
module/kernel is importable. Details and container-identity pitfalls (rolling
tags, arm64 lag): `references/environment-discovery.md`.

**Exit criteria:** a pasted environment block in your working notes, and a
confirmed compute capability — kernel availability and *which kernel class
you must use* are usually arch-gated (e.g. separate `sm90_*` / `sm100_*`
classes with different constructors and argument orders).

### Phase 2 — Locate the kernel and its Rosetta stone
Find (a) the kernel class/function in the installed package and (b) **the
vendor's own orchestration code that calls it** — typically an
`_interface.py`, `api.py`, wrapper, or test inside the installed package.
That call site is the single most valuable artifact you will find: it shows
the true argument order, every buffer allocation with exact shapes and
dtypes, workspace construction, layout transforms, and flag couplings the
constructor won't tell you about. Use `scripts/contract_report.py` to dump
signatures and call sites automatically. Procedure and grep recipes:
`references/contract-discovery.md`.

### Phase 3 — Extract the data contract
Fill in the full checklist in `references/contract-discovery.md` (inputs,
outputs, shapes, dtypes, layout/stride requirements, alignment, scalars,
workspace, zero-init requirements, aliasing, optional-vs-required per arch,
arch constraints, coupled flags). Evidence hierarchy when sources disagree:

1. **Executable code in the installed package** — assertions in the kernel
   body (`assert`, `check_dim`, dtype checks) and the orchestration's buffer
   allocations. These are the contract.
2. Tests/examples shipped *inside the installed package*.
3. Upstream source at the exact installed version (match the commit).
4. Version-labeled docs.
5. General docs, online examples, and your prior knowledge — hypothesis
   generators only, never evidence.

When (1) contradicts anything else — including signature *comments* in the
same file — (1) wins. Record the discrepancy; it often marks an API
transition and predicts other differences. When two pieces of executable code
disagree (wrapper vs kernel), the kernel body governs the kernel's tensor
contract and the wrapper governs orchestration (buffer shapes, call order).

### Phase 4 — Choose and verify the integration mechanism
Discover what the installed bridge actually provides — do not assume:
introspect `cutlass.jax` exports and read the installed bridge source for the
launcher convention, output allocation, aliasing semantics, and spec types.
Keep **two invocation paths** implemented throughout the project: the
JAX-native path (e.g. `cutlass_call`) *and* the direct path (e.g.
`cute.compile`). The A/B between them is your primary tool for separating
bridge bugs from kernel bugs from your bugs. Mechanics, conventions, and
version-sensitive behaviors to verify: `references/jax-integration.md`.

### Phase 5 — Minimal standalone proof
Before any notebook or abstraction: a single plain-Python script that runs
the kernel once with fixed seeds, prints **value-based** evidence (NaN
counts, checksums, fixed-position samples), and exposes every configuration
choice as an env-var toggle. This script is simultaneously your repro for
bug reports and your bisection harness. Where you own the output buffers
(direct path), prefill them with a NaN sentinel — `jnp.full(shape, jnp.nan,
dtype)` — so unwritten regions are detectable; do **not** use `jnp.empty`,
which can return arbitrary uninitialized bits that evade `jnp.isnan`
(bridge-path variant: validation.md Gate 3). Then run
it **N≥3 iterations** in one process (repeated execution is where buffer and
lifetime bugs hide) and **twice in fresh processes** (identical checksums;
drift is a red flag).

### Phase 6 — JAX wrapper
Only after Phase 5 passes: wrap with the JAX-native mechanism, jit it,
confirm the jaxpr contains a custom call, and re-run the Phase 5 checks
through the wrapper — outputs should be bit-identical to the direct path.
Autodiff is a separate deliverable: `jax.custom_vjp` with the corresponding
backward kernel, whose contract gets its own full Phase 2–5 pass. Do not
assume the backward's buffers mirror the forward's.

### Phase 7 — Validation gate
No result is "working" until it passes the gates in
`references/validation.md`: independent mathematical reference (host-side if
GPU-side references share failure modes with the kernel), analytic
invariants, repeated-execution stability, size scaling beyond the toy config
(some bugs only appear past hardware occupancy thresholds — test a size
where total tiles exceed one CTA wave), and — for anything exposing config
knobs or feeding an autotuner — a **per-configuration correctness check**,
because invalid configs can be silently wrong *and faster*.

### Phase 8 — Debugging
Symptom-indexed decision trees from real failures:
`references/debugging.md`. Headline discipline: crashes surface at innocent
*later* operations (async execution), so the faulting op named in a traceback
is usually the victim, not the culprit. Establish the earliest corrupted
artifact, then work backwards.

## Agent traps (each of these was committed or nearly committed)

- Copying an invocation from docs/examples without version-matching it.
- Inferring a tensor's shape from its name (`lse` ≠ "per-block" just because
  the op is block-sparse — it was per-token).
- Trusting a signature comment over the code three lines below it.
- Assuming a Python wrapper's simplified API equals the kernel's contract.
- Assuming the kernel exists / behaves the same on every GPU arch.
- Assuming default constructor arguments are valid combinations (a default
  flag pair produced silent 40%-wrong output).
- Treating a clean `compute-sanitizer` run as proof of memory safety (it is
  blind to overflows into valid neighboring allocations and to unwritten
  output).
- Treating `block_until_ready()` or printed shapes as correctness.
- Reading device data with `np.asarray` twice (JAX caches the first host
  copy; use `jnp.copy(x)` to force a fresh device read).
- Testing only at small sizes (waves-of-work bugs hide below occupancy).
- Blaming the platform/container/driver before exhausting caller-side bugs —
  and, symmetrically, burning days on caller-side theories without A/B-ing
  the invocation paths.
- Letting a rolling container tag define your environment.
- Silently importing torch because the vendor's public API takes torch
  tensors — read one level deeper; the kernel classes underneath are
  framework-agnostic via DLPack.

## Design notes

**Not hard-coded, on purpose:** concrete signatures, argument orders, buffer
shapes, spec/aliasing semantics, flag names, and version numbers. All of
these changed at least once during the work this skill distills, sometimes
between adjacent minor versions. Where the references show concrete code, it
is labeled *Example (version-specific)* and paired with the discovery step
that regenerates it for any version.

**How this adapts to change:** every phase's output is derived from the
installed artifacts (signatures, assertions, vendor allocations, bridge
source), so a renamed argument or reshaped buffer changes the *result* of the
procedure, not the procedure. The scripts print what exists rather than
checking against expectations.

**Still requires human judgment:** deciding whether an observed misbehavior
is a vendor bug worth reporting versus an undocumented contract you must
satisfy; choosing performance configs after correctness; prioritizing which
arch/version combinations to validate; and anything requiring contact with
kernel owners (known-issue status, backports, roadmap).
