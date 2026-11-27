# jax-cudnn-frontend

An agent skill for implementing, integrating, debugging, and validating
[cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend) / CuTe DSL kernels
from **pure JAX** — no PyTorch in the implementation path.

The supported cudnn-frontend API surface for these kernels is PyTorch-based,
but the kernels themselves are framework-agnostic CuTe DSL classes. Calling
them from JAX means working against undocumented internals whose signatures,
tensor contracts, and supported flag combinations change between package
versions. This skill teaches an agent to **re-derive every contract from the
installed environment** instead of trusting memory, docs, or online examples —
and to validate with value checks, not just "it compiled and ran".

## What the skill covers

- Environment fingerprinting (versions, compute capability, arch-gated kernel
  classes, container-identity pitfalls).
- Contract discovery: finding the vendor's own call site ("Rosetta stone")
  inside the installed package and extracting buffer shapes, argument order,
  workspace layout, layout/stride requirements, and flag couplings from it.
- JAX integration via both available paths — the `cutlass.jax.cutlass_call`
  bridge and direct `cute.compile` — kept alive in parallel so failures can be
  A/B-isolated to the kernel, the bridge, or the integration.
- Zero-initialization, aliasing, `custom_vjp` backward wiring, and
  jit-compatible workarounds for known bridge limitations.
- Validation gates that catch defects compilation cannot: independent dense
  references, analytic invariants, NaN coverage maps, size scaling past one
  hardware wave, repeated-execution checksums.
- Symptom-indexed debugging decision trees for asynchronous-CUDA failure
  modes (misattributed faults, silent buffer overflows, sanitizer blind
  spots), and an escalation ladder for isolating kernel vs. bridge vs.
  user bugs — ending in a file-able repro script.

## Layout

| Path | Purpose |
|---|---|
| `SKILL.md` | The control plane: non-negotiable rules and the 8-phase workflow. Agents start here. |
| `references/environment-discovery.md` | Fingerprinting the environment; container/version pitfalls. |
| `references/contract-discovery.md` | Deriving a kernel's data contract from installed source. |
| `references/jax-integration.md` | Both invocation paths, bridge conventions, zero-init, backward wiring. |
| `references/validation.md` | Value-level validation gates. |
| `references/debugging.md` | Symptom-indexed decision trees. |
| `references/worked-example.md` | The phases applied to a real kernel family (block-sparse attention), version-specific details bracketed `[EX]`. |
| `scripts/inspect_environment.py` | Paste-ready environment fingerprint report. |
| `scripts/contract_report.py` | Dumps signatures, in-body constraints, and vendor call sites for a kernel class. |

## Requirements

- A CUDA GPU environment with `jax`, `nvidia-cudnn-frontend` (with the CuTe
  DSL kernels, i.e. the `csrc` kernel classes), and `nvidia-cutlass-dsl`
  installed. The NVIDIA JAX container is the reference environment.
- No repository clones are needed — the workflow introspects installed
  packages (see `references/contract-discovery.md`).

## Usage

Place this directory wherever your coding agent discovers skills — e.g.
`.claude/skills/` or `~/.claude/skills/` for Claude Code, or the equivalent
location for your tool (`.cursor/`, `.codex/`, ...). Agents without a skill
mechanism can simply be pointed at `SKILL.md` as instructions — it is
self-contained. Then invoke the skill explicitly or let it trigger on
matching tasks:

> Using the jax-cudnn-frontend skill: wrap the cuDNN Frontend CuTe DSL
> block sparse attention kernels for inference and training from pure JAX,
> exposing the kernel configuration knobs so we can autotune per problem
> shape.

The expected output of a run is (1) a standalone, env-var-toggled
repro/validation script, (2) the integration library or notebook with a
`custom_vjp` backward and inline invariant checks, and (3) bug reports for
anything isolated to the vendor.

## Scope and caveats

- The skill encodes *procedure*, not APIs: everything version-specific in the
  worked example is bracketed `[EX]` and must be re-derived per environment.
- Kernel classes under `csrc` are experimental and unsupported; contracts
  here may change without notice between cudnn-frontend releases.
