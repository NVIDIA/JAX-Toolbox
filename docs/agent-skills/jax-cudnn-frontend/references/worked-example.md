# Worked example: Block-Sparse Attention (BSA) forward + backward

How the skill's phases were applied to a real kernel family. **Everything in
`[EX]` brackets is example-specific to cudnn-frontend 1.27.0 +
nvidia-cutlass-dsl 4.6.1/4.7.0 and must be re-derived for any other version.
Unbracketed statements are the general rules being illustrated.**

## Phase 1 — Environment

Fingerprinted per environment-discovery.md. Decision made from compute
capability: `[EX: cc 9.0 → sm90_blk64 classes; cc 10.0 → sm100_blk64
classes]`. The GPU marketing name misled once — a "GB200" node reports B200
GPUs at cc 10.0, so the SM100 (not a hypothetical SM120) class applied.

## Phase 2 — Rosetta stone

Grepping the installed package for the kernel class name found
`[EX: cudnn/block_sparse_attention/_interface.py]` — the vendor's
orchestration. This single file yielded:

- the true output shape of the auxiliary `lse` tensor —
  `[EX: torch.empty((batch, num_heads, seqlen_q)) — per-TOKEN]` — directly
  contradicting stale docs that described it per-block. *General rule:
  vendor allocations are the buffer contract.* Missing this caused a 64×
  buffer overflow whose fallout (illegal addresses on re-execution, corrupted
  metadata, phantom platform bugs on three machines) consumed the majority of
  all debugging time.
- exact positional call order per arch `[EX: SM90 places scale after the
  sparse-metadata tensors; SM100 places it before them]`;
- the workspace formula and its zeroing semantics `[EX: per (B,H):
  2·q_r + q_r·d + 2·k_r·d fp32, accumulator tail zeroed on the flattened
  view; rounding differs per arch]`;
- a flag coupling `[EX: is_persistent = use_clc_scheduler]` — which predicted
  (correctly) that the decoupled combination was never vendor-tested. The
  decoupled default silently dropped ~40% of tiles past one CTA wave: found
  by the coverage diagnostic, confirmed by toggling one env var per run, and
  filed upstream.

## Phase 3 — Contract highlights

- Layout: the kernel asserted `[EX: stride==1 at position 1 for Q/K/O and
  position 0 for V]`. The signature *comment* described a different dim
  order than the remap code enforced — the code won. Satisfied with
  zero-copy mode permutations `[EX: TensorSpec(mode=(2,3,1,0)) for Q/K/O]`;
  no physical relayout needed because the contiguous dim was already right.
- Optionals: `[EX: blocksparse_num_blocks_q2k]` is `Optional` on one arch,
  required (indexed unconditionally) on another → trace-time
  `'NoneType' object is not subscriptable`. Built the real tensor with the
  shape implied by the kernel's indexing expression.
- Arch constraints from asserts: `[EX: bf16 only; head_dim=128 for blk64
  paths, forward and backward]`.

## Phases 4–6 — Integration

- Both invocation paths kept alive throughout. The A/B earned its keep
  twice: `[EX: proving a persistent-scheduler bug was invocation-independent
  (kernel bug → upstream report), and proving a CSR-builder kernel was
  correct under cute.compile but silently empty under cutlass_call
  (bridge aliasing bug in 4.6, confirmed by the bridge team as known and
  fixed in 4.7.1)]`.
- Zero-init handled with a prologue `@cute.kernel` writing zeros inside the
  launcher — chosen over `input_output_aliases` after the aliasing bug;
  works on all bridge versions and keeps the whole pipeline jittable.
- Backward = its own full contract pass: a separate kernel per arch plus an
  auxiliary index-inversion kernel and a workspace, wired under
  `jax.custom_vjp`. Nothing about it was inferable from the forward.

## Phase 7 — Validation that caught real defects

- Dense-masked reference (host-side numpy when GPU-side references became
  suspect) — caught wrong-values states that shape checks blessed.
- Two independent references cross-validated to 5e-7 — settled a
  kernel-vs-reference dispute in the reference's favor.
- Analytic invariant `sum(dV) == sum(dOut) == 1.0` — free exact check on the
  backward, now asserted in the shipped artifact.
- `jnp.empty` + NaN coverage map — turned "output has NaN" into "the
  scheduler stops issuing tiles after the first wave".
- Size scaling: every defect above was invisible at the small smoke-test
  size and appeared only past one hardware wave of work.
- Repeated execution (N≥3 + fresh-process reruns with checksum comparison) —
  exposed the buffer-overflow class that single-shot tests blessed.

## Final artifact shape (general pattern)

1. A standalone repro/validation script: env-var toggles for every config
   axis, per-stage checksums, NaN counts, fixed-position samples, hard
   asserts. It served as bisection harness, regression test, and the body of
   two upstream bug reports without modification.
2. The integration notebook/library: forward + custom_vjp backward, one
   analytic invariant asserted inline, version-sensitive workarounds
   commented with the version they apply to.
3. Bug reports for anything isolated to the vendor, each carrying: exact
   environment fingerprint, PASS/FAIL command matrix, isolation performed,
   and the sanitizer-blind-spot caveat.
