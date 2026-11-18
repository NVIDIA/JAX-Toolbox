# Validation

Correctness gates. A kernel integration is not done until every applicable
gate passes. The source experience contains multiple runs that "passed" by
weaker standards and were later shown to be silently wrong.

## Gate 0 — Understand what does NOT count as evidence

- Printed shapes/dtypes: available before the kernel even executes (async).
- `block_until_ready()` returning: syncs a stream, does not check values, and
  may not even surface faults from work on other streams.
- Clean `compute-sanitizer --tool memcheck`: blind to (a) writes that land in
  *valid neighboring allocations* (buffer overflow between live tensors) and
  (b) output regions the kernel *failed to write*.
- Loss "going down" alone: verify gradient values first; a training loop can
  descend on partially-wrong gradients.

## Gate 1 — Independent mathematical reference

Implement the operation independently (dense/masked formulation, or a
gather-based one) and compare element-wise.

- **Two independent references beat one.** Cross-validate them against each
  other on CPU first (they agreed to ~5e-7 in the source work, which then
  exonerated the reference during a kernel-vs-reference dispute).
- **Run the reference on the host (numpy)** when the GPU is under suspicion —
  a GPU-side reference shares allocators, streams, and library bugs with the
  thing you're testing.
- Numerical hygiene: mask with a large finite value (`-1e30`), not `-inf`
  (softmax-of-`-inf` NaN semantics differ across versions/backends).
- Tolerances: bf16 inputs with fp32 accumulation → expect max-abs error
  around 1e-3–1e-2 relative to output scale. Compare in fp32. An error of
  ~0.3 on O(1) outputs is not "loose tolerance", it is *wrong-values* —
  typically a wrong attended set / permuted metadata, not precision.

## Gate 2 — Analytic invariants

Cheap, exact, and independent of any reference implementation. Examples that
caught / confirmed real behavior:
- softmax rows sum to 1 ⇒ for attention backward, `sum(dV) == sum(dOut)`
  exactly (up to accumulation rounding);
- uniform inputs ⇒ known closed-form outputs;
- conservation-style checksums that must match across configurations.
Assert at least one such invariant in the shipped artifact, not just during
development.

## Gate 3 — Coverage diagnostic

Prefill outputs with a NaN sentinel during testing so unwritten regions are
detectable, then map defects to work units. Do **not** use `jnp.empty` for
this: it may return genuinely uninitialized memory (documented from
JAX 0.11, and never guaranteed to be NaN before that), so an unwritten tile
can contain any finite value and evade `jnp.isnan`.

- **Direct path** (you allocate the buffers): `out = jnp.full(shape,
  jnp.nan, dtype)`. Integer outputs need a finite sentinel instead — e.g.
  `jnp.full(shape, -1, jnp.int32)`, then check for surviving `-1`s.
- **Bridge path** (`cutlass_call`-style, where the bridge allocates outputs
  uninitialized): the caller cannot prefill. In diagnostic builds, add a
  sentinel-fill prologue kernel inside the launcher — the same pattern as
  the zero-prologue in jax-integration.md, writing NaN instead of zero — or
  run this gate on the direct path, where the coverage finding transfers.

```python
out = jnp.full((B, H, S, D), jnp.nan, jnp.float32)  # sentinel prefill
# ... kernel writes into out ...
bad = jnp.isnan(out.astype(jnp.float32))
per_tile = bad.reshape(B, H, n_tiles, tile, D).any(axis=(0, 1, 3, 4))
```

The *pattern* is diagnostic: structured tail-of-work NaNs ⇒ scheduler/config
dropping tiles past an occupancy boundary; scattered ⇒ corruption; everything
⇒ kernel never ran / wrong buffers.

## Gate 4 — Repeated execution and determinism

- Execute the compiled op **N≥3 times in one process**; compare checksums and
  fixed-position samples across iterations. First-call-only correctness is a
  known real failure mode (buffer lifetime/overflow bugs).
- Run the whole script twice in **fresh processes**: identical checksums
  expected for fixed seeds. Cross-run drift ⇒ unwritten memory or a race.
- For training integrations, the loop itself is a repeated-execution test —
  but only if gradient values were independently verified first (Gate 1/2).

## Gate 5 — Size scaling

Test at least one size where total work exceeds one hardware wave
(`work_items > SM count`), one where it doesn't, and your target size. A
scheduler bug in the source experience was **invisible at 128 tiles and
dropped 40% of output at 512 tiles** — every small smoke test passed.

## Gate 6 — Cross-path A/B

Same kernel, same inputs, through the JAX-native bridge and the direct
`cute.compile`-style path: outputs must be bit-identical (same kernel binary,
same math). Any difference is an integration bug by construction.

## Gate 7 — Config-space correctness (autotuning)

If configuration knobs are exposed (tile sizes, scheduler flags, splits):
run Gate 1 or Gate 2 **per configuration**, not just per kernel. Silently
wrong configs exist and may be faster than correct ones — a latency-only
autotuner will select them. Reject or quarantine any config that fails
values before it is timed.

## Gate 8 — Invalid-input behavior (cheap, optional)

Feed one deliberately wrong input (bad dtype, undersized metadata) and note
whether the stack raises or silently proceeds. If it silently proceeds,
raise your own validation into the wrapper — you now know the kernel won't
protect users.

## Session hygiene during all gates

- Fresh kernel/process after *any* illegal-access error — later results in a
  poisoned context are meaningless, including apparently clean ones.
- Never re-run measurement cells in a notebook and trust the result;
  notebooks re-executing definition cells recompile kernels and historically
  produced phantom failures *and* phantom successes. Plain scripts, fresh
  processes, env-var toggles.
