# Debugging decision trees

Symptom-indexed. Each tree is ordered: check the cheap/likely causes before
expensive theories. Overarching law: **CUDA execution is asynchronous — the
operation named in a crash is usually the victim, not the culprit.** Establish
the earliest corrupted artifact, then walk backwards.

## Universal first moves (any GPU fault)

1. Restart the process. Everything after an illegal access is untrustworthy.
2. Reproduce in the standalone script (fresh process), not the notebook.
3. Re-run with `CUDA_LAUNCH_BLOCKING=1` — serializes launches so errors are
   attributed to the actual faulting launch.
4. If still ambiguous: `compute-sanitizer --tool memcheck python repro.py`.
   Remember its blind spots: overflows into valid allocations and unwritten
   output are invisible; a clean run does NOT clear the kernel.

## Symptom: illegal memory access reported at an unrelated op (e.g. a reduce/convert kernel)

The named op tripped over corruption left by an earlier kernel.
1. **Check every output/workspace buffer size against the vendor's own
   allocations** (contract-discovery.md, Rosetta stone). The #1 historical
   cause: an output allocated smaller than the kernel writes → overflow into
   neighboring live buffers. Costed the most time of any bug class.
2. Check metadata tensors' *ranks/shapes* against what the kernel indexes
   (a kernel indexing `t[a, b, c]` needs rank ≥ 3 — a rank-1 tensor "works"
   under some compilers until it reads garbage).
3. Only then consider bridge/platform theories.

Audit tool — is a live buffer being clobbered?

```python
before = np.asarray(jnp.copy(suspect))         # fresh device read
run_kernel(...); out.block_until_ready()
after = np.asarray(jnp.copy(suspect))
print(np.array_equal(before, after))
# also compare device-side vs host-cached views:
print(int(jnp.max(suspect)))                    # device compute
```

If a supposedly-int tensor contains float bit patterns, reinterpret them
(`arr.view(np.float32)`) and fingerprint against live float tensors — a match
identifies exactly *which* buffer is being written over it.

## Symptom: first execution correct, second execution faults

1. First execution overflowed a buffer (see above) — sizes vs vendor
   allocations, again.
2. Bridge aliasing (`input_output_aliases`) bug in the installed version —
   A/B against the direct path; try the zero-prologue pattern instead of
   aliasing.
3. Borrowed DLPack memory freed by its owner — `.copy()` anything returned
   across a framework boundary.

## Symptom: outputs silently all-zero / empty, no error

1. Bridge aliasing machinery broken in installed version (a real, known bug
   in one release: aliased configurations returned empty outputs). A/B
   against the direct path immediately — if direct is correct, it's the
   bridge; check for a newer bridge release, use aliased-output-last or the
   zero-prologue workaround.
2. A buffer the kernel needed zero-initialized wasn't (counters read garbage
   / kernel exited early).
3. Kernel gated off for this arch/dtype and exiting silently.

## Symptom: structured NaN in part of the output; early regions fine

Tiles/work-units never written. Use the coverage map (validation.md Gate 3).
1. Scheduler/config flags: check the vendor wrapper for **coupled flags**
   (e.g. persistence paired with a companion scheduler). A constructor-default
   combination the vendor never exercises may drop all work past one hardware
   wave — test above and below `work_items == SM_count`.
2. Grid/tile mapping mismatch with your problem shape.

## Symptom: values uniformly wrong (finite, plausible magnitude, everywhere)

The kernel computed *something else* — usually attention/gather over the
wrong data.
1. Metadata view/permutation wrong (mode/layout specs) — kernel reads the
   right buffer with the wrong coordinate mapping.
2. Metadata buffer corrupted before the kernel read it (audit tool above;
   also re-upload fresh: `jnp.asarray(np.asarray(jnp.copy(meta)))`).
3. Your reference is wrong — cross-validate two independent references on
   CPU before convicting the kernel (this exact dispute happened; the
   reference won).

## Symptom: "dim must be contiguous in mode k" (or similar stride assert)

The kernel's layout contract. Read the assert AND the tensor-remapping lines
above it in `__call__` — the remap defines position semantics (comments may
disagree; trust the code). Fix preference:
1. Mode/logical permutation in the bridge spec (zero-copy) if the required
   dim is already stride-1 in row-major.
2. Physical relayout only if no permutation satisfies it.

## Symptom: `'NoneType' object is not subscriptable` during compile/trace

You passed `None` for a parameter that is optional in another arch's
signature but required (indexed) in this one. Build the real tensor with the
shape the kernel's indexing implies, cross-checked against the vendor call
site.

## Symptom: TypeError binding args at compile

Argument count/order mismatch with `__call__` — re-dump the signature for
*this* arch's class (they differ between arch variants, including scalar
placement). Also check: bridge dropped aliased outputs from your launcher's
expected arguments (read installed bridge source).

## Symptom: "not a TVM-FFI tensor" (or similar wrapper-type error)

The direct path needs an enable flag on the DSL's `from_dlpack` in this
version. Read the error; it names the flag.

## Symptom: works on machine A, fails on machine B

1. Fingerprint both (inspect_environment.py); diff versions — rolling
   container tags and arch-lagged builds are the historical cause of
   *apparent* platform bugs.
2. Fresh-process repro on B (session contamination).
3. Same *size*/config on both? (occupancy-dependent bugs masquerade as
   machine-dependent).
4. Only after 1–3: consider genuine platform/driver issues, and test with
   the direct path to remove the bridge from the equation.

## Symptom: checksums differ across identical fresh runs

Nondeterminism red flag: unwritten regions (empty-allocated buffers),
races, or cross-stream interference. Not "just floating point" when the same
binary/config/seed is used — investigate before proceeding.

## Escalation: kernel bug vs your bug vs bridge bug

The isolation ladder, cheapest first:
1. Trivial elementwise `@cute.kernel` through the bridge → tests the bridge
   machinery itself.
2. Target kernel through the **direct path** with a value check → tests the
   kernel + your contract.
3. Target kernel through the bridge → tests the integration.
4. Toggle exactly one config flag per run via env vars in the repro script.

When a real vendor bug is isolated: the repro script with PASS/FAIL env
toggles *is* the bug report body. Include the isolation matrix (what was
ruled out), the sanitizer result *with its blind-spot caveat stated*, and
exact environment fingerprints — triagers close silent-corruption bugs as
unreproducible unless told why the sanitizer is clean.
