# Kernel data-contract discovery

Goal: a written contract for the kernel covering every row of the checklist
below, with each entry traceable to installed source. `scripts/contract_report.py`
automates the mechanical parts (signatures, assertion lines, call sites).

## No repository clone required

This entire workflow operates on **installed packages in site-packages** —
the cuDNN Frontend CuTe DSL kernels and the cutlass-dsl JAX bridge ship as
Python source inside their pip wheels, fully introspectable via `importlib`.
Do not clone repositories by default. Two exceptions:

- **Compiled components** (e.g. the classic cuDNN graph API's C++ backend
  `.so`): the Python surface is still inspectable, but implementation
  contracts must come from signatures plus empirical probing (Phase 5).
- **Tests/examples missing from the wheel**: fall to evidence tier 3 —
  fetch upstream source *pinned to the exact installed version* (match the
  version string or the `+commit` hash), never repository HEAD.

## The Rosetta stone rule

The highest-value artifact is **the vendor's own call site** for the kernel —
the orchestration module inside the installed package (`_interface.py`,
`api.py`, a `*_wrapper`, or the package's tests). Find it by grepping the
installed tree for the kernel class name:

```bash
PKG=$(python -c "import cudnn, os; print(os.path.dirname(cudnn.__file__))")
grep -rn "KernelClassName" "$PKG" --include='*.py' | grep -v "class KernelClassName"
```

From the call site, extract — in this order of importance:

1. **Buffer allocations** (`torch.empty(...)`, `np.zeros(...)`, etc. — the
   framework used by the vendor's wrapper is irrelevant; the *shapes* are the
   contract). This is where the single most expensive mistake in the source
   experience lived: an output the operation's docs described as
   per-block `(B, H, S/64)` was actually allocated per-token `(B, H, S)` by
   the vendor. The 64× under-allocation corrupted neighboring buffers and
   produced a week of misdirected debugging across three machines.
2. **The exact positional argument order** of the kernel call, including
   `None`s for optional slots.
3. **Workspace construction** — helper functions often carry layout comments
   the kernel file lacks (e.g. "fields are laid out field-major across all
   B·H entries; zero the accumulator tail on the *flattened* view").
4. **Flag couplings** — assignments like `flag_a = flag_b` in the wrapper
   mean the vendor never exercises the decoupled combinations; decoupled
   combinations may be silently broken (one such default combination
   dropped 40% of output tiles).
5. **Layout transforms** applied to tensors before the call (transposes,
   `mark_layout_dynamic`, alignment hints) — replicate their *semantics*.

## Contract checklist

For each kernel, fill in every row. "Unknown" is acceptable only with a plan
to determine it empirically in Phase 5.

| Item | Where the truth lives |
|---|---|
| Compile-time config (ctor args) | `inspect.signature(Cls.__init__)` — **per architecture**; sibling arch classes (`sm90_*` vs `sm100_*`) frequently differ in both parameters and semantics (one takes `dtype` at construction, another infers it at runtime) |
| Runtime args + order | `__call__` signature; confirm against the vendor call site |
| Input shapes/ranks | Body assertions first, vendor allocations second, comments last |
| Output shapes | **Vendor allocations only.** Never from the op name or docs |
| Dtypes | Body checks (`element_type != ...`, dtype asserts) |
| Layout/stride requirements | Body assertions of the form "stride at position k must be 1" (`check_dim`-style). Positional comments in signatures can be wrong — one kernel's comment described dim order `(d, s, h, b)` while the code's reshape+assert demanded the contiguous dim at position 1, i.e. effectively `(s, d, h, b)`. **The assertion + the reshape lines are the contract; the comment is a rumor.** |
| Alignment | Vendor conversion helpers (`assumed_align=` arguments) |
| Scalar params | Signature types (e.g. runtime `Int32` vs Python-int constexpr); note constraints in asserts ("even, >= 2") |
| Workspace | Vendor helper: exact element count formula, dtype, which regions must be **zero-initialized** and on what view |
| Zero-init requirements | Any buffer the kernel accumulates into or counts with. JAX-side outputs are **uninitialized** memory — this must be handled explicitly (see jax-integration.md) |
| Aliasing/mutability | Which args the kernel writes; whether in-place semantics are expected |
| Optional vs required | `Optional[...]` in one arch's signature may be **required** in another (passing `None` produced `'NoneType' object is not subscriptable` *at trace time* on the arch where it was required) |
| Arch constraints | Grep asserts for `head_dim`, block sizes, dtype ("bwd only supports bfloat16", "requires head_dim=128") — these differ per arch and per fwd/bwd |
| Coupled flags | Wrapper assignments and heuristic-chooser functions |

## Reading order for a kernel file

1. `__init__` — compile-time specialization surface.
2. `__call__` signature — runtime surface. Read comments *skeptically*.
3. First ~50 lines of the `__call__` body — this is where tensors get
   reshaped/remapped and asserted. The remap code defines what each position
   means, overriding the signature comment.
4. Grep the body for `assert`, `check_dim`, `element_type`, `const_expr(` —
   collect every constraint.
5. Count internal `.launch(` calls — multi-launch bodies historically
   stress integration bridges differently than single-launch ones.

## When name-based intuition tempts you

Do not infer:
- shape from a tensor's name (`lse`, `stats`, `counts` — resolution and rank
  vary by version and kernel);
- rank from the public API (a wrapper accepting rank-1 metadata may expand it
  to rank-3 before the kernel; the kernel may index it as 3-D);
- semantics from a sibling arch's kernel (argument orders differed across
  all three arch variants of the same operation in the source experience —
  including where the softmax scale sits relative to the metadata tensors).

Every one of these intuitions was wrong at least once.
