# JAX integration

Goal: expose the kernel to JAX as a jittable operation with correct buffer
semantics — after (never before) the standalone proof in Phase 5.

## Discover the mechanism, don't assume it

The bridge surface changes between versions. First introspect what exists:

```python
import cutlass.jax as cjax
print([a for a in dir(cjax) if not a.startswith('_')])
import inspect
print(inspect.signature(cjax.cutlass_call))      # or whatever exists
```

Then **read the installed bridge source** (it's Python, it's short, and it is
the only authoritative statement of the conventions below):

```bash
python -c "import cutlass.jax.primitive as p; print(p.__file__)"
# read: the call-builder (primitive.py-like) and the compile/launch wrapper
# (compile.py-like). Both file names are version-specific.
```

Verify each of the following against that source for *your* version. All are
known to exist in some version, and several changed or were buggy in specific
versions:

## Conventions to verify (each burned us once)

**Launcher convention.** The bridge traces a `@cute.jit` function you supply.
Typical contract: `launcher(stream, *inputs, *outputs, **constexpr_kwargs)`
with outputs allocated by XLA from your `output_shape_dtype` declaration.
Your launcher's job is purely argument reordering into the kernel's
`__call__` order.

**Outputs are uninitialized.** XLA hands the launcher raw buffers. Any buffer
the kernel *accumulates into or increments* (counters, workspaces) must be
zeroed. Two patterns:
- *Zero-prologue kernel* (robust across versions): a tiny `@cute.kernel`
  that flattens the tensor and writes zeros, launched inside your launcher
  before the main kernel, on the same stream. Prefer this.
- *Aliased zeroed input*: pass `jnp.zeros(...)` as an input and declare
  `input_output_aliases`. **Version-sensitive**: the aliasing machinery was
  silently broken in one bridge release (empty outputs, no error) and its
  argument-list behavior is subtle — verify in the installed source whether
  aliased outputs are *removed from the launcher's output arguments* (in the
  studied version they were: you write through the input arg, and the
  Python-level return still contains all declared outputs).

**Per-tensor specs.** If the bridge has a spec type (layout/mode/alignment/
static-ness per tensor), note: (a) it usually requires exactly one spec per
tensor — no broadcasting a single spec; (b) *mode*-style logical permutations
are zero-copy views — if the kernel demands "dim X contiguous at position k"
and dim X is already contiguous in your row-major array, a mode permutation
alone satisfies it with no physical relayout; (c) *layout*-style fields ask
XLA to materialize a different physical order — only needed when no
permutation of the existing layout satisfies the stride assertions.

**Scalars/config.** Constexpr kwargs specialize the compiled kernel. Tuples
(e.g. problem shapes) must be hashable. Prefer passing plain Python numbers
and letting the DSL coerce, over constructing DSL scalar types inside the
traced launcher, unless the installed version's own examples do otherwise.

**Streams.** The bridge injects XLA's stream into the launcher; pass it
through to the kernel. Never fabricate stream handles inside launcher code.

**JIT.** Wrap the calling function in `jax.jit` (config choices that alter
shapes go in `static_argnums`). Confirm integration by checking
`jax.make_jaxpr(...)` shows the op, and — more importantly — by bit-comparing
the jitted path's output against the direct path (below).

## Keep the direct path alive

Alongside the JAX-native wrapper, maintain a `cute.compile`-style direct
invocation of the same kernel instance (concrete arrays via the DSL's
`from_dlpack`; check whether a TVM-FFI enable flag is required by the
installed version — a missing flag produces an explicit "not a TVM-FFI
tensor" error). This path:

- is what the vendor's own wrappers use — the best-tested route;
- runs eagerly on concrete arrays (cannot live under `jax.jit`);
- is your A/B oracle: same kernel + same tensors through both paths must be
  bit-identical. Direct-correct + bridge-wrong = bridge bug (report it);
  both-wrong = kernel or contract problem; both-correct = your remaining
  bugs are above this layer.

## Pointers and ownership

- JAX↔anything transfers via DLPack are zero-copy views. If a returned array
  borrows memory owned by another framework/pool, `.copy()` it into
  JAX-owned memory before the owner can release it — a freed borrowed buffer
  produces illegal-address crashes at *later, unrelated* operations.
- `np.asarray(jax_array)` caches: the second call returns the first call's
  host copy even if device memory changed since. Force a fresh device read
  with `np.asarray(jnp.copy(x))` when auditing device state.

## Autodiff and batching

- The custom call defines no gradient. Provide `jax.custom_vjp`: forward
  returns residuals (typically the outputs the backward kernel consumes —
  their shapes discovered via a full contract pass on the *backward*
  kernel, which is a separate kernel with its own arch variants, workspace,
  and metadata pipeline, often including auxiliary kernels such as index
  inverters).
- `jax.vmap` does not flow through custom calls; map over batch dimensions
  the kernel natively supports instead.
- If any stage must run eagerly (e.g. a sub-kernel broken under the bridge
  in the installed version), the enclosing training step cannot be jitted —
  jit the pure-JAX portions separately and document the constraint.

## Example (version-specific)

The following worked against nvidia-cutlass-dsl 4.6.1/4.7.0 + cudnn-frontend
1.27.0. It illustrates the *shape* of a solution, not a timeless API — every
name below must be re-verified per Phase 4.

```python
@cute.jit
def _launcher(stream, x, meta, out, aux, *, scale: float):
    _zero_f32(aux).launch(                      # prologue: aux is accumulated into
        grid=(cute.ceil_div(cute.size(aux), 256), 1, 1),
        block=(256, 1, 1), stream=stream)
    _kernel(x, out, aux, meta, cutlass.Float32(scale), stream)

@functools.partial(jax.jit, static_argnums=())
def op(x, meta):
    return cjax.cutlass_call(
        _launcher,
        output_shape_dtype=[
            jax.ShapeDtypeStruct(x.shape, x.dtype),          # out
            jax.ShapeDtypeStruct(aux_shape, jnp.float32),    # aux — shape from
        ],                                                   # vendor allocation!
        softmax_scale=..., 
    )(x, meta)
```
