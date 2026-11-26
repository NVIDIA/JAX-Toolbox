# Programmatic Dependent Launch in XLA

- [Read the blog post: Using Programmatic Dependent Launch in XLA for JAX Inference](pdl_xla_jax_toolbox_blog.md).
- [Run the Gemma, TimesFM, and AlphaFold benchmarks and capture profiles](benchmark_scripts/README.md).
- [Script provenance and changes from the original experiment runners](benchmark_scripts/PROVENANCE.md).

The two PNG figures used in the article are included alongside it. The
profile-derived figure also has an SVG companion. The original raw profiles,
model inputs, and weights are not part of this publication bundle.

Shell examples in the article and script documentation run from this directory
(`blog/xla-pdl`), not the top level of the JAX Toolbox repository. On a GPU
machine, mount this directory into the appropriate model container as described
in the article.

GPU-free launcher tests:

```bash
python3 -m unittest discover -s benchmark_scripts/tests -v
```
