# Native FP8 Support In PaxML
The most recent version of XLA already supports FP8 GEMM (General Matrix Muliptlication) through the utilization of custom quantization operations provided by Paxml/Praxis. To enable this feature, users simply need to include the option `--fdl.USE_FP8=True` in their experiment configuration. This will activate the recommended layers to use FP8 GEMMs within the transformer layer. It's important to note that `ENABLE_TE` must be turned off.

In addition to the suggested XLA flags mentioned in [this section](https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/rosetta/projects/pax/README.md#xla-flags), we also recommend setting these following XLA flags. The execution script should look like:
```bash
export XLA_FLAGS=" \
    --xla_gpu_enable_reduction_epilogue_fusion=false \
    --xla_gpu_enable_triton_gemm=false \
    --xla_gpu_enable_cublaslt=true \
    ..."
export ENABLE_TE=0
python -m paxml.main \
    ...
    --fdl.USE_FP8=True \
    ...
```

# Transformer Engine vs Native FP8 Support
Native XLA-FP8 specifically targets matrix multiplication operations. In contrast, the Transformer Engine focuses on enhancing the overall performance of the entire transformer layer. This encompasses not only the FP8 matrix multiplication but also attention mechanisms, layer normalizations, and other components.

In practical terms, XLA-FP8 performs pattern matching and rewrites the matrix multiplication operations in the operation graph to utilize FP8 matrix multiplication. On the other hand, with TE, the [entire Praxis transformer](https://github.com/google/praxis/blob/main/praxis/layers/transformers.py) layer will be substituted with our [Transformer Engine
layer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/jax.html#transformer_engine.jax.flax.TransformerLayer), offering a comprehensive performance enhancement.

