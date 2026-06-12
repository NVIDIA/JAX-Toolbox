// JAX on NVIDIA GPU stack diagram — source of truth.
//
// columns: the three vertical bands; width values must sum to 12.
// rows:    horizontal layers, top to bottom; cell widths per row must sum to 12.
// projects: registry of every node; id must be unique.
//
// To add a project:
//   1. Add an entry to `projects` with a short `description`.
//   2. Place its id in the appropriate row cell's `projects` array.

export const STACK = {

  // ------------------------------------------------------------------
  // Grid columns (total = 12 units)
  // ------------------------------------------------------------------
  columns: [
    { id: "tooling",   label: "Tooling",                 width: 2 },
    { id: "training",  label: "Training / Post-Training", width: 7 },
    { id: "inference", label: "Inference",                width: 3 },
  ],

  // ------------------------------------------------------------------
  // Grid rows (top = frameworks, bottom = hardware)
  // ------------------------------------------------------------------
  rows: [
    {
      id: "frameworks", label: "Frameworks",
      cells: [
        { width: 2, label: "NVIDIA Profiling", projects: ["nsight", "nsys-jax"] },
        { width: 7, label: "Frameworks", projects: ["maxtext", "maxdiffusion"] },
        { width: 3, projects: [] }, 
      ],
    },
    {
      id: "ecosystem", label: "JAX ecosystem",
      cells: [
        { width: 2, label: "Native Profiling",    projects: ["xprof"] },
        { width: 4, label: "Native Libraries", projects: ["grain", "optax", "orbax", "tunix", "qwix", "flax", "pallas-mosaicgpu", "tokamax"] },
        { width: 4, label: "NVIDIA Libraries & kernel DSLs",  projects: ["jaxpp", "te", "warp", "cuequivariance", "cutile", "cutlass"] },

        { width: 2, label: "NVIDIA Libraries",          projects: ["flashinfer", "mlir-trt"] },
      ],
    },
    {
      id: "core", label: "JAX",
      cells: [
        { width: 12, label: "Framework", projects: ["jax"] },
      ],
    },
    {
      id: "core", label: "XLA",
      cells: [
        { width: 12, label: "Compiler", projects: ["xla"] },
      ],
    },

    {
      id: "native-xla", label: "Native XLA",
      cells: [
        { width: 2,  label: "NVIDIA Profiling", projects: ["cupti"] },
        { width: 10, label: "NVIDIA Libraries", projects: ["cudnn", "cublas", "nccl", "nvshmem", "RAPIDS"] },
      ],
    },
    {
      id: "hardware", label: "Hardware",
      cells: [
        { width: 12, label: "Hardware", projects: ["hw-datacenter", "hw-workstation", "hw-rtx", "hw-jetson"] },
      ],
    },
  ],

  // ------------------------------------------------------------------
  // Project registry
  // category: "nvidia" | "jax" | "other"
  // nvidia_participates: true — NVIDIA contributes to an OSS/jax project
  // isStatic: true — not clickable (no overview panel)
  // ------------------------------------------------------------------
  projects: [

    // Frameworks row
    {
      id: "maxtext", name: "MaxText", category: "jax",
      href: "../frameworks/maxtext/README.md",
      nvidia_participates: true,
      description: `A scalable, high-performance JAX LLM framework (GPT, Llama, Gemma,
        Mistral, Mixtral). NVIDIA actively tests and contributes to MaxText and publishes
        a tuned container, making it the reference pre-training framework for JAX on
        NVIDIA GPUs.`,
    },
    {
      id: "maxdiffusion", name: "MaxDiffusion", category: "jax",
      href: "https://github.com/AI-Hypercomputer/maxdiffusion",
      nvidia_participates: true,
      description: `A scalable JAX framework for diffusion-model training, sharing
        MaxText's performance philosophy. Runs on NVIDIA GPUs through the same XLA/JAX
        stack.`,
    },
    {
      id: "axlearn", name: "AXLearn", category: "jax",
      href: "../frameworks/axlearn/README.md",
      nvidia_participates: true,
      description: `Apple's deep-learning framework built on JAX and XLA for large-scale
        models such as the Fuji family. NVIDIA validates AXLearn on GPU clusters and
        publishes an axlearn container.`,
    },
    {
      id: "tokamax", name: "Tokamax", category: "jax",
      href: "https://github.com/google/tokamax",
      nvidia_participates: true,
      description: `A curated library of state-of-the-art custom kernels (attention and
        more) exposed through Pallas, with per-hardware autotuned implementations on
        NVIDIA GPUs.`,
    },
    {
      id: "alphafold3", name: "AlphaFold3", category: "nvidia",
      href: "https://github.com/google-deepmind/alphafold3",
      description: `DeepMind's protein-structure model. NVIDIA provides an optimized
        inference container (ghcr.io/nvidia/jax:alphafold) for running AlphaFold3 on
        GPUs.`,
    },

    // Tooling
    {
      id: "nsight", name: "Nsight", category: "nvidia",
      href: "https://developer.nvidia.com/nsight-systems",
      description: `NVIDIA Nsight Systems (system-wide CPU+GPU+NVLink timeline) and
        Nsight Compute (per-kernel roofline, memory throughput, warp efficiency),
        bundled in the NGC JAX container for deep GPU-side profiling.`,
    },
    {
      id: "nsys-jax", name: "nsys-jax", category: "nvidia",
      href: "../nsys-jax.md",
      description: `A JAX-Toolbox wrapper around Nsight Systems that captures and
        post-processes profiles of JAX programs, aligning XLA/HLO with GPU timelines.`,
    },

    // JAX ecosystem — profiling
    {
      id: "xprof", name: "XProf", category: "jax",
      href: "https://github.com/openxla/xprof",
      description: `OpenXLA's profiler — trace viewer, HLO op profile, memory viewer,
        and roofline analysis. On NVIDIA GPUs it collects traces via CUPTI and shares
        the XLA HLO graph view; NVIDIA contributes to the GPU path.`,
    },

    // JAX ecosystem — training libraries
    {
      id: "grain", name: "Grain", category: "jax",
      href: "https://github.com/google/grain",
      description: `A deterministic, checkpointable input-data pipeline for JAX.
        Framework-independent and runs on NVIDIA GPUs.`,
    },
    {
      id: "optax", name: "Optax", category: "jax",
      href: "https://github.com/google-deepmind/optax",
      nvidia_participates: true,
      description: `A library of composable gradient-processing and optimization
        transforms for JAX, used by training frameworks across the ecosystem.`,
    },
    {
      id: "orbax", name: "Orbax", category: "jax",
      href: "https://github.com/google/orbax",
      nvidia_participates: true,
      description: `Any-scale distributed checkpointing for JAX, shared across TPU
        and GPU workflows.`,
    },
    {
      id: "tunix", name: "Tunix", category: "jax",
      href: "https://github.com/google/tunix",
      description: `A JAX-native framework for post-training and alignment —
        supervised fine-tuning and RL methods such as PPO and DPO/RLHF.
        Framework-independent and runs on NVIDIA GPUs.`,
    },
    {
      id: "qwix", name: "Qwix", category: "jax",
      href: "https://github.com/google/qwix",
      description: `A JAX-native quantization library (PTQ, QAT, and QLoRA). Its
        non-intrusive interception applies quantization without changing model code,
        targeting XLA on both NVIDIA GPUs and TPUs.`,
    },
    {
      id: "flax", name: "Flax", category: "jax",
      href: "https://github.com/google/flax",
      nvidia_participates: true,
      description: `A flexible neural-network authoring library for JAX (the NNX API).
        Used across NVIDIA's NGC containers and Google's MaxText; NVIDIA contributes
        upstream.`,
    },
    {
      id: "pallas-mosaicgpu", name: "Pallas / Mosaic GPU", category: "jax",
      href: "https://docs.jax.dev/en/latest/pallas/index.html",
      description: `Pallas is JAX's kernel-authoring API; on NVIDIA GPUs it lowers
        primarily to the Mosaic GPU backend (Hopper+), exposing the GMEM→SMEM→registers
        pipeline, TMA async copies, and warp specialization. A Triton backend remains
        as an Ampere+ fallback.`,
    },

    // JAX ecosystem — scale & precision
    {
      id: "jaxpp", name: "JaxPP", category: "nvidia",
      href: "https://github.com/NVIDIA/JaxPP",
      description: `NVIDIA's pipeline-parallelism library for JAX, splitting
        large-model training into pipeline stages across GPUs and nodes with flexible
        scheduling to keep devices busy.`,
    },
    {
      id: "te", name: "TransformerEngine", category: "nvidia",
      href: "https://github.com/NVIDIA/TransformerEngine",
      description: `NVIDIA's mixed-precision training library. The JAX extension
        delivers FP8, MXFP8, and NVFP4 training on Hopper and Blackwell Tensor Cores
        and ships inside NVIDIA's NGC JAX container, unlocking low-precision paths
        that reach deeper into NVIDIA silicon than framework-agnostic quantization.`,
    },

    // JAX ecosystem — kernels
    {
      id: "warp", name: "Warp", category: "nvidia",
      href: "https://github.com/NVIDIA/warp",
      description: `NVIDIA Warp is a Python framework for writing high-performance,
        differentiable GPU kernels (simulation, geometry, custom ops). Warp kernels
        can be called from JAX through the foreign-function interface.`,
    },
    {
      id: "cuequivariance", name: "cuEquivariance", category: "nvidia",
      href: "https://github.com/NVIDIA/cuEquivariance",
      description: `An NVIDIA library that accelerates equivariant neural networks —
        common in chemistry, materials, and structural biology — with optimized
        primitives and JAX bindings for GPU execution.`,
    },
    {
      id: "cutile", name: "cuTile", category: "nvidia",
      description: `NVIDIA's tile-based programming model for authoring GPU kernels
        at a higher level than raw CUDA: computation is expressed over tiles that the
        compiler maps onto the hardware.`,
    },
    {
      id: "cutlass", name: "CUTLASS", category: "nvidia",
      href: "https://github.com/NVIDIA/cutlass",
      description: `NVIDIA's open-source CUDA templates for high-performance GEMM
        and convolution — a building block for custom kernels and library backends on
        NVIDIA GPUs.`,
    },

    // JAX ecosystem — inference
    {
      id: "flashinfer", name: "FlashInfer", category: "nvidia",
      href: "https://github.com/flashinfer-ai/flashinfer",
      description: `A high-performance library of attention and LLM-serving kernels
        for NVIDIA GPUs (prefill/decode attention, paged KV cache), used to accelerate
        JAX/GPU inference paths.`,
    },
    {
      id: "mlir-trt", name: "MLIR-TRT", category: "nvidia",
      href: "https://github.com/NVIDIA/TensorRT-Incubator",
      description: `MLIR-TensorRT — an MLIR-based compiler path that lowers StableHLO
        to optimized TensorRT engines for NVIDIA inference.`,
    },

    // Core row
    {
      id: "jax", name: "JAX", category: "jax",
      href: "https://github.com/jax-ml/jax",
      nvidia_participates: true,
      description: `Accelerator-oriented array computation with composable transforms
        — jit, grad, vmap, pmap. Hardware-agnostic and the foundation of the whole
        stack; runs on NVIDIA GPUs through XLA's CUDA backend.`,
    },
    {
      id: "xla", name: "XLA", category: "jax",
      href: "https://github.com/openxla/xla",
      nvidia_participates: true,
      description: `The shared JIT compiler behind JAX. Programs trace to StableHLO,
        which XLA:GPU lowers via LLVM NVPTX to PTX/SASS, calling cuBLAS and cuDNN as
        library ops. OpenXLA is co-developed by NVIDIA alongside Google and others.`,
    },
    {
      id: "pathways", name: "Pathways", category: "other",
      description: `Google's proprietary single-controller runtime for orchestrating
        large TPU pods. Shown for context — it is not part of the NVIDIA GPU path,
        where multi-node JAX uses HPC-X/OpenMPI, Ray, or SLURM.`,
    },

    // Native XLA row
    {
      id: "cupti", name: "CUPTI", category: "nvidia",
      href: "https://docs.nvidia.com/cupti/",
      description: `The CUDA Profiling Tools Interface — the low-level API that
        profilers (including XProf's GPU path and nsys-jax) use to collect GPU
        traces and metrics.`,
    },
    {
      id: "cudnn", name: "cuDNN", category: "nvidia",
      href: "https://developer.nvidia.com/cudnn",
      description: `NVIDIA's deep-learning primitive library, called by XLA:GPU for
        convolutions and attention. Provides fused FlashAttention and paged-attention
        kernels used under JAX.`,
    },
    {
      id: "cublas", name: "cuBLAS / cuBLASLt", category: "nvidia",
      href: "https://developer.nvidia.com/cublas",
      description: `NVIDIA's GEMM library, invoked by XLA:GPU for matrix multiply —
        including FP8 and MXFP8 matmul on Hopper and Blackwell.`,
    },
    {
      id: "nccl", name: "NCCL", category: "nvidia",
      href: "https://developer.nvidia.com/nccl",
      description: `The NVIDIA Collective Communications Library — topology-aware
        AllReduce, AllGather, and ReduceScatter across NVLink and InfiniBand. JAX/XLA
        drives multi-GPU and multi-node collectives through NCCL.`,
    },
    {
      id: "nvshmem", name: "NVSHMEM", category: "nvidia",
      href: "https://developer.nvidia.com/nvshmem",
      description: `GPU-initiated symmetric memory for low-latency, fine-grained
        communication. Integrates with Mosaic GPU kernels to hide latency,
        complementing NCCL's bulk collectives.`,
    },

    // Hardware row (static — no overview panel)
    { id: "hw-datacenter",  name: "Datacenter",  category: "nvidia", isStatic: true },
    { id: "hw-workstation", name: "Workstation", category: "nvidia", isStatic: true },
    { id: "hw-rtx",         name: "GeForce RTX",  category: "nvidia", isStatic: true },
    { id: "hw-jetson",      name: "Jetson",       category: "nvidia", isStatic: true },
  ],
};
