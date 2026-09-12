# Agent skills

[`docs/agent-skills/`](docs/agent-skills/) holds procedural skills for AI
coding agents working with JAX. They aren't about navigating this repo's own
code — they cover general JAX integration work (e.g. wiring up an external
kernel library) and are stored here purely for discoverability. Each skill is
self-contained (instructions + reference docs + scripts) and tool-agnostic —
read the skill's own file directly regardless of which agent or IDE you're
using.

- [`jax-cudnn-frontend`](docs/agent-skills/jax-cudnn-frontend/SKILL.md) —
  implementing, integrating, debugging, and validating cuDNN Frontend / CuTe
  DSL kernels from pure JAX. Use before wrapping, calling, porting, or
  autotuning a cudnn-frontend kernel (attention variants, GEMM fusions,
  experimental csrc kernel classes) from JAX.
