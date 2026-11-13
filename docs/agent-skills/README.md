---
title: Agent Skills
subtitle: Procedural skills for AI coding agents working with JAX on NVIDIA GPUs.
slug: agent-skills
---

JAX-Toolbox ships **agent skills**: self-contained packages of instructions,
reference documentation, and scripts that teach an AI coding agent how to do a
specific kind of JAX integration work correctly. They encode *procedure* —
how to discover APIs from the installed environment, how to validate results,
which failure modes to expect — rather than API snippets that go stale between
package versions.

Skills live under [`docs/agent-skills/`](https://github.com/NVIDIA/JAX-Toolbox/tree/main/docs/agent-skills)
in this repository. Each skill is a directory containing:

- `SKILL.md` — the entry point: rules and a phased workflow the agent follows.
- `references/` — deeper documentation the workflow points into as needed.
- `scripts/` — plain Python helpers the agent runs (e.g. environment
  fingerprinting, API introspection).
- `README.md` — a human-facing overview of what the skill does.

## Prerequisites

**An agentic coding tool.** Skills are written for AI agents that can read
files and execute shell commands. The format follows the `SKILL.md`
convention (a directory with a `SKILL.md` entry point), which many agentic
tools discover automatically — but nothing in these skills depends on any
particular tool: any agent can simply be told to read the skill's `SKILL.md`
and follow it.

**The agent must run inside the target environment.** These skills are
discovery-driven: they instruct the agent to introspect installed packages,
run probe scripts, and validate results on a GPU. Run your agent on the
machine (or inside the container) where the work will execute — an agent
without access to the GPU and the installed Python environment cannot follow
the workflow. The [NVIDIA JAX containers](https://github.com/NVIDIA/JAX-Toolbox#containers)
are the reference environment; each skill's own README lists any additional
package requirements.

## Using a skill

If your tool supports skills, copy (or symlink) the skill directory into
wherever it discovers them — for example `.claude/skills/` (project) or
`~/.claude/skills/` (user) for Claude Code, or the equivalent under
`.cursor/`, `.codex/`, etc. Then either mention the skill by name or just
describe the task — the skill triggers on matching requests:

> Using the jax-cudnn-frontend skill: wrap the cuDNN Frontend CuTe DSL block
> sparse attention kernels for inference and training from pure JAX, exposing
> the kernel configuration knobs so we can autotune per problem shape.

If your tool has no skills mechanism, include an instruction like *"Read
`docs/agent-skills/<name>/SKILL.md` and follow it for this task"* in your
prompt — the skills are plain markdown and Python, with no tool-specific
dependencies.

## Available skills

| Skill | Use for |
|---|---|
| [`jax-cudnn-frontend`](jax-cudnn-frontend/README.md) | Implementing, integrating, debugging, and validating cuDNN Frontend / CuTe DSL kernels from pure JAX (no PyTorch) — wrapping, calling, porting, or autotuning attention variants, GEMM fusions, and experimental `csrc` kernel classes. |
