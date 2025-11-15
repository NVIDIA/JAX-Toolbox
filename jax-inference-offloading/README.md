# JAX-vLLM Rollout Offloading Bridge

## Overview

This project couples JAX training with vLLM inference to accelerate reinforcement-learning (RL) post-training. In RL, the model must generate many completions ("rollouts") each iteration. Those rollouts behave like inference workloads and often dominate total compute. The bridge keeps training in JAX while offloading rollouts to vLLM, combining JAX's scalable training with an inference-optimized engine.

## Why offload rollouts?

- Rollouts are autoregressive, bandwidth/latency-bound, and benefit from inference frameworks' specialized scheduling and kernel paths.
- In-framework decoding is straightforward but typically slower; offloading provides inference-grade throughput while the trainer stays in JAX.

## What this bridge provides

- Lightweight coupling layer between JAX and vLLM.
- RPC gateway (control plane): trainer <-> rollout engine coordination; runtime negotiation of model/parallelism settings.
- NCCL data plane (fast path): direct GPU-to-GPU weight streaming (no disk I/O or serialization).
- Resharding and layout mapping: handles different parallelism schemes (e.g., FSDP in JAX vs TP in vLLM) with fan-in/fan-out strategies.
- Simple rollout API surface: send prompts and sampling parameters from JAX, receive generated outputs from vLLM.

## Architecture (high level)

1. Trainer (JAX): runs the RL loop and periodically exports current weights.
2. Gateway (gRPC): routes control messages and orchestrates transport setup.
3. Rollout engine (vLLM): maintains the serving model, accepts live weight updates, and performs batched/token-efficient generation.
4. Transport: NCCL streams TP-aware weight shards directly to vLLM ranks; mapping logic aligns tensor names/layouts across frameworks.

## Capabilities

- Frequent, low-overhead weight refresh from JAX into vLLM.
- TP-aware sharding (pre-sharded updates) to reduce bandwidth and memory.
- Flexible deployment: trainer and rollout can run on different GPU "meshes" and sizes.
- Extensible tensor mappings: reference mappings for common LLM families; adaptable for custom models.
