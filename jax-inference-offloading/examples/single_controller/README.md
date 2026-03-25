# Single Controller Example (Dual Topic Sets)

This example adds a **single control-plane orchestrator** while keeping the
existing NCCL weight data path direct between JAX and vLLM.

It is additive: existing examples in:
- `examples/decoupled_synchronous/`
- `examples/decoupled_asynchronous/`

remain unchanged.

## Goal

In this mode:
- JAX and vLLM do **not** share control topics directly.
- The single controller mediates control flow and scheduling.
- NCCL remains the only direct JAX<->vLLM communication path.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                               Gateway                                      │
│                        (gRPC RPC + Message Broker)                         │
│                                                                            │
│  - Receives RPC ingress: handshake/create_transport/weight_updates/        │
│    inference_requests                                                      │
│  - Fan-out/fan-in via pub/sub topics                                       │
│  - KV store for cross-process bootstrap                                    │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┬───────────────────┐
       │                            │                            │                   │
       ▼                            ▼                            ▼                   ▼
┌─────────────────┐       ┌────────────────────┐       ┌─────────────────┐  ┌─────────────────┐
│   vLLM Worker   │       │ Single Controller  │       │   JAX Worker    │  │  Prompt Source  │
│   (vLLM GPUs)   │       │     (CPU-only)     │       │   (JAX GPUs)    │  │   (CPU-only)    │
│                 │       │                    │       │                 │  │                 │
│ - Subscribes to │◄──────│ - Bridges default  │──────►│ - Receives      │  │ - Sends prompts │
│   SC_* control  │       │   ingress -> SC_*  │       │   commands      │  │   via requester │
│ - Runs inference│       │ - Owns sync/async  │       │ - Executes NCCL │  │ - Sync mode     │
│ - Publishes     │──────►│   scheduling       │──────►│   weight xfer   │  │   waits on      │
│   SC results    │       │ - Forwards results │       │ - Publishes     │  │   sync signal   │
└─────────────────┘       └────────────────────┘       │   events/results│  └─────────────────┘
         ▲                                             └─────────────────┘
         │
         ╚══════════════ Direct NCCL weight data path ═════════════════╝
```

## Processes

1. `gateway.py`
2. `single_controller.py`
3. `jax_worker.py`
4. `vllm_worker.py`
5. `prompt_source.py`

## Topic Split

Default ingress topics (produced by existing RPC paths):
- `handshake`
- `create_transport`
- `weight_updates`
- `inference_requests`

Single-controller vLLM topics:
- `sc/handshake`
- `sc/create_transport`
- `sc/weight_updates`
- `sc/inference_requests`
- `sc/results`
- `sc/vllm/events` (worker readiness event)

The vLLM worker subscribes only to `sc/*` topics in this example.

## Synchronization Mechanism

The single controller is the only scheduler.

### Startup Coordination

1. **Gateway** starts first.
2. **Single Controller** subscribes to default ingress topics and SC/internal topics.
3. **vLLM Worker** starts, subscribes to `SC_*` topics, publishes `sc/vllm/events=ready`.
4. **JAX Worker** starts and performs handshake via existing RPC ingress.
   - Controller defers handshake forwarding until `vLLM ready` is observed.
5. **JAX Worker** writes `inference_response_topic=inference/results/sc_forwarded` to KV.
6. **Prompt Source** resolves response topic from KV and starts dispatch.

### Sync Message Flow

```
Single Controller        JAX Worker           Prompt Source         vLLM Worker
      │                      │                      │                    │
      │ 1) "update_weights"  │                      │                    │
      │─────────────────────►│                      │                    │
      │                      │ 2) NCCL transfer     │                    │
      │                      │══════════════════════════════════════════►│
      │                      │ 3) "weights_updated" │                    │
      │◄─────────────────────│                      │                    │
      │ 4) publish sync/weights_ready               │                    │
      │────────────────────────────────────────────►│                    │
      │                      │                      │ 5) AsyncInference  │
      │◄────────────────────────────────────────────│ (default ingress)  │
      │ 6) forward to SC_INFERENCE_REQUESTS ────────────────────────────►│
      │                      │                      │                    │
      │ 7) receive SC_RESULTS ◄──────────────────────────────────────────│
      │ 8) forward to jax forwarded results ────────────────────────────►│
```

### Async Message Flow

```
Prompt Source       Single Controller            JAX Worker              vLLM Worker
     │                     │                         │                       │
     │ AsyncInference      │                         │                       │
     │────────────────────►│ queue/dispatch policy   │                       │
     │                     │────────────────────────────────────────────────►│
     │                     │                         │                       │
     │                     │<────────────────────────────────────────────────│
     │                     │    streamed RolloutResult (SC_RESULTS)          │
     │                     │────────────────────────► forward to JAX results │
     │                     │                         │                       │
     │                     │ after UPDATE_INTERVAL prompts:                  │
     │                     │ "update_weights" ──────►│                       │
     │                     │                         │ NCCL transfer ═══════►│
     │                     │◄────────────────────────│ "weights_updated"     │
```

## Topics Used

| Topic | Publisher | Subscriber | Purpose |
|-------|-----------|------------|---------|
| `handshake` | JAX worker (via RPC ingress) | Single controller | ingress handshake |
| `create_transport` | JAX worker (via RPC ingress) | Single controller | ingress transport creation |
| `weight_updates` | JAX worker (via RPC ingress) | Single controller | ingress weight update signal |
| `inference_requests` | Prompt source (via RPC ingress) | Single controller | ingress inference requests |
| `sc/handshake` | Single controller | vLLM worker | vLLM-side handshake |
| `sc/create_transport` | Single controller | vLLM worker | vLLM-side transport creation |
| `sc/weight_updates` | Single controller | vLLM worker | vLLM-side weight update trigger |
| `sc/inference_requests` | Single controller | vLLM worker | vLLM-side inference execution |
| `sc/results` | vLLM worker | Single controller | handshake/inference results return |
| `sc/vllm/events` | vLLM worker | Single controller | readiness signal |
| `sc/jax/commands` | Single controller | JAX worker | `update_weights`, `shutdown` |
| `sc/jax/events` | JAX worker | Single controller | `ready`, `weights_updated` |
| `inference/results/sc_forwarded` | Single controller | JAX worker | final forwarded rollout outputs |
| `sync/weights_ready` | Single controller | Prompt source (sync mode) | dispatch gate |

## Scheduling

`single_controller.py` supports two modes:
- `SC_MODE=sync`
  - Command JAX worker to transfer weights.
  - After completion, signal prompt source.
  - Dispatch one inference request and wait for result.
- `SC_MODE=async`
  - Dispatch requests while respecting `MAX_STALENESS` as max in-flight prompts.
  - Trigger weight updates every `UPDATE_INTERVAL` completed prompts.
  - Optionally stop after `MAX_COMPLETED_PROMPTS`.

## Run

From `examples/single_controller/`:

### Sync

```bash
bash ./run_single_controller.sh \
  --mode=sync \
  --model-path=/path/to/model \
  --param-mapping-path=../mappings/llama3_1b_param_mapping.json \
  --num-iterations=3
```

### Async

```bash
bash ./run_single_controller.sh \
  --mode=async \
  --model-path=/path/to/model \
  --param-mapping-path=../mappings/llama3_1b_param_mapping.json \
  --num-batches=10 \
  --batch-size=3 \
  --num-rollouts=4 \
  --update-interval=5 \
  --max-staleness=20
```

## Notes

- `prompt_source.py` still uses `VLLMRolloutRequester` and default inference ingress;
  the single controller consumes those ingress messages and republishes to `sc/*`.
- `jax_worker.py` is command-driven and publishes readiness/update-complete events.
- `vllm_worker.py` uses `topic_family=\"single_controller\"`.
