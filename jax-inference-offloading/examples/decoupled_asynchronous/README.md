# Decoupled Asynchronous Example

This example demonstrates an **asynchronous decoupled architecture** for JAX-vLLM inference offloading, designed for high-throughput RL post-training workflows like GRPO.

## Key Differences from Synchronous Version

| Aspect | Synchronous | Asynchronous |
|--------|-------------|--------------|
| **Driver** | JAX Controller | Prompt Dispatcher (autonomous) |
| **Prompt dispatch** | Waits for `sync/weights_ready` signal | Continuous, blocks only on queue backpressure |
| **Weight updates** | Blocking, every iteration | Controlled via `MAX_STALENESS` |
| **Result delivery** | All B×R results in one message | Streamed per-rollout |
| **JAX processing** | Waits for full batch | Processes each prompt group as it completes |
| **Staleness control** | None (always fresh) | Configurable via `--max-staleness` |

## Use Case

This architecture is designed for **RL post-training** workflows (e.g., GRPO) where:
- High throughput is critical
- Controllable staleness is acceptable (vLLM runs ahead by at most N prompts)
- Results should be processed as they become available
- Weight updates are less frequent than in synchronous mode

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Gateway                                    │
│                        (gRPC Message Broker)                            │
│                                                                         │
│  - Routes messages between processes via pub/sub topics                 │
│  - Provides KV store for cross-process coordination                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   vLLM Worker   │      │  JAX Controller │      │Prompt Dispatcher│
│   (vLLM GPUs)   │      │   (JAX GPUs)    │      │   (CPU only)    │
│                 │      │                 │      │                 │
│ - Bounded       │◄─────│ - Accumulates   │      │ - Autonomous    │
│   staleness     │ NCCL │   rollout       │      │   dispatch loop │
│   control       │      │   results       │      │ - Blocks on     │
│ - Streams       │      │ - Pushes weight │      │   backpressure  │
│   results       │─────►│   updates after │      │ - No sync       │
│ - Partial batch │      │   N prompts     │      │   signals       │
│   processing    │      │                 │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

## The Four Processes

### 1. Gateway (`gateway.py`)
The central message broker (shared with synchronous version).

- **Location**: `jax_inference_offloading/controller/gateway.py`
- **Role**: Routes gRPC messages, provides KV store
- **GPU**: None required
- **New in async**: Supports bounded queues for backpressure

### 2. vLLM Worker (`vllm_worker.py`)
Runs inference with streaming results and bounded staleness control.

- **Location**: `examples/decoupled_asynchronous/vllm_worker.py`
- **Role**: Generates rollouts, streams results, receives weight updates
- **GPU**: Requires vLLM-assigned GPUs
- **Library components used**:
  - `RolloutServicer` - Shared with sync version, handles inference and weight updates
  - `AsyncRolloutClient` - Adds staleness control, request queuing, partial batch processing
- **Key async behavior**:
  - Receives weight updates via `WEIGHT_UPDATES` topic (same mechanism as sync)
  - Streams individual `RolloutResult` messages as rollouts complete
  - **Bounded staleness**: Limits itself to `MAX_STALENESS` prompts before waiting for weight update
  - **Partial batch processing**: Can split batches to hit exact staleness limits
  - Queues inference requests when at staleness limit, drains queue after weight update

### 3. JAX Controller (`jax_controller.py`)
Accumulates streamed results and pushes weight updates.

- **Location**: `examples/decoupled_asynchronous/jax_controller.py`
- **Role**: Accumulates rollouts by prompt, triggers weight updates
- **GPU**: Requires JAX-assigned GPUs
- **Library components used**:
  - `RolloutAccumulator` - Thread-safe accumulator for grouping rollout results
  - `result_consumer_thread` - Background thread for consuming gRPC stream
- **Key async behavior**:
  - **Consumer thread** continuously reads from gRPC stream (never blocks)
  - **Main thread** polls for completed groups and handles weight updates
  - When main thread blocks on NCCL, consumer thread keeps reading (no message loss)
  - Groups results by `(batch_id, prompt_index)`
  - After N prompts complete (`UPDATE_INTERVAL`), pushes weight update

### 4. Prompt Dispatcher (`prompt_dispatcher.py`)
Autonomously dispatches prompts without waiting for signals.

- **Location**: `examples/decoupled_asynchronous/prompt_dispatcher.py`
- **Role**: Continuously sends inference requests
- **GPU**: None required
- **Key async behavior**:
  - No sync signal subscription
  - Blocks only when inference queue is full (backpressure)
  - Generates unique `batch_id` for result correlation

## Data Flow

### Per-Rollout Streaming

```
vLLM Worker                 Gateway              JAX Controller
     │                         │                      │
     │  generate rollouts      │                      │
     │                         │                      │
     │  RolloutResult(P0,R0)   │                      │
     │────────────────────────►│─────────────────────►│
     │                         │               [accumulate]
     │  RolloutResult(P1,R0)   │                      │
     │────────────────────────►│─────────────────────►│
     │                         │               [accumulate]
     │  RolloutResult(P0,R1)   │                      │
     │────────────────────────►│─────────────────────►│
     │                         │               [P0 complete!]
     │         ...             │                      │
```

### Weight Updates (Same as Sync, Less Frequent)

```
JAX Controller              Gateway              vLLM Worker
     │                         │                      │
     │  [enough prompts done]  │                      │
     │                         │                      │
     │  StartWeightUpdate      │                      │
     │  (via WEIGHT_UPDATES)   │                      │
     │────────────────────────►│─────────────────────►│
     │                         │               [start NCCL recv]
     │                         │                      │
     │◄════════════ NCCL Transfer ════════════════════►│
     │                         │                      │
     │  [continue accumulating │      [continue with  │
     │   rollout results]      │       new weights]   │
```

## Result Correlation for GRPO

Each streamed `RolloutResult` includes:
- `batch_id`: Unique ID for the inference request
- `prompt_index`: Which prompt within the batch (0-based)
- `rollout_index`: Which rollout for this prompt (0 to num_outputs-1)

The JAX controller uses a **thread-safe accumulator** with a dedicated consumer thread:

```python
# Consumer thread (never blocks, runs independently)
def result_consumer_thread(results_stream, accumulator):
    for delivery in results_stream:
        result = unpack(delivery)
        accumulator.add_result(result)  # Thread-safe

# Main thread (can block on NCCL without losing results)
while not done:
    groups = accumulator.get_completed_groups(timeout=0.1)  # Non-blocking
    for batch_id, prompt_idx, group in groups:
        # Ready for GRPO: compute rewards, advantages, gradients
        process_group(group)
        
        if should_update_weights:
            transfer_engine.update_weights(params)  # Blocks, but consumer keeps reading
```

This design ensures that **no results are lost** even when the main thread blocks on NCCL weight transfers.

## Bounded Staleness Control

By default, vLLM runs as fast as possible, which can lead to unbounded staleness (vLLM may process many prompts with stale weights before JAX sends an update). The `--max-staleness` option provides controllable staleness:

```bash
./decoupled_async.sh \
  --update-interval=5 \
  --max-staleness=5 \
  ...
```

### How It Works

1. vLLM tracks how many prompts it has processed since the last weight update
2. When the count reaches `MAX_STALENESS`, vLLM queues subsequent inference requests
3. When a weight update arrives, vLLM processes it and drains the queue
4. This ensures vLLM never runs more than `MAX_STALENESS` prompts ahead

```
Without staleness control (MAX_STALENESS=0):
vLLM:  [INF_1][INF_2]...[INF_10][WU_1][WU_2]...
       └────────── All use stale weights ──────┘

With staleness control (MAX_STALENESS=5):
vLLM:  [INF_1][INF_2][WU_1][INF_3][INF_4][WU_2]...
       └─ 5 prompts ─┘     └─ 5 prompts ─┘
```

### Configuration Constraint

**Simple rule**: `MAX_STALENESS >= UPDATE_INTERVAL`

vLLM supports **partial batch processing** - if a batch of 3 prompts arrives but only 2 fit within the staleness limit, vLLM will:
1. Process 2 prompts immediately
2. Queue the remaining 1 prompt until the next weight update

This means vLLM can process exactly up to `MAX_STALENESS` prompts, regardless of batch size.

**Examples**:
```bash
# Valid: MAX_STALENESS >= UPDATE_INTERVAL
--max-staleness=5 --update-interval=5  # ✓ vLLM processes exactly 5
--max-staleness=5 --update-interval=3  # ✓ vLLM processes 5, JAX updates at 3
--max-staleness=1 --update-interval=1  # ✓ vLLM processes 1 prompt at a time

# Invalid: would deadlock
--max-staleness=3 --update-interval=5  # ✗ vLLM blocks at 3, JAX needs 5
```

The vLLM worker validates this at startup and will error with a helpful message if misconfigured.

## Running the Example

```bash
cd examples/decoupled_asynchronous

./decoupled_async.sh \
  --model-path=/path/to/model \
  --param-mapping-path=../mappings/llama3_1b_param_mapping.json \
  --num-batches=10 \
  --batch-size=3 \
  --num-rollouts=4 \
  --update-interval=10
```

### Required Arguments
- `--model-path` - Path to HuggingFace model checkpoint
- `--param-mapping-path` - Path to JSON parameter mapping file

### Async-Specific Arguments
- `--num-batches=N` - Number of batches to dispatch (default: 10, 0 for infinite)
- `--batch-size=N` - Prompts per batch (default: 3)
- `--num-rollouts=N` - Rollouts per prompt (default: 4)
- `--update-interval=N` - Push weight update after N prompts (default: 10)
- `--max-staleness=N` - Max prompts vLLM processes before requiring weight update (default: 0=unlimited)
- `--max-completed-prompts=N` - Stop after N prompts (default: 100)
- `--dispatch-delay=FLOAT` - Delay between dispatches in seconds (default: 0.0)

### Common Arguments
- `--n-gpus-vllm=N` - GPUs for vLLM (default: 4)
- `--n-gpus-jax=N` - GPUs for JAX (default: 4)
- `--transfer-mode=MODE` - Weight transfer mode: `grouped`/`fused`/`unfused`
- `--gateway-port=PORT` - gRPC port (default: 50051)
- `--debug` - Enable verbose logging

## Customization for GRPO

To add GRPO training logic, modify `jax_controller.py`:

```python
# In the main loop, after getting completed groups from accumulator
for batch_id, prompt_idx, group in accumulator.get_completed_groups():
    # group is a list of RolloutResult messages for this prompt
    
    # 1. Compute rewards for each rollout
    rewards = compute_rewards(group)
    
    # 2. Compute GRPO advantages (relative within group)
    advantages = grpo_advantages(rewards)
    
    # 3. Accumulate gradients
    grads = compute_grpo_gradients(group, advantages, params)
    accumulated_grads = accumulate(accumulated_grads, grads)
    
    prompts_processed += 1
    
    # 4. After enough prompts, apply gradients and push weights
    if prompts_processed >= update_interval:
        params = apply_gradients(params, accumulated_grads)
        transfer_engine.update_weights(params)  # NCCL transfer to vLLM
        prompts_processed = 0
        accumulated_grads = None
```

## Trade-offs

### Advantages
- **Higher throughput**: vLLM doesn't wait for JAX; dispatch doesn't wait for inference
- **Better GPU utilization**: Both JAX and vLLM can work in parallel
- **Incremental processing**: Process results as they arrive
- **No message loss**: Consumer thread reads results even when main thread blocks on NCCL
- **Controllable staleness**: Use `--max-staleness` to limit how far vLLM runs ahead

### Considerations
- **Slightly stale weights**: vLLM may generate some rollouts with weights from previous update
- **More complex state management**: Need to track pending groups, correlate results
- **Staleness/throughput trade-off**: Lower `MAX_STALENESS` = fresher weights but lower throughput

## Library Components

This example uses the following library components from `jax_inference_offloading`:

### For vLLM Worker
```python
from jax_inference_offloading.controller.rollout_client import (
    RolloutServicer,        # Handles inference, weight updates, handshake
    AsyncRolloutClient,     # Adds staleness control and request queuing
    make_async_rollout_client,  # Factory function
)
```

### For JAX Controller
```python
from jax_inference_offloading.engines import (
    RolloutAccumulator,      # Thread-safe result accumulator
    result_consumer_thread,  # Background gRPC consumer
)
```

These components can be reused in custom implementations without copying the example code.

## Environment Variables

Set `VERBOSE_CONSUMER=1` to see each result as the consumer thread receives it (helpful for debugging).
