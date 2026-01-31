# Decoupled Synchronous Example

This example demonstrates a **decoupled architecture** for JAX-vLLM inference offloading, where weight transfer, prompt dispatching, and inference execution run in separate processes that coordinate synchronously.

## Use Case

This architecture is designed for **RL post-training** workflows where:
- A JAX-based trainer updates model weights
- vLLM handles fast inference/rollout generation
- Prompts may come from a separate data pipeline

By decoupling these concerns into separate processes, you gain flexibility to:
- Scale each component independently
- Replace the prompt source without modifying the trainer
- Add custom logic between weight updates and inference

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Gateway                                    │
│                        (gRPC Message Broker)                            │
│                                                                         │
│  - Routes messages between processes via pub/sub topics                 │
│  - Provides KV store for cross-process coordination                     │
│  - No GPU required                                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   vLLM Worker   │      │  JAX Controller │      │Prompt Dispatcher│
│   (vLLM GPUs)   │      │   (JAX GPUs)    │      │   (CPU only)    │
│                 │      │                 │      │                 │
│ - Runs vLLM     │◄─────│ - Loads model   │      │ - Loads prompts │
│ - Receives      │ NCCL │ - Transfers     │      │ - Waits for     │
│   weights       │      │   weights       │      │   sync signal   │
│ - Executes      │      │ - Receives      │      │ - Sends         │
│   inference     │      │   results       │      │   requests      │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

## The Four Processes

### 1. Gateway (`gateway.py`)
The central message broker that all other processes connect to.

- **Location**: `jax_inference_offloading/controller/gateway.py`
- **Role**: Routes gRPC messages between processes
- **Features**:
  - Pub/sub messaging via topics
  - KV store for sharing configuration between processes
  - No GPU required

### 2. vLLM Worker (`vllm_worker.py`)
Runs the vLLM inference engine and executes inference requests.

- **Location**: `examples/decoupled_synchronous/vllm_worker.py`
- **Role**: Receives weights, executes inference, returns results
- **GPU**: Requires vLLM-assigned GPUs
- **Subscribes to**: `HANDSHAKE`, `CREATE_TRANSPORT`, `WEIGHT_UPDATES`, `INFERENCE_REQUESTS`

### 3. JAX Controller (`jax_controller.py`)
The "driver" process that orchestrates the training loop.

- **Location**: `examples/decoupled_synchronous/jax_controller.py`
- **Role**: Loads model, transfers weights, receives inference results
- **GPU**: Requires JAX-assigned GPUs
- **Key components used**:
  - `OffloadingSession` - manages gRPC connection and handshake
  - `VLLMTransferEngine` - transfers weights to vLLM

### 4. Prompt Dispatcher (`prompt_dispatcher.py`)
Lightweight process that sends inference requests.

- **Location**: `examples/decoupled_synchronous/prompt_dispatcher.py`
- **Role**: Waits for sync signal, loads prompts, sends inference requests
- **GPU**: None required (CPU only)
- **Key component used**:
  - `VLLMRolloutRequester` - sends inference requests to vLLM

## Building Blocks

These are the key classes you can reuse to build your own setup:

### VLLMTransferEngine

Handles GPU-to-GPU weight transfer from JAX to vLLM via NCCL.

```python
from jax_inference_offloading import OffloadingSession, VLLMTransferEngine

# Create session (performs handshake with vLLM)
session = OffloadingSession(
    gateway_url="localhost:50051",
    mesh=mesh,
    param_mapping_path="/path/to/mapping.json",
)

# Create transfer engine
transfer_engine = VLLMTransferEngine(session=session)

# Transfer weights (blocking call)
transfer_engine.update_weights(params)
```

**Key methods**:
- `update_weights(params)` - Transfers JAX parameters to vLLM. Accepts `Dict[str, jax.Array]`, `flax.nnx.State`, or `flax.nnx.Module`.

### VLLMRolloutRequester

Lightweight client for sending inference requests. No JAX dependency.

```python
from jax_inference_offloading import VLLMRolloutRequester, InferenceConfig

# Create requester (auto-discovers response topic from gateway KV)
requester = VLLMRolloutRequester(gateway_url="localhost:50051")

# Send inference request (fire-and-forget, non-blocking)
config = InferenceConfig(max_tokens=256, temperature=0.7)
requester.request(prompts=["Hello, world!"], config=config)
```

**Key methods**:
- `request(prompts, config)` - Sends inference request. Does not wait for response.

**Constructor options**:
- `gateway_url` - Gateway address
- `response_topic` - Optional. If not provided, polls gateway KV to discover it.

### OffloadingSession

Manages the gRPC connection and handshake with vLLM.

```python
from jax_inference_offloading import OffloadingSession

session = OffloadingSession(
    gateway_url="localhost:50051",
    mesh=jax_mesh,
    param_mapping_path="/path/to/mapping.json",
)

# Access gRPC stubs for custom operations
session.controller_stub  # For RPC calls (KVPut, KVGet, etc.)
session.broker_stub      # For pub/sub (Publish, SubscriptionStream)
```

## Synchronization Mechanism

The processes coordinate using **pub/sub messaging** through the gateway:

### Startup Coordination

1. **Gateway** starts first
2. **vLLM Worker** connects and subscribes to control topics
3. **JAX Controller** performs handshake, creates NCCL transport, stores response topic in KV:
   ```python
   session.controller_stub.KVPut(key="inference_response_topic", value=RESULTS_TOPIC)
   ```
4. **Prompt Dispatcher** polls KV until topic is available:
   ```python
   requester = VLLMRolloutRequester(gateway_url=gateway_url)  # Auto-polls KV
   ```

### Per-Iteration Synchronization

```
JAX Controller              Prompt Dispatcher           vLLM Worker
      │                            │                         │
      │  1. transfer_weights()     │                         │
      │────────────────────────────────────────────────────►│
      │                            │                   [receives weights]
      │                            │                         │
      │  2. publish("sync/weights_ready")                    │
      │─────────────────────────►│                          │
      │                     [receives signal]                │
      │                            │                         │
      │                     3. requester.request(prompts)    │
      │                            │────────────────────────►│
      │                            │                   [runs inference]
      │                            │                         │
      │  4. receive results        │                         │
      │◄─────────────────────────────────────────────────────│
      │                            │                         │
[process results]                  │                         │
      │                            │                         │
```

### Topics Used

| Topic | Publisher | Subscriber | Purpose |
|-------|-----------|------------|---------|
| `sync/weights_ready` | JAX Controller | Prompt Dispatcher | Signal that weights are ready |
| `inference/results/shared` | vLLM Worker | JAX Controller | Inference results |
| `inference_response_topic` (KV) | JAX Controller | Prompt Dispatcher | Share results topic |

## Data Flow

### Step 1: Weight Transfer (JAX → vLLM)
```
JAX Controller                           vLLM Worker
     │                                        │
     │  VLLMTransferEngine.update_weights()   │
     │                                        │
     │  ┌─────────────────────────────────┐   │
     │  │    NCCL GPU-to-GPU Transfer     │   │
     │  └─────────────────────────────────┘   │
     │ ──────────────────────────────────────►│
     │                                        │
```

### Step 2: Sync Signal (JAX → Prompt Dispatcher)
```
JAX Controller              Gateway              Prompt Dispatcher
     │                         │                        │
     │  Publish(sync/ready)    │                        │
     │────────────────────────►│                        │
     │                         │  SubscriptionStream    │
     │                         │───────────────────────►│
     │                         │                        │
```

### Step 3: Inference Request (Prompt Dispatcher → vLLM)
```
Prompt Dispatcher           Gateway              vLLM Worker
     │                         │                      │
     │  requester.request()    │                      │
     │────────────────────────►│                      │
     │                         │  AsyncInference      │
     │                         │─────────────────────►│
     │                         │                      │
```

### Step 4: Results (vLLM → JAX Controller)
```
vLLM Worker                 Gateway              JAX Controller
     │                         │                      │
     │  Publish(results)       │                      │
     │────────────────────────►│                      │
     │                         │  SubscriptionStream  │
     │                         │─────────────────────►│
     │                         │                      │
```

## Running the Example

```bash
cd examples/decoupled_synchronous

./decoupled_sync.sh \
  --model-path=/path/to/model \
  --param-mapping-path=../mappings/llama3_1b_param_mapping.json \
  --num-iterations=5
```

### Required Arguments
- `--model-path` - Path to HuggingFace model checkpoint
- `--param-mapping-path` - Path to JSON parameter mapping file

### Optional Arguments
- `--num-iterations=N` - Number of training iterations (default: 3)
- `--n-gpus-vllm=N` - GPUs for vLLM (default: 4)
- `--n-gpus-jax=N` - GPUs for JAX (default: 4)
- `--transfer-mode=MODE` - Weight transfer mode: `grouped`/`fused`/`unfused`
- `--gateway-port=PORT` - gRPC port (default: 50051)
- `--debug` - Enable verbose logging

## Customization Guide

### Replace Prompt Loading

Edit `prompt_dispatcher.py` to load prompts from your data source:

```python
def get_prompts():
    # Replace with your data loading logic
    # Examples:
    # - Load from dataset
    # - Read from message queue
    # - Generate dynamically
    return load_prompts_from_dataset()
```

### Add Training Logic

Edit `jax_controller.py` to add training after receiving results:

```python
# After receiving results
result = ctrl.InferenceResponse()
delivery.message.payload.Unpack(result)

# Add your training logic here
rewards = compute_rewards(result.outputs)
grads = compute_gradients(params, rewards)
params = update_params(params, grads)
```

### Use Different Models

1. Create a new parameter mapping JSON file for your model
2. Point `--param-mapping-path` to your mapping file
3. Ensure `--model-path` points to compatible checkpoint