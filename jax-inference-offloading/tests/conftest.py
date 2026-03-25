"""Shared CPU-only test fixtures for the offloading control plane.

These fixtures intentionally avoid real GPU workers and NCCL transfers. They
spin up the real gRPC gateway in-process and pair it with a small fake LLM
surface so tests can validate message routing, scheduling, and protobuf
shaping deterministically.
"""

from __future__ import annotations

from concurrent import futures
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import time
from types import SimpleNamespace
import threading
import types
from typing import Any, Callable
import uuid

from google.protobuf.wrappers_pb2 import StringValue
import grpc
import numpy as np
import pytest

import jax_inference_offloading.api.controller_pb2 as ctrl
import jax_inference_offloading.api.controller_pb2_grpc as ctrl_grpc
from jax_inference_offloading.api.message_broker_pb2 import (
    PublishRequest,
    SubscribeRequest,
)
import jax_inference_offloading.api.message_broker_pb2_grpc as broker_grpc


def _install_optional_dependency_stubs() -> None:
    """Install lightweight stubs for optional heavy runtime dependencies.

    The CPU-only tests exercise control-plane code that imports modules such as
    `vllm` and `cupy` at module-import time even though the tests never execute
    the real GPU paths. Stubbing them keeps the tests independent of a full GPU
    runtime.
    """

    try:
        import vllm  # noqa: F401
    except ImportError:
        vllm_module = types.ModuleType("vllm")
        vllm_module.LLM = object
        sys.modules["vllm"] = vllm_module

    try:
        from cupy.cuda import nccl  # noqa: F401
    except ImportError:
        cupy_module = types.ModuleType("cupy")
        cuda_module = types.ModuleType("cupy.cuda")
        nccl_module = types.ModuleType("cupy.cuda.nccl")
        nccl_module.get_unique_id = lambda: tuple(range(128))
        cuda_module.nccl = nccl_module
        cupy_module.cuda = cuda_module
        sys.modules["cupy"] = cupy_module
        sys.modules["cupy.cuda"] = cuda_module
        sys.modules["cupy.cuda.nccl"] = nccl_module


_install_optional_dependency_stubs()


class FakeTokenizer:
    """Deterministic tokenizer used by the fake rollout worker."""

    def encode(self, text: str, add_special_tokens: bool = True):
        del add_special_tokens
        return [101, len(text), 102]

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        add_special_tokens: bool = True,
    ):
        del tokenize, add_generation_prompt, add_special_tokens
        return [201, len(messages), 202]


class FakeLLM:
    """Small fake vLLM surface used by the rollout servicer tests."""

    def __init__(self):
        self.rpc_calls = []
        self.generate_calls = []
        self.reset_prefix_cache_calls = 0
        self.llm_engine = SimpleNamespace(
            engine_core=SimpleNamespace(shutdown=lambda: None)
        )

    def collective_rpc(self, method: str, timeout=None, args=()):
        """Record collective calls and return simple fake worker responses."""
        del timeout
        self.rpc_calls.append((method, args))
        if method == "get_tp_sharding_specs":
            return [(0, {})]
        return "ok"

    def get_tokenizer(self):
        """Return the fake tokenizer used by RolloutServicer."""
        return FakeTokenizer()

    def generate(self, prompts, sampling_params):
        """Return deterministic outputs shaped like vLLM responses."""
        self.generate_calls.append((prompts, sampling_params))

        responses = []
        for prompt_index, prompt in enumerate(prompts):
            prompt_token_ids = list(prompt["prompt_token_ids"])
            outputs = []
            for rollout_index in range(int(sampling_params.n)):
                token_ids = [prompt_index + 1, rollout_index + 11]
                logprobs = [
                    {token_id: SimpleNamespace(logprob=-(idx + 1) * 0.1)}
                    for idx, token_id in enumerate(token_ids)
                ]
                outputs.append(
                    SimpleNamespace(
                        text=f"prompt-{prompt_index}-rollout-{rollout_index}",
                        token_ids=token_ids,
                        logprobs=logprobs,
                    )
                )
            responses.append(
                SimpleNamespace(
                    prompt_token_ids=prompt_token_ids,
                    outputs=outputs,
                )
            )

        return responses

    def reset_prefix_cache(self):
        """Record prefix cache invalidations after weight updates."""
        self.reset_prefix_cache_calls += 1


@pytest.fixture
def gateway_url():
    """Start the real gateway server on an ephemeral localhost port."""
    from jax_inference_offloading.controller.gateway import (
        ControllerServicer,
        MessageBrokerServicer,
        MessageQueues,
    )

    # These tests keep several broker streams open at the same time. Use a
    # larger worker pool so unary RPCs like Publish and Shutdown are not
    # starved behind long-lived SubscriptionStream handlers.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    queues = MessageQueues()
    ctrl_grpc.add_CouplingControllerServicer_to_server(
        ControllerServicer(server, queues), server
    )
    broker_grpc.add_MessageBrokerServicer_to_server(
        MessageBrokerServicer(queues), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield f"127.0.0.1:{port}"
    finally:
        server.stop(0).wait()


@pytest.fixture
def mapping_json(tmp_path: Path) -> Path:
    """Create a minimal JSON mapping accepted by RolloutServicer.handshake()."""
    path = tmp_path / "param_mapping.json"
    path.write_text(
        json.dumps(
            {
                "mesh_axes": ["fsdp", "tp"],
                "mappings": [
                    {
                        "jax_param": {"name": "embedder.input_embedding"},
                        "vllm_param": {
                            "name": "model.embed_tokens.weight",
                            "shape": [8, 4],
                        },
                    }
                ],
            }
        )
    )
    return path


@pytest.fixture
def cpu_mesh():
    """Build a single-device mesh so handshake TP values stay CPU-friendly."""
    import jax

    device = jax.devices()[0]
    devices = np.asarray([device]).reshape((1, 1))
    return jax.sharding.Mesh(devices, ("fsdp", "tp"))


class SubscriptionHandle:
    """Background reader for a gRPC subscription stream.

    The reader thread continuously drains the stream into an in-memory queue so
    tests can perform timeout-based assertions without leaving blocked `next()`
    calls behind on the same iterator.
    """

    def __init__(self, gateway_url: str, topic_ids):
        self._channel = grpc.insecure_channel(gateway_url)
        grpc.channel_ready_future(self._channel).result(timeout=5)
        stub = broker_grpc.MessageBrokerStub(self._channel)
        request = SubscribeRequest()
        for topic_id in topic_ids:
            request.topics.add(id=topic_id)
        self._stream = stub.SubscriptionStream(request)
        self._queue = Queue()
        self._thread = threading.Thread(target=self._consume, daemon=True)
        self._thread.start()

    def _consume(self):
        try:
            for delivery in self._stream:
                self._queue.put(("ok", delivery))
        except Exception as exc:  # pragma: no cover - surfaced directly to tests
            self._queue.put(("error", exc))

    def read(self, proto_cls: Callable[[], Any], timeout: float = 2.0):
        """Read and unpack one protobuf payload from the subscription queue."""
        try:
            status, value = self._queue.get(timeout=timeout)
        except Empty as exc:
            raise TimeoutError(
                f"No message received within {timeout} seconds."
            ) from exc

        if status == "error":
            raise value

        message = proto_cls()
        value.message.payload.Unpack(message)
        return message

    def close(self):
        """Close the underlying gRPC channel."""
        self._channel.close()


def open_subscription(gateway_url: str, *topic_ids: str) -> SubscriptionHandle:
    """Subscribe to one or more broker topics."""
    return SubscriptionHandle(gateway_url, topic_ids)


def read_stream_message(stream, proto_cls: Callable[[], Any], timeout: float = 2.0):
    """Read and unpack one streamed protobuf message with a timeout.

    The subscription handle already has a dedicated background reader, so
    timeout assertions remain reliable and do not leak blocked reader calls.
    """
    return stream.read(proto_cls, timeout=timeout)


def publish_proto(gateway_url: str, topic_id: str, payload) -> None:
    """Publish a protobuf payload to a broker topic."""
    channel = grpc.insecure_channel(gateway_url)
    grpc.channel_ready_future(channel).result(timeout=5)
    try:
        stub = broker_grpc.MessageBrokerStub(channel)
        request = PublishRequest()
        request.topic.id = topic_id
        request.message.payload.Pack(payload)
        stub.Publish(request, timeout=2)
    finally:
        channel.close()


def put_kv_string(gateway_url: str, key: str, value: str) -> None:
    """Store a string in the gateway KV store for requester discovery."""
    channel = grpc.insecure_channel(gateway_url)
    grpc.channel_ready_future(channel).result(timeout=5)
    try:
        stub = ctrl_grpc.CouplingControllerStub(channel)
        request = ctrl.KVPutRequest(key=key)
        request.value.Pack(StringValue(value=value))
        stub.KVPut(request, timeout=2)
    finally:
        channel.close()


def perform_handshake(gateway_url: str, mapping_json_path: str, jax_tp: int = 1):
    """Perform a minimal handshake and return the `HandshakeResponse`.

    Async weight-update paths assume the rollout servicer has already processed
    at least one handshake so `_mapping_specs` is populated.
    """
    response_topic = f"tests/handshake/{uuid.uuid4().hex}"
    response_stream = open_subscription(gateway_url, response_topic)

    channel = grpc.insecure_channel(gateway_url)
    grpc.channel_ready_future(channel).result(timeout=5)
    try:
        stub = ctrl_grpc.CouplingControllerStub(channel)
        request = ctrl.HandshakeRequest(
            response_topic=response_topic,
            jax_parallelism=ctrl.JaxParallelism(tp=jax_tp),
            param_mapping_path=mapping_json_path,
        )
        stub.AsyncHandshake(request, timeout=2)
        return read_stream_message(response_stream, ctrl.HandshakeResponse)
    finally:
        response_stream.close()
        channel.close()


def start_weight_update(gateway_url: str, mode: str = "grouped") -> None:
    """Trigger a weight update RPC to exercise async queue draining."""
    channel = grpc.insecure_channel(gateway_url)
    grpc.channel_ready_future(channel).result(timeout=5)
    try:
        stub = ctrl_grpc.CouplingControllerStub(channel)
        stub.StartWeightUpdate(ctrl.StartWeightUpdateRequest(mode=mode), timeout=2)
    finally:
        channel.close()


def close_rollout_client(client) -> None:
    """Best-effort cleanup for rollout clients created in tests."""
    try:
        client._channel.close()
    except Exception:
        pass

    try:
        client._executor.shutdown(wait=True, cancel_futures=True)
    except TypeError:
        client._executor.shutdown(wait=True)


def visible_cuda_device_ids() -> list[str]:
    """Return the CUDA device identifiers visible to the current process."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [device_id.strip() for device_id in visible.split(",") if device_id.strip()]

    try:
        import jax

        return [str(index) for index, _ in enumerate(jax.devices("gpu"))]
    except Exception:
        return []


@pytest.fixture
def gpu_device_partitions() -> dict[str, str]:
    """Split eight visible GPUs into JAX and rollout groups for subprocess tests."""
    device_ids = visible_cuda_device_ids()
    if len(device_ids) < 8:
        pytest.skip("GPU grouped transfer tests require at least 8 visible CUDA devices.")

    return {
        "jax": ",".join(device_ids[:4]),
        "rollout": ",".join(device_ids[4:8]),
    }


@pytest.fixture
def gpu_test_script() -> Path:
    """Return the helper script used by the GPU subprocess test."""
    return Path(__file__).with_name("gpu_test_utils.py")


def build_gpu_subprocess_env(cuda_visible_devices: str) -> dict[str, str]:
    """Create a subprocess environment pinned to a specific GPU subset."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def wait_for_path(path: Path, timeout: float = 30.0) -> None:
    """Poll until a path exists or raise a TimeoutError."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.1)

    raise TimeoutError(f"{path} was not created within {timeout} seconds.")


def format_subprocess_failure(
    command: list[str], stdout: str | None, stderr: str | None
) -> str:
    """Build a compact failure message for subprocess-based test helpers."""
    rendered = [
        f"Command failed: {' '.join(command)}",
        f"stdout:\n{stdout or '<empty>'}",
        f"stderr:\n{stderr or '<empty>'}",
    ]
    return "\n\n".join(rendered)


def wait_for_process(
    process: subprocess.Popen[str], timeout: float, command: list[str]
) -> tuple[str, str]:
    """Wait for a subprocess and raise a readable assertion on failure."""
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        raise AssertionError(
            format_subprocess_failure(command, stdout, stderr)
        ) from None

    if process.returncode != 0:
        raise AssertionError(format_subprocess_failure(command, stdout, stderr))

    return stdout, stderr

