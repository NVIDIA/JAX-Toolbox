"""CPU-only tests for the single-controller example scheduler.

The tests in this file validate single-controller scheduling behavior in sync mode. It verifies that:
- JAX and vLLM do not handshake until vLLM is ready
- JAX triggers a weight transfer to vLLM only when the single-controller signals it to.
- The prompt source only dispatches prompts after weight transfer / refit is complete. o
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import threading
import uuid

from google.protobuf.wrappers_pb2 import StringValue

import jax_inference_offloading.api.controller_pb2 as ctrl

from conftest import open_subscription, publish_proto, read_stream_message


def _load_single_controller_class():
    """Load the example controller class directly from its source file."""
    module_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "single_controller"
        / "single_controller.py"
    )
    module_name = f"single_controller_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.SingleController


def _start_single_controller(monkeypatch, gateway_url: str):
    """Start a single-controller instance with test-local topic names."""
    suffix = uuid.uuid4().hex
    topics = {
        "jax_commands": f"tests/sc/{suffix}/jax/commands",
        "jax_events": f"tests/sc/{suffix}/jax/events",
        "jax_results": f"tests/sc/{suffix}/jax/results",
        "sync": f"tests/sc/{suffix}/sync",
        "vllm_events": f"tests/sc/{suffix}/vllm/events",
    }

    monkeypatch.setenv("GATEWAY_URL", gateway_url)
    monkeypatch.setenv("SC_MODE", "sync")
    monkeypatch.setenv("NUM_ITERATIONS", "1")
    monkeypatch.setenv("SC_JAX_COMMAND_TOPIC", topics["jax_commands"])
    monkeypatch.setenv("SC_JAX_EVENT_TOPIC", topics["jax_events"])
    monkeypatch.setenv("SC_JAX_RESULTS_TOPIC", topics["jax_results"])
    monkeypatch.setenv("SC_SYNC_TOPIC", topics["sync"])
    monkeypatch.setenv("SC_VLLM_EVENT_TOPIC", topics["vllm_events"])

    controller = _load_single_controller_class()()
    thread = threading.Thread(target=controller.run, daemon=True, name="SingleController")
    thread.start()
    return thread, topics


def _stop_single_controller(gateway_url: str, thread: threading.Thread):
    """Publish a shutdown message and wait briefly for the controller to exit."""
    try:
        publish_proto(gateway_url, "shutdown", ctrl.ShutdownRequest(grace_period=0))
    except Exception:
        # The controller may already have initiated gateway shutdown as part of
        # the sync iteration completion path, so teardown should not hang or
        # fail trying to send a second shutdown request.
        pass
    thread.join(timeout=2)


def _publish_until_message(
    gateway_url: str,
    publish_topic: str,
    payload,
    stream,
    proto_cls,
    predicate,
    *,
    attempts: int = 10,
    timeout: float = 0.2,
):
    """Retry publishing a non-persistent event until the expected message appears."""
    last_message = None

    for _ in range(attempts):
        publish_proto(gateway_url, publish_topic, payload)
        try:
            message = read_stream_message(stream, proto_cls, timeout=timeout)
        except TimeoutError:
            continue

        last_message = message
        if predicate(message):
            return message

    if last_message is None:
        raise AssertionError(
            f"No matching message observed on subscribed stream after {attempts} attempts."
        )
    raise AssertionError(f"Observed unexpected message instead: {last_message!r}")


def test_single_controller_defers_handshake_until_vllm_ready(monkeypatch, gateway_url):
    """Verify the controller withholds handshake forwarding until vLLM is ready."""
    thread, topics = _start_single_controller(monkeypatch, gateway_url)

    sc_handshake_stream = open_subscription(gateway_url, "sc/handshake")
    response_topic = "tests/sc/handshake/response"
    response_stream = open_subscription(gateway_url, response_topic)

    try:
        request = ctrl.HandshakeRequest(
            response_topic=response_topic,
            jax_parallelism=ctrl.JaxParallelism(tp=1),
            param_mapping_path="/tmp/not-used-by-controller.json",
        )
        publish_proto(gateway_url, "handshake", request)

        # The handshake must be held until the controller observes the vLLM
        # worker's readiness event on the dedicated single-controller topic.
        try:
            read_stream_message(sc_handshake_stream, ctrl.HandshakeRequest, timeout=0.3)
        except TimeoutError:
            pass
        else:  # pragma: no cover - makes the failure mode explicit
            raise AssertionError("Handshake was forwarded before the vLLM ready event.")

        forwarded = _publish_until_message(
            gateway_url,
            topics["vllm_events"],
            StringValue(value="ready"),
            sc_handshake_stream,
            ctrl.HandshakeRequest,
            lambda message: message.response_topic == "sc/results",
        )

        assert forwarded.response_topic == "sc/results"

        publish_proto(
            gateway_url,
            "sc/results",
            ctrl.HandshakeResponse(
                jax_parallelism=ctrl.JaxParallelism(tp=1),
                vllm_parallelism=ctrl.VllmParallelism(tp=1),
            ),
        )
        bridged = read_stream_message(response_stream, ctrl.HandshakeResponse)

        assert bridged.jax_parallelism.tp == 1
        assert bridged.vllm_parallelism.tp == 1
    finally:
        _stop_single_controller(gateway_url, thread)
        sc_handshake_stream.close()
        response_stream.close()


def test_single_controller_sync_scheduler_commands_jax_and_forwards_results(
    monkeypatch, gateway_url
):
    """Verify sync-mode scheduling commands JAX and forwards inference results."""
    thread, topics = _start_single_controller(monkeypatch, gateway_url)

    commands_stream = open_subscription(gateway_url, topics["jax_commands"])
    sync_stream = open_subscription(gateway_url, topics["sync"])
    sc_request_stream = open_subscription(gateway_url, "sc/inference_requests")
    jax_results_stream = open_subscription(gateway_url, topics["jax_results"])

    try:
        command = _publish_until_message(
            gateway_url,
            topics["jax_events"],
            StringValue(value="ready"),
            commands_stream,
            StringValue,
            lambda message: message.value == "update_weights",
        )
        assert command.value == "update_weights"

        sync_signal = _publish_until_message(
            gateway_url,
            topics["jax_events"],
            StringValue(value="weights_updated"),
            sync_stream,
            StringValue,
            lambda message: message.value == "0",
        )
        assert sync_signal.value == "0"

        request = ctrl.InferenceRequest(response_topic="unused/by/controller")
        request.prompts.add(text_prompt="hello from sync mode")
        request.config.CopyFrom(ctrl.RolloutConfig(num_outputs=1))
        publish_proto(gateway_url, "inference_requests", request)

        forwarded_request = read_stream_message(sc_request_stream, ctrl.InferenceRequest)
        assert forwarded_request.response_topic == "sc/results"
        assert len(forwarded_request.prompts) == 1
        assert forwarded_request.prompts[0].text_prompt == "hello from sync mode"

        result = ctrl.InferenceResponse()
        output = result.outputs.add()
        output.generated_text = "controller-forwarded-output"
        output.generated_tokens.ids.extend([1, 2, 3])
        publish_proto(gateway_url, "sc/results", result)

        forwarded_result = read_stream_message(jax_results_stream, ctrl.InferenceResponse)
        assert len(forwarded_result.outputs) == 1
        assert forwarded_result.outputs[0].generated_text == "controller-forwarded-output"
    finally:
        _stop_single_controller(gateway_url, thread)
        commands_stream.close()
        sync_stream.close()
        sc_request_stream.close()
        jax_results_stream.close()

