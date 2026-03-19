"""CPU-only synchronous request/response test.

These test validates the following:
- The VLLMRolloutRequester can discover the response topic from the gateway KV store.
- A non-streaming inference request gets routed through the gateway to the rollout worker.
- The rollout worker returns one aggregated InferenceResponse containing all prompt/rollout outputs.
"""

from __future__ import annotations

from jax_inference_offloading import InferenceConfig, VLLMRolloutRequester
import jax_inference_offloading.api.controller_pb2 as ctrl
from jax_inference_offloading.controller.rollout_client import make_rollout_client

from conftest import (
    FakeLLM,
    close_rollout_client,
    open_subscription,
    put_kv_string,
    read_stream_message,
)


def test_sync_requester_discovers_kv_topic_and_receives_batched_response(
    gateway_url, mapping_json
):
    """Verify the non-streaming requester path returns one aggregated response."""
    llm = FakeLLM()
    rollout_client = make_rollout_client(gateway_url)
    rollout_client.subscribe_to_control_messages(
        llm, mapping_json_path=str(mapping_json)
    )

    results_topic = "tests/results/sync"
    put_kv_string(gateway_url, "inference_response_topic", results_topic)
    results_stream = open_subscription(gateway_url, results_topic)

    requester = VLLMRolloutRequester(gateway_url=gateway_url)

    try:
        requester.request(
            prompts=["hello", "world"],
            config=InferenceConfig(max_tokens=8, n=2),
            streaming=False,
        )

        response = read_stream_message(results_stream, ctrl.InferenceResponse)

        # The non-streaming path should bundle all prompt/rollout pairs into one
        # protobuf response rather than sending incremental rollout messages.
        assert len(response.outputs) == 4
        assert response.outputs[0].generated_text == "prompt-0-rollout-0"
        assert list(response.outputs[0].tokenized_prompt.ids) == [101, 5, 102]
        assert response.outputs[3].generated_text == "prompt-1-rollout-1"
    finally:
        requester.shutdown()
        results_stream.close()
        close_rollout_client(rollout_client)

