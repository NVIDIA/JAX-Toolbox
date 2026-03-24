"""CPU-only async streaming tests for bounded staleness behavior.

This test extends the sync roundtrip test to validate the async streaming behavior with bounded staleness.
It verifies the following:
- the async rollout client enforces the set max_staleness (2 in this case)
- that multi-prompt requests can be partially processed and that the unprocessed remaining prompts are queued and processed after the weight update
"""

from __future__ import annotations

import pytest

from jax_inference_offloading import InferenceConfig, VLLMRolloutRequester
import jax_inference_offloading.api.controller_pb2 as ctrl
from jax_inference_offloading.controller.rollout_client import make_async_rollout_client

from conftest import (
    FakeLLM,
    close_rollout_client,
    open_subscription,
    perform_handshake,
    read_stream_message,
    start_weight_update,
)


def test_async_rollout_client_queues_remaining_prompts_until_weight_update(
    gateway_url, mapping_json
):
    """Verify async streaming respects `max_staleness` and drains queued work later."""
    llm = FakeLLM()
    rollout_client = make_async_rollout_client(gateway_url, max_staleness=2)
    rollout_client.subscribe_to_control_messages(
        llm, mapping_json_path=str(mapping_json)
    )

    results_topic = "tests/results/async"
    results_stream = open_subscription(gateway_url, results_topic)
    requester = VLLMRolloutRequester(
        gateway_url=gateway_url,
        response_topic=results_topic,
    )

    try:
        config = InferenceConfig(max_tokens=8, n=1)

        # Weight updates are only valid after the rollout side has seen a
        # handshake and captured mapping specs, mirroring the real startup flow.
        perform_handshake(gateway_url, str(mapping_json))

        # First request consumes one prompt of the two-prompt staleness budget.
        requester.request(
            prompts=["prompt-0"],
            config=config,
            batch_id="batch-1",
            streaming=True,
        )
        # The second request has two prompts, so only the first can run before
        # the worker must queue the remainder and wait for a weight update.
        requester.request(
            prompts=["prompt-1", "prompt-2"],
            config=config,
            batch_id="batch-2",
            streaming=True,
        )

        first = read_stream_message(results_stream, ctrl.RolloutResult)
        second = read_stream_message(results_stream, ctrl.RolloutResult)

        assert (first.batch_id, first.prompt_index, first.rollout_index) == ("batch-1", 0, 0)
        assert (second.batch_id, second.prompt_index, second.rollout_index) == ("batch-2", 0, 0)

        # No third result should appear until the update resets staleness and
        # lets the queued remainder of batch-2 resume.
        with pytest.raises(TimeoutError):
            read_stream_message(results_stream, ctrl.RolloutResult, timeout=0.3)

        start_weight_update(gateway_url, mode="grouped")

        resumed = read_stream_message(results_stream, ctrl.RolloutResult)

        assert (resumed.batch_id, resumed.prompt_index, resumed.rollout_index) == (
            "batch-2",
            1,
            0,
        )
        assert llm.reset_prefix_cache_calls == 1
        assert any(method == "update_weights_grouped" for method, _ in llm.rpc_calls)
    finally:
        requester.shutdown()
        results_stream.close()
        close_rollout_client(rollout_client)

