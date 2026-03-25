"""Unit-style CPU tests for streamed rollout grouping.

There are two tests in this file:
- the first test ensures that a group is not emitted by the RolloutAccumulator until all rollouts for a prompt are present in an async setup
- the second test ensures that on the consumer (JAX) side, that an update is not performed until the expected number of groups are completed.
"""

from __future__ import annotations

import jax_inference_offloading.api.controller_pb2 as ctrl
from jax_inference_offloading.engines import RolloutAccumulator


def _make_result(batch_id: str, prompt_index: int, rollout_index: int) -> ctrl.RolloutResult:
    """Build a small `RolloutResult` message for accumulator tests."""
    return ctrl.RolloutResult(
        batch_id=batch_id,
        prompt_index=prompt_index,
        rollout_index=rollout_index,
    )


def test_rollout_accumulator_emits_groups_only_after_all_rollouts_arrive():
    """Verify groups complete only when every rollout for a prompt is present."""
    accumulator = RolloutAccumulator(num_rollouts=2)

    accumulator.add_result(_make_result("batch-a", 0, 0))
    assert accumulator.pending_count == 1
    assert accumulator.get_completed_groups(timeout=0.05) == []

    accumulator.add_result(_make_result("batch-a", 0, 1))
    completed = accumulator.get_completed_groups(timeout=0.05)

    # The first prompt group becomes available only after both rollout indices
    # have been observed for the same `(batch_id, prompt_index)` key.
    assert len(completed) == 1
    batch_id, prompt_index, group = completed[0]
    assert (batch_id, prompt_index) == ("batch-a", 0)
    assert [result.rollout_index for result in group] == [0, 1]


def test_rollout_accumulator_can_return_multiple_completed_groups():
    """Verify the accumulator can hand back multiple ready prompt groups at once."""
    accumulator = RolloutAccumulator(num_rollouts=2)

    accumulator.add_result(_make_result("batch-a", 0, 0))
    accumulator.add_result(_make_result("batch-b", 1, 0))
    accumulator.add_result(_make_result("batch-a", 0, 1))
    accumulator.add_result(_make_result("batch-b", 1, 1))

    groups = accumulator.get_completed_groups(timeout=0.05)

    assert len(groups) == 2
    assert {(batch_id, prompt_index) for batch_id, prompt_index, _ in groups} == {
        ("batch-a", 0),
        ("batch-b", 1),
    }
    assert accumulator.pending_count == 0
    assert accumulator.completed_queue_size == 0

