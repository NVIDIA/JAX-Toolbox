"""GPU test for the grouped NCCL weight-transfer data plane.

- This test verifies the correctness of the grouped NCCL transfer path.
- JAX sends tensors to the rollout side via a NCCL grouped transfer
- The rollout side receives the tensors and writes them to a JSON file.
- The test verifies that the sent tensors match the tensors written to the JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from conftest import (
    build_gpu_subprocess_env,
    format_subprocess_failure,
    wait_for_path,
    wait_for_process,
)
from gpu_test_utils import (
    GROUPED_PARAM_NAME,
    build_dummy_mapping_json,
    build_expected_rollout_tensors,
)


@pytest.mark.gpu
def test_grouped_transfer_moves_expected_tensor_values(
    gateway_url: str,
    gpu_device_partitions: dict[str, str],
    gpu_test_script: Path,
    tmp_path: Path,
):
    """Verify grouped correctness by checking the rollout-side tensor values exactly.

    The test launches a lightweight rollout receiver on four GPUs and a JAX
    sender on four different GPUs. The receiver performs the normal handshake,
    creates the rollout-side NCCL transport from the controller config, and then
    receives grouped weights so the test can compare the observed per-rank
    tensors against explicit expected values.
    """
    repo_root = Path(__file__).resolve().parents[1]
    mapping_json = tmp_path / "grouped_mapping.json"
    mapping_json.write_text(json.dumps(build_dummy_mapping_json()))

    ready_file = tmp_path / "receiver.ready"
    result_json = tmp_path / "received_tensors.json"

    # The rollout subprocess owns the receive side of the handshake and NCCL setup.
    receiver_command = [
        sys.executable,
        str(gpu_test_script),
        "receiver",
        "--gateway-url",
        gateway_url,
        "--ready-file",
        str(ready_file),
        "--result-json",
        str(result_json),
    ]
    receiver_process = subprocess.Popen(
        receiver_command,
        cwd=repo_root,
        env=build_gpu_subprocess_env(gpu_device_partitions["rollout"]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        wait_for_path(ready_file, timeout=30)

        # The JAX subprocess performs the real grouped transfer on a disjoint GPU set.
        sender_command = [
            sys.executable,
            str(gpu_test_script),
            "sender",
            "--gateway-url",
            gateway_url,
            "--mapping-json-path",
            str(mapping_json),
        ]
        try:
            sender_result = subprocess.run(
                sender_command,
                cwd=repo_root,
                env=build_gpu_subprocess_env(gpu_device_partitions["jax"]),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=240,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AssertionError(
                format_subprocess_failure(
                    sender_command,
                    exc.stdout if exc.stdout is not None else "",
                    exc.stderr if exc.stderr is not None else "",
                )
            ) from None

        if sender_result.returncode != 0:
            raise AssertionError(
                format_subprocess_failure(
                    sender_command,
                    sender_result.stdout,
                    sender_result.stderr,
                )
            )

        wait_for_process(receiver_process, timeout=240, command=receiver_command)
        wait_for_path(result_json, timeout=5)

        actual = json.loads(result_json.read_text())
        expected = build_expected_rollout_tensors()

        assert set(actual) == set(expected)
        for rank, expected_tensors in expected.items():
            assert set(actual[rank]) == {GROUPED_PARAM_NAME}
            observed = np.asarray(actual[rank][GROUPED_PARAM_NAME], dtype=np.float32)
            target = np.asarray(expected_tensors[GROUPED_PARAM_NAME], dtype=np.float32)
            assert observed.shape == target.shape
            np.testing.assert_array_equal(observed, target)

    finally:
        if receiver_process.poll() is None:
            receiver_process.kill()
            receiver_process.communicate()
