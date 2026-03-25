"""CPU-only handshake test for the offloading session.

This test validates the following:
- the gateway correctly routed the handshake request and response between the JAX and vLLM processes
- the OffloadingSession actually performed the handshake during initialization
- the parameter mapping JSON was loaded and returned through the protobuf response and that the returned mapping was parsed into session.mapping_specs
- JAX sharding was computed from the mesh initialized on the JAX side
- vLLM TP was computed on the vLLM side and round-tripped correctltly. 
"""

from __future__ import annotations

from jax_inference_offloading import OffloadingSession
from jax_inference_offloading.controller.rollout_client import make_rollout_client

from conftest import FakeLLM, close_rollout_client


def test_offloading_session_handshake_returns_mapping_and_tp_sizes(
    gateway_url, mapping_json, cpu_mesh
):
    """Verify that `OffloadingSession` can complete a real handshake round-trip."""
    llm = FakeLLM()
    rollout_client = make_rollout_client(gateway_url)
    rollout_client.subscribe_to_control_messages(
        llm, mapping_json_path=str(mapping_json)
    )
    session = None

    try:
        session = OffloadingSession(
            gateway_url=gateway_url,
            mesh=cpu_mesh,
            param_mapping_path=str(mapping_json),
        )

        # The fake worker reports one TP rank and returns the mapping from the
        # temporary JSON file, which should be parsed into the session state.
        assert session.jax_parallelism.tp == 1
        assert session.vllm_parallelism.tp == 1
        assert len(session.mapping_specs.mappings) == 1
        assert (
            session.mapping_specs.mappings[0].jax_param.name
            == "embedder.input_embedding"
        )
        assert (
            session.mapping_specs.mappings[0].vllm_param.name
            == "model.embed_tokens.weight"
        )
    finally:
        if session is not None:
            session.shutdown(shutdown_gateway=False)
        close_rollout_client(rollout_client)

