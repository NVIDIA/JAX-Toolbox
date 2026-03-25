"""Helpers and subprocess entrypoints for GPU grouped-transfer tests."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import traceback

import grpc
import numpy as np

import jax_inference_offloading.api.controller_pb2 as ctrl
import jax_inference_offloading.api.message_broker_pb2 as broker
import jax_inference_offloading.api.message_broker_pb2_grpc as broker_grpc
import jax_inference_offloading.api.param_mapping_pb2 as mapping_pb2
from jax_inference_offloading.controller import utils as ctrl_utils

GROUPED_PARAM_NAME = "param_a"
GROUPED_PARAM_SHAPE = (8, 2)
GROUPED_PARAM_DTYPE = "float32"
ROLLOUT_RANKS = 4


def build_global_param() -> np.ndarray:
    """Create a tiny tensor whose shard ordering is easy to verify by eye."""
    return np.arange(np.prod(GROUPED_PARAM_SHAPE), dtype=np.float32).reshape(
        GROUPED_PARAM_SHAPE
    )


def build_expected_rollout_tensors() -> dict[str, dict[str, list[list[float]]]]:
    """Return the per-rank tensors the rollout side should observe."""
    shards = np.split(build_global_param(), ROLLOUT_RANKS, axis=0)
    return {
        str(rank): {GROUPED_PARAM_NAME: shard.tolist()}
        for rank, shard in enumerate(shards)
    }


def build_dummy_mapping_json() -> dict[str, object]:
    """Create a minimal JSON mapping path for `OffloadingSession` validation."""
    return {
        "mesh_axes": ["tp"],
        "mappings": [
            {
                "jax_param": {"name": GROUPED_PARAM_NAME},
                "vllm_param": {
                    "name": GROUPED_PARAM_NAME,
                    "shape": list(GROUPED_PARAM_SHAPE),
                },
            }
        ],
    }


def build_grouped_mapping_specs() -> mapping_pb2.TpModelMappingSpecs:
    """Create the mapping specs returned during the receiver handshake."""
    specs = mapping_pb2.TpModelMappingSpecs()
    specs.mesh_axes.extend(["tp"])

    mapping = specs.mappings.add()
    mapping.jax_param.name = GROUPED_PARAM_NAME
    mapping.vllm_param.name = GROUPED_PARAM_NAME
    mapping.vllm_param.shape.extend(GROUPED_PARAM_SHAPE)
    mapping.vllm_param.dtype = GROUPED_PARAM_DTYPE
    mapping.vllm_param.tp_sharding.dim = 0
    mapping.vllm_param.tp_sharding.parallelism = ROLLOUT_RANKS
    mapping.vllm_param.tp_sharding.aux_dim = 1
    mapping.vllm_param.tp_sharding.aux_parallelism = 1
    return specs


def _publish_message(
    stub: broker_grpc.MessageBrokerStub, topic_id: str, payload
) -> None:
    request = broker.PublishRequest()
    request.topic.id = topic_id
    request.message.payload.Pack(payload)
    stub.Publish(request, timeout=10)


def _build_grouped_param_specs(config_mode: str):
    """Mirror the rollout-side grouped receive shape logic."""
    specs = build_grouped_mapping_specs()
    param_specs = []
    names = []

    for mapping in specs.mappings:
        names.append(mapping.vllm_param.name)
        sharding = mapping.vllm_param.tp_sharding
        shape = np.array(mapping.vllm_param.shape, dtype=np.int32)

        if config_mode == "fan-in":
            if sharding.parallelism > 0:
                shape[sharding.dim] //= sharding.parallelism
            param_specs.append(
                (
                    shape,
                    mapping.vllm_param.dtype,
                    sharding.aux_dim,
                    sharding.aux_parallelism,
                )
            )
        else:
            param_specs.append((shape, mapping.vllm_param.dtype))

    return names, param_specs


def _receiver_worker(rank: int, command_queue, result_queue) -> None:
    """Run one rollout TP rank pinned to one visible CUDA device."""
    import torch
    from cupy import cuda
    from jax_inference_offloading.transport.tensor.nccl_star import NcclStarTransport

    transport = None
    config_mode = None

    torch.cuda.set_device(rank)
    cuda.Device(rank).use()

    while True:
        command, payload = command_queue.get()
        if command == "create_transport":
            config = json.loads(payload)
            config_mode = config["MODE"]
            cuda.Device(rank).use()
            transport = NcclStarTransport.create_rollout_transport(config, tp_rank=rank)
            result_queue.put(("transport_ready", rank))
            continue

        if command == "receive_grouped":
            if transport is None or config_mode is None:
                raise RuntimeError("Transport must be created before receiving weights.")

            names, param_specs = _build_grouped_param_specs(config_mode)
            if config_mode == "fan-in":
                tensors = transport.gather_grouped(param_specs)
            else:
                tensors = transport.recv_grouped(param_specs)

            serialized = {
                name: tensor.detach().cpu().numpy().tolist()
                for name, tensor in zip(names, tensors)
            }
            result_queue.put(("received", rank, serialized))
            return

        if command == "stop":
            return

        raise ValueError(f"Unknown worker command: {command}")


def run_receiver(gateway_url: str, ready_file: str, result_json: str) -> None:
    """Serve handshake/control messages and collect rollout-side tensors."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_count = len([device for device in visible_devices.split(",") if device])
    if visible_count != ROLLOUT_RANKS:
        raise ValueError(
            f"Receiver expects exactly {ROLLOUT_RANKS} visible GPUs, got {visible_devices!r}."
        )

    ctx = mp.get_context("spawn")
    command_queues = [ctx.Queue() for _ in range(ROLLOUT_RANKS)]
    result_queue = ctx.Queue()
    workers = [
        ctx.Process(
            target=_receiver_worker,
            args=(rank, command_queue, result_queue),
        )
        for rank, command_queue in enumerate(command_queues)
    ]
    for worker in workers:
        worker.start()

    channel = grpc.insecure_channel(gateway_url)
    grpc.channel_ready_future(channel).result(timeout=30)
    broker_stub = broker_grpc.MessageBrokerStub(channel)

    request = broker.SubscribeRequest(
        topics=[
            ctrl_utils.HANDSHAKE,
            ctrl_utils.CREATE_TRANSPORT,
            ctrl_utils.WEIGHT_UPDATES,
        ]
    )
    stream = broker_stub.SubscriptionStream(request)
    Path(ready_file).write_text("ready")

    try:
        for delivery in stream:
            if delivery.topic.id == ctrl_utils.HANDSHAKE.id:
                handshake = ctrl.HandshakeRequest()
                delivery.message.payload.Unpack(handshake)

                response = ctrl.HandshakeResponse()
                response.jax_parallelism.CopyFrom(handshake.jax_parallelism)
                response.vllm_parallelism.tp = ROLLOUT_RANKS
                response.mapping_specs.CopyFrom(build_grouped_mapping_specs())
                _publish_message(broker_stub, handshake.response_topic, response)
                continue

            if delivery.topic.id == ctrl_utils.CREATE_TRANSPORT.id:
                create_transport = ctrl.CreateTransportRequest()
                delivery.message.payload.Unpack(create_transport)

                for command_queue in command_queues:
                    command_queue.put(("create_transport", create_transport.config_json))
                for _ in range(ROLLOUT_RANKS):
                    kind, _rank = result_queue.get(timeout=60)
                    if kind != "transport_ready":
                        raise RuntimeError(f"Unexpected worker result: {kind!r}")
                continue

            if delivery.topic.id == ctrl_utils.WEIGHT_UPDATES.id:
                weight_update = ctrl.StartWeightUpdateRequest()
                delivery.message.payload.Unpack(weight_update)
                if weight_update.mode != "grouped":
                    raise ValueError(
                        f"Expected grouped update, received {weight_update.mode!r}."
                    )

                for command_queue in command_queues:
                    command_queue.put(("receive_grouped", None))

                received = {}
                for _ in range(ROLLOUT_RANKS):
                    kind, rank, tensors = result_queue.get(timeout=120)
                    if kind != "received":
                        raise RuntimeError(f"Unexpected worker result: {kind!r}")
                    received[str(rank)] = tensors

                Path(result_json).write_text(json.dumps(received, sort_keys=True))
                return

    finally:
        channel.close()
        for command_queue in command_queues:
            command_queue.put(("stop", None))
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)


def run_sender(gateway_url: str, mapping_json_path: str) -> None:
    """Use the real JAX-side session and grouped transfer engine."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec
    from jax.sharding import Mesh
    from jax_inference_offloading.engines.vllm_transfer_engine import (
        VLLMTransferEngine,
    )
    from jax_inference_offloading.session import OffloadingSession

    visible_devices = jax.devices("gpu")
    if len(visible_devices) != ROLLOUT_RANKS:
        raise ValueError(
            f"Sender expects exactly {ROLLOUT_RANKS} visible devices, got {visible_devices}."
        )

    mesh = Mesh(np.asarray(visible_devices), ("tp",))
    param = jax.device_put(
        jnp.asarray(build_global_param()),
        NamedSharding(mesh, PartitionSpec("tp", None)),
    )

    session = OffloadingSession(
        gateway_url=gateway_url,
        mesh=mesh,
        param_mapping_path=mapping_json_path,
    )
    try:
        transfer_engine = VLLMTransferEngine(session, transfer_mode="grouped")
        transfer_engine.update_weights({GROUPED_PARAM_NAME: param})
    finally:
        session.shutdown(shutdown_gateway=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    receiver = subparsers.add_parser("receiver")
    receiver.add_argument("--gateway-url", required=True)
    receiver.add_argument("--ready-file", required=True)
    receiver.add_argument("--result-json", required=True)

    sender = subparsers.add_parser("sender")
    sender.add_argument("--gateway-url", required=True)
    sender.add_argument("--mapping-json-path", required=True)

    args = parser.parse_args()

    try:
        if args.command == "receiver":
            run_receiver(
                gateway_url=args.gateway_url,
                ready_file=args.ready_file,
                result_json=args.result_json,
            )
        else:
            run_sender(
                gateway_url=args.gateway_url,
                mapping_json_path=args.mapping_json_path,
            )
    except Exception:  # pragma: no cover - exercised by subprocess exit handling
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
