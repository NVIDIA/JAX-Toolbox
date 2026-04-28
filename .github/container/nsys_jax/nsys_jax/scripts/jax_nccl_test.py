#!/usr/bin/env python
import argparse
from cuda.bindings.driver import (  # type: ignore
    cuCtxGetDevice_v2,
    cuDeviceGetCount,
    cuDevicePrimaryCtxRetain,
    cuEventCreate,
    cuEventElapsedTime,
    cuEventRecord,
    cuEventSynchronize,
    cuGetErrorName,
    cuInit,
    cuMemGetInfo,
    cuProfilerStart,
    cuProfilerStop,
    cuStreamGetCtx,
    CUevent,
    CUevent_flags,
    CUresult,
)
from functools import partial
import jax
from jax.experimental.buffer_callback import buffer_callback
from jax.experimental.mesh_utils import create_device_mesh
from jax.experimental.multihost_utils import process_allgather
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import os
import random
import time
from uncertainties import ufloat  # type: ignore


def int_or_env(value) -> int:
    try:
        return int(value)
    except ValueError:
        return int(os.environ[value])


def checkCudaErrors(result):
    if result[0].value:
        if isinstance(result[0].value, CUresult):
            err, name = cuGetErrorName(result[0].value)
            raise RuntimeError(
                "CUDA error code={}({})".format(
                    result[0].value,
                    name if err == CUresult.CUDA_SUCCESS else "<unknown>",
                )
            )
        raise RuntimeError("Unknown error type: {}".format(result[0].value))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


_MEASUREMENT_DTYPE = jnp.float32
_MINIMUM_ERROR = 0.01  # ms; avoid printing silly numbers of significant figures


class BandwidthTable:
    """
    This is rolled by hand so it can print out live.
    """

    def __init__(
        self,
        *,
        collectives: list[str],
        max_element_count: int,
        num_participants: int,
    ):
        self._collectives = collectives
        self._element_size = jnp.dtype(_MEASUREMENT_DTYPE).itemsize
        self._num_participants = num_participants
        self._size_heading = "Size [B]"
        self._size_width = max(
            len(self._size_heading),
            len(f"{max_element_count * self._element_size:,}"),
        )
        self._bus_bw_heading = "busbw [GB/s]"
        self._bus_bw_width = max(len(self._bus_bw_heading), 15)
        self._time_heading = "time [ms]"
        self._time_width = max(len(self._time_heading), 12)
        self._coll_heading = f"{self._time_heading:>{self._time_width}} | {self._bus_bw_heading:>{self._bus_bw_width}}"
        self._coll_width = len(self._coll_heading)
        self._header = [
            " | ".join(
                [" " * self._size_width]
                + [f"{coll:^{self._coll_width}}" for coll in self._collectives]
            ),
            " | ".join(
                [f"{self._size_heading:<{self._size_width}}"]
                + [self._coll_heading] * len(self._collectives)
            ),
        ]
        assert all(len(self._header[0]) == len(h) for h in self._header[1:])

    def _collective_correction(self, kind: str) -> tuple[float, float]:
        """
        Calculate the correction factor from algorithm bandwidth to bus bandwidth, see:
        https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bus-bandwidth
        """
        match kind:
            # For AllGather the size in the bandwidth calculation is the total/output size, but the nominal element count is the per-rank/input size
            # https://github.com/NVIDIA/nccl-tests/blob/c6afef0b6f76ffc55d4172d971be6cf5a08a73a4/doc/PERFORMANCE.md#allgather
            case "all_gather":
                return (
                    self._num_participants,
                    (self._num_participants - 1) / self._num_participants,
                )
            case "all_reduce":
                return (1, 2 * (self._num_participants - 1) / self._num_participants)
            case "broadcast":
                return (1, 1)
            case "permute":
                return (1, 1)
            # For ReduceScatter the size in the bandwidth calculation is the total/input size, but the nominal element count is the per-rank/output size
            # https://github.com/NVIDIA/nccl-tests/blob/c6afef0b6f76ffc55d4172d971be6cf5a08a73a4/doc/PERFORMANCE.md#reducescatter
            case "reduce_scatter":
                return (
                    self._num_participants,
                    (self._num_participants - 1) / self._num_participants,
                )
            case _:
                assert False, f"Unknown collective kind {kind}"

    def _format_size(self, element_count):
        return f"{element_count * self._element_size:<{self._size_width},}"

    def print_header(self):
        print("-" * len(self._header[0]))
        print(f"{self._num_participants}-device collective performance")
        print("-" * len(self._header[0]))
        for h in self._header:
            print(h, flush=True)

    def print_footer(self):
        print("-" * len(self._header[0]), flush=True)

    def print_row(
        self, *, element_count: int, collective_times: dict[str, list[float]]
    ):
        # Make sure we didn't get measurements for a collective we didn't print a header for.
        assert len(collective_times.keys() - self._collectives) == 0
        # Make sure we got the same number of measurements for each collective that had data for this `element_count`
        op_measurement_counts = set(map(len, collective_times.values()))
        assert len(op_measurement_counts) == 1, op_measurement_counts
        # If we have M measurements (each from one GPU) and collective size N (self._num_participants), assume that:
        # - If M>=N this is M // N measurements of all N members in each collective.
        # - If M< N this is the first M members of a single collective.
        num_measurements = next(iter(op_measurement_counts))  # M
        if num_measurements >= self._num_participants:
            assert num_measurements % self._num_participants == 0, (
                num_measurements,
                self._num_participants,
            )
            shape = (
                self._num_participants,  # N
                num_measurements // self._num_participants,  # M // N
            )
        else:
            shape = (
                num_measurements,
                1,
            )
        runtime_strs = []
        for op in self._collectives:
            if op in collective_times:
                op_timings = np.reshape(collective_times[op], shape)
                # Take the minimum across participants in each replica; the assumption is that this subtracts launch jitter
                replica_mins = np.min(op_timings, axis=0)
                runtime_ms = ufloat(
                    np.mean(replica_mins),
                    max(np.std(replica_mins), _MINIMUM_ERROR),
                )
                bw_correction, bus_correction = self._collective_correction(op)
                # Calculate bandwidth in GB/s (not GiB/s)
                alg_bw = (
                    bw_correction
                    * element_count
                    * self._element_size
                    * 1e-6
                    / runtime_ms
                )
                bus_bw = alg_bw * bus_correction
                runtime_strs.append(
                    f"{runtime_ms:>{self._time_width}S} | {bus_bw:>{self._bus_bw_width}S}"
                )
            else:
                runtime_strs.append("-" * self._coll_width)
        print(" | ".join([self._format_size(element_count)] + runtime_strs), flush=True)


def stream_event_timer_data(events, callback=None):
    # Extract the list of local devices, collectives and element_counts
    devs = sorted({t[0] for t in events})
    collectives = sorted({t[1] for t in events})
    element_counts = sorted({t[2] for t in events})
    # Get the set of (collective, element_count) pairs we measured; sort so all processes agree on the indices
    collective_sizes = sorted(set(t[1:-1] for t in events))
    # Construct an array of local-process results that we can gather across processes later
    collective_timings = np.zeros(
        (jax.local_device_count(), len(collective_sizes)), dtype=float
    )
    for element_count in element_counts:
        element_count_times = {}
        for collective in collectives:
            # Print out process 0's results as they come in
            try:
                index = collective_sizes.index((collective, element_count))
            except ValueError:
                continue

            def _elapsed(dev):
                start = events[(dev, collective, element_count, "start")]
                end = events[(dev, collective, element_count, "end")]
                checkCudaErrors(cuEventSynchronize(end))
                return checkCudaErrors(cuEventElapsedTime(start, end))

            times = [_elapsed(dev) for dev in devs]
            element_count_times[collective] = times
            collective_timings[:, index] = times
        if callback is not None:
            callback(element_count, element_count_times)
    return collective_sizes, collective_timings


def _get_total_memory(dev: jax.Device) -> int:
    checkCudaErrors(cuDevicePrimaryCtxRetain(dev.local_hardware_id))
    _, total_mem = checkCudaErrors(cuMemGetInfo())
    return total_mem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure-JAX implementation of a NCCL performance test"
    )
    parser.add_argument(
        "--coordinator-address",
        help="Distributed coordinator address:port; used if --distributed is passed.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Run jax.distributed.initialize()",
    )
    parser.add_argument(
        "--gpus-per-process",
        help=(
            "Number of GPUs driven by each controller process. "
            "Defaults to 1 with --distributed and all of them otherwise."
        ),
        type=int,
    )
    parser.add_argument(
        "--process-count",
        help=(
            "When --distributed is passed this gives the total number of processes. "
            "This can either be an integer of the name of an environment variable."
        ),
        type=int_or_env,
    )
    parser.add_argument(
        "--process-id",
        help=(
            "When --distributed is passed this gives the global index of this process."
            "This can either be an integer or the name of an environment variable."
        ),
        type=int_or_env,
    )
    parser.add_argument(
        "--replica-count",
        help=(
            "Divide the GPUs into this many parallel groups that do not communicate "
            "with one another. For example with 8 GPUs and a replica count of 2, the "
            "emitted collectives will be for 2 groups of 4 GPUs."
        ),
        type=int_or_env,
        default=1,
    )
    parser.add_argument(
        "--collectives",
        help=(
            "Which collectives to profile. Recognised values: all_gather, "
            "all_reduce, broadcast, permute, reduce_scatter."
        ),
        action="append",
        type=str,
    )
    parser.add_argument(
        "--num-iterations",
        help="How many times to call each (collective, size) point.",
        type=int,
        # Default is warmup + 1 that nsys-jax discarded + 2 that nsys-jax keeps
        default=4,
    )
    parser.add_argument(
        "--min-size-power",
        help="The smallest message size to test, as an exponent of 2.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max-size-power",
        help="The largest message size to test, as an exponent of 2.",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--process-id-permutation-seed",
        help=(
            "Apply a permutation between the set of process IDs passed via --process-id "
            "and the set of process IDs passed to jax.distributed.initialize with this "
            "seed. --process-id=0 is always re-mapped to 0 to simplify configuring the "
            "coordinator address."
        ),
        type=int,
    )
    args = parser.parse_args()
    assert args.replica_count >= 1, args.replica_count
    assert args.num_iterations >= 1, args.num_iterations
    assert args.process_id is None or args.distributed, (
        "--process-id is only relevant with --distributed"
    )
    if args.collectives is None:
        args.collectives = ["all_gather", "all_reduce", "broadcast", "permute", "reduce_scatter"]
    if args.distributed:
        null_args = {
            args.coordinator_address is None,
            args.gpus_per_process is None,
            args.process_count is None,
            args.process_id is None,
        }
        if all(null_args):
            # Use default behaviour
            jax.distributed.initialize()
        else:
            assert not any(null_args), (
                "All of --coordinator-address, --gpus-per-process, --process-count and "
                "--process-id must be passed if any of them are."
            )
            checkCudaErrors(cuInit(0))
            visible_devices = checkCudaErrors(cuDeviceGetCount())
            local_processes, rem = divmod(visible_devices, args.gpus_per_process)
            assert rem == 0, (
                f"--gpus-per-process={args.gpus_per_process} does not divide the "
                "visible device count {visible_devices}"
            )
            # assume processes within a node are globally numbered contiguously
            local_process_id = args.process_id % local_processes
            first_local_device = local_process_id * args.gpus_per_process
            local_device_ids = list(
                range(first_local_device, first_local_device + args.gpus_per_process)
            )
            process_id = args.process_id
            if args.process_id_permutation_seed is not None:
                # Test whether JAX/XLA can successfully make sense of being launched with a
                # strange ordering of nominal process IDs.
                process_ids = list(range(1, args.process_count))
                random.seed(args.process_id_permutation_seed)
                random.shuffle(process_ids)
                # Make sure that --process-id=0 is unchanged so it's possible to reason about
                # the coordinator address.
                process_ids = [0] + process_ids
                process_id = process_ids[process_id]
            print(
                f"Rank {args.process_id} has local rank {local_process_id} and "
                f"devices {local_device_ids} from a total of {visible_devices} "
                f"visible on this node, {args.process_count} processes and "
                f"{args.process_count * args.gpus_per_process} total devices. "
                f"JAX is being told it is rank {process_id}.",
                flush=True,
            )
            jax.distributed.initialize(
                coordinator_address=args.coordinator_address,
                local_device_ids=local_device_ids,
                num_processes=args.process_count,
                process_id=process_id,
            )
    elif args.gpus_per_process is not None:
        # Respect --gpus-per-process even without --distributed
        jax.config.update(
            "jax_cuda_visible_devices",
            ",".join(str(x) for x in range(args.gpus_per_process)),
        )

    jax.config.update("jax_compiler_enable_remat_pass", False)

    # log2 because process_allgather silently returns zeroes for values that don't fit in int32
    local_total_mems_log2 = np.log2(
        np.array([_get_total_memory(d) for d in jax.local_devices()])
    )
    global_total_mems_log2 = process_allgather(local_total_mems_log2)
    smallest_total_mem = 2 ** np.min(global_total_mems_log2)
    max_elements_per_device = (0.8 * smallest_total_mem) // jnp.dtype(
        _MEASUREMENT_DTYPE
    ).itemsize
    max_per_device_output_size = max_elements_per_device - 2**args.max_size_power
    if jax.process_index() == 0:
        print(f"JAX devices: {jax.devices()}")
        print(f"Partition IDs: {[d.slice_index for d in jax.devices()]}")
        print(
            f"Smallest global memory size: {smallest_total_mem / (1024 * 1024 * 1024):.1f}GiB"
        )
    n_devices = jax.device_count()
    assert (
        args.gpus_per_process is None
        or jax.local_device_count() == args.gpus_per_process
    ), (
        f"Got {jax.local_device_count()} local devices despite "
        f"--gpus-per-process={args.gpus_per_process}"
    )
    n_reps = args.replica_count
    assert n_devices % n_reps == 0, (
        f"Replica count {n_reps} does not divide device count {n_devices}"
    )
    n_devices_per_rep = n_devices // n_reps
    mesh = Mesh(
        create_device_mesh((n_reps, n_devices_per_rep)), axis_names=("replica", "i")
    )
    sharding = partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P("replica", "i"), P("replica", "i", None)),
        out_specs=P("replica", "i"),
        check_vma=False,  # for ppermute
    )

    events: dict[tuple, CUevent] = {}
    _local_to_global = {d.local_hardware_id: d.id for d in jax.local_devices()}

    def _record_event(ctx, _out, _in, *, key):
        cuda_ctx = checkCudaErrors(cuStreamGetCtx(ctx.stream))
        cuda_dev = checkCudaErrors(cuCtxGetDevice_v2(cuda_ctx))
        tup = (_local_to_global[int(cuda_dev)], *key)
        event = events.get(tup, None)
        if event is None:
            event = checkCudaErrors(cuEventCreate(CUevent_flags.CU_EVENT_DEFAULT))
            events[tup] = event
        checkCudaErrors(cuEventRecord(event, ctx.stream))

    def record_event(input, *, key):
        return buffer_callback(
            partial(_record_event, key=key),
            jax.ShapeDtypeStruct(input.shape, input.dtype),
            input_output_aliases={0: 0},
            command_buffer_compatible=False,  # It's convenient to create events on the fly
            has_side_effect=True,
        )(input)

    @partial(jax.jit, donate_argnames="buffer")
    @sharding
    def measure_all(sync, buffer):
        # Bring the whole mesh into sync
        sync = jax.lax.psum(sync, ("i", "replica"))
        assert sync.shape == (1, 1), sync.shape
        # Make sure the collectives have a data-dependency on this sync
        buffer = buffer * sync
        for size in range(args.min_size_power, args.max_size_power + 1):
            assert buffer.shape == (1, 1, 2**args.max_size_power), buffer.shape
            values_per_device = 2**size

            def measure(
                op, input, callable, epilogue=None, input_count=values_per_device
            ):
                if op not in args.collectives:
                    return input
                buf = record_event(
                    jax.lax.slice(input, (0, 0, 0), (1, 1, input_count)),
                    key=(op, values_per_device, "start"),
                )
                result = record_event(callable(buf), key=(op, values_per_device, "end"))
                if epilogue is not None:
                    result = epilogue(result)
                # Loop the result back into `input` to ensure there are data dependencies
                # between the different sizes of collectives.
                assert result.shape == (1, 1, values_per_device), result.shape
                return jax.lax.dynamic_update_slice(input, result, (0, 0, 0))

            # Trigger the collective we want to measure
            if values_per_device * n_devices_per_rep <= max_per_device_output_size:
                buffer = measure(
                    "all_gather",
                    buffer,
                    lambda buf: jax.lax.all_gather(buf, "i"),
                    # `result` gains a new 0th dimension of size `n_devices_per_rep`,
                    # reduce over it to match other ops
                    epilogue=lambda result: jnp.sum(jnp.sqrt(result), axis=0),
                )
            buffer = measure("all_reduce", buffer, lambda buf: jax.lax.psum(buf, "i"))
            buffer = measure(
                "broadcast", buffer, lambda buf: jax.lax.pbroadcast(buf, "i", 0)
            )
            # TODO: make this sensitive to whether the permutation does or
            # does not cross NVLink domain boundaries
            buffer = measure(
                "permute",
                buffer,
                lambda buf: jax.lax.ppermute(
                    buf,
                    perm=[
                        (i, (i + 1) % n_devices_per_rep)
                        for i in range(n_devices_per_rep)
                    ],
                    axis_name="i",
                ),
            )
            # Use the NCCL tests' convention where the element count for reduce
            # scatter is the number of output elements and the input buffer needs
            # to be n_devices_per_rep times larger than that.
            if values_per_device * n_devices_per_rep <= buffer.shape[-1]:
                buffer = measure(
                    "reduce_scatter",
                    buffer,
                    lambda buf: jax.lax.psum_scatter(
                        buf, "i", scatter_dimension=2, tiled=True
                    ),
                    input_count=values_per_device * n_devices_per_rep,
                )
        return buffer

    def device_put_local(x: jax.Array):
        return [jax.device_put(x, d) for d in jax.local_devices()]

    # This helper is used to trigger a small barrier before the main measurement, again
    # to improve measurement quality. It's always the same and is sharded with one
    # value per device.
    sync = jax.make_array_from_single_device_arrays(
        (n_reps, n_devices_per_rep),
        NamedSharding(mesh, P("replica", "i")),
        device_put_local(jnp.ones((1, 1))),
    )
    input = jax.make_array_from_single_device_arrays(
        (n_reps, n_devices_per_rep, 2**args.max_size_power),
        NamedSharding(mesh, P("replica", "i")),
        device_put_local(
            jax.random.normal(
                jax.random.key(1),
                (1, 1, 2**args.max_size_power),
                dtype=_MEASUREMENT_DTYPE,
            )
        ),
    )
    if jax.process_index() == 0:
        print(f"Data for pre-measurement synchronisation {sync.shape}")
        jax.debug.visualize_array_sharding(sync)

    # Warmup, create events
    warmup_start = time.time()
    input = measure_all(sync, input)
    print_table = jax.process_index() == 0
    if print_table:
        live_table = BandwidthTable(
            collectives=args.collectives,
            max_element_count=2**args.max_size_power,
            num_participants=n_devices_per_rep,
        )
        print(
            "WARNING: there is no warmup run before the collection of these results. "
            "They are printed to help distinguish between 'slow progress' and 'hung'. "
            "Higher quality measurements will be printed next."
        )
        live_table.print_header()
    collective_sizes, _ = stream_event_timer_data(
        events,
        lambda element_count, collective_times: (
            live_table.print_row(
                element_count=element_count, collective_times=collective_times
            )
            if print_table
            else None
        ),
    )
    if print_table:
        live_table.print_footer()
    input.block_until_ready()
    warmup_time = time.time() - warmup_start
    if jax.process_index() == 0:
        print(f"Warmup time: {warmup_time:.2f}s")

    # Real run
    cuProfilerStart()
    measure_time = time.time()
    timing_result_stack = []
    for _ in range(args.num_iterations - 1):
        input = measure_all(sync, input)
        # No callback, no live printing.
        check_collective_sizes, collective_timings = stream_event_timer_data(events)
        assert check_collective_sizes == collective_sizes
        timing_result_stack.append(collective_timings)
    input.block_until_ready()
    measure_time = time.time() - measure_time
    if jax.process_index() == 0:
        print(f"Measurement time: {measure_time:.2f}s")
    cuProfilerStop()

    # Stack up the results from repeatedly calling `measure_all`.
    collective_timings = np.concatenate(timing_result_stack)
    assert collective_timings.shape == (
        (args.num_iterations - 1) * jax.local_device_count(),
        len(collective_sizes),
    )

    # Gather timing information from all processes
    collective_timings = process_allgather(collective_timings, tiled=True)
    assert collective_timings.shape == (
        (args.num_iterations - 1) * jax.device_count(),
        len(collective_sizes),
    )

    if jax.process_index() == 0:
        element_counts = sorted(
            {element_count for _, element_count in collective_sizes}
        )
        table = BandwidthTable(
            collectives=args.collectives,
            max_element_count=max(element_counts),
            num_participants=n_devices_per_rep,
        )
        table.print_header()
        for size in element_counts:
            timings = {}
            for n, (coll, coll_size) in enumerate(collective_sizes):
                if size != coll_size:
                    continue
                assert coll not in timings
                timings[coll] = collective_timings[:, n]
            table.print_row(element_count=size, collective_times=timings)
        table.print_footer()
    if jax.process_index() == 0:
        print("Exiting...")
