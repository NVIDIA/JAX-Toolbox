import functools
import lzma
import pathlib
import typing


def _host_memory_space(inst):
    return inst.shape.layout.memory_space == 5


class StackFrame(typing.NamedTuple):
    column: int
    file: str
    function: str
    line: int

    def __str__(self):
        colstr = f":{self.column}" if self.column else ""
        s = f"{self.file}:{self.line}{colstr}[{self.function}]"
        assert ";" not in s
        return s


class HloInstruction:
    def __init__(self, wrapped_hlo_proto, proto):
        self._proto = proto
        # If this instruction represents the launch of a collective operation, find the
        # proto representing the actual collective, which will be different if the
        # async launch is handled by an async-start op
        # TODO: can any of copy-start, custom-call, recv, send represent communication?
        # This also aims to identify, and (for now) flag as communication, kernels that
        # implement device-to-host and host-to-device copies for memory offloading.
        # For example, a device-to-host offload might look like
        #   computation {
        #     ...
        #     ROOT r1 = bf16[2,8,128,2048]{3,2,1,0:S(5)} dynamic-update-slice(...)
        #   }
        #   async_computation {
        #     ...
        #     ROOT r2 = bf16[2,8,128,2048]{3,2,1,0:S(5)} fusion(...), calls=computation
        #   }
        #   start = (...) async-start(...), calls=async_computation
        # where the :S(5) annotation shows that a buffer is in host memory.
        # A host-to-device load might look like
        #   computation {
        #     param_0 = bf16[2,8,128,2048]{3,2,1,0:S(5)} parameter(0)
        #     ...
        #     ROOT r1 = bf16[2,8,128,2048]{3,2,1,0} dynamic-slice(param_0, ...)
        #   }
        #   async_computation {
        #     param_0 = bf16[2,8,128,2048]{3,2,1,0:S(5)} parameter(0)
        #     ...
        #     ROOT r2 = bf16[2,8,128,2048]{3,2,1,0} fusion(param_0, ...), calls=computation
        #   }
        #   start = (...) async-start(...), calls=async_computation
        # where the :S(5) memory space annotation is in a parameter instead of in the
        # return value.
        # For now, handling host-device kernels as single-device "collective"
        # communication should be sufficient.
        self._comm_proto = None
        comm_opcodes = {
            "all-gather",
            "all-reduce",
            "all-to-all",
            "collective-broadcast",
            "collective-permute",
            "reduce-scatter",
        }
        comm_start_opcodes = {
            "all-gather-start",
            "all-reduce-start",
            "collective-permute-start",
        }

        def _is_offloading_instruction(inst):
            host_dest = _host_memory_space(inst)

            def _host_operand(i):
                _, op = wrapped_hlo_proto.find_instruction_by_id(inst.operand_ids[i])
                return _host_memory_space(op.proto())

            if inst.opcode == "dynamic-slice" and host_dest != _host_operand(0):
                return True
            elif (
                inst.opcode == "dynamic-update-slice"
                and host_dest == _host_operand(0)
                and host_dest != _host_operand(1)
            ):
                return True
            return False

        if self._proto.opcode in comm_opcodes | comm_start_opcodes:
            self._comm_proto = self._proto
        elif self._proto.opcode in {"async-start", "fusion"}:
            # fusion example:
            #   computation {
            #     param_0 = f32[...]{...:S(5)} parameter(0)
            #     ...
            #     ROOT dus = f32[...]{...:S(5)} dynamic-update-slice(param_0, ...)
            #   }
            #   inst = f32[256,128,128]{2,1,0:S(5)} fusion(...), calls=computation
            # This might be thinly wrapping an opcode in `comm_opcodes`
            def _visit_computation(computation_id):
                computation = wrapped_hlo_proto.find_computation(computation_id)
                for called_inst in computation.instructions:
                    for called_id in called_inst.called_computation_ids:
                        _visit_computation(called_id)
                    if called_inst.opcode in comm_opcodes or _is_offloading_instruction(
                        called_inst
                    ):
                        assert (
                            self._comm_proto is None
                        ), f"Found {called_inst.opcode} child having already found {self._comm_proto.opcode}"
                        self._comm_proto = called_inst

            for called_id in self._proto.called_computation_ids:
                _visit_computation(called_id)

    def communication_proto(self):
        return self._comm_proto

    def is_communication(self) -> bool:
        """
        Classify this instruction as representing communication or computation. This is
        a little more complicated than you might hope, because async communications are
        not handled uniformly.
        """
        return self._comm_proto is not None

    def proto(self):
        """
        Access the HloInstruction protobuf object directly.
        """
        return self._proto


class HloProto:
    def __init__(self, proto):
        self._proto = proto
        # Build lookup tables
        self._computations = {}
        self._instructions = {}
        self._instructions_by_id = {}
        for comp in self._proto.hlo_module.computations:
            assert comp.id not in self._computations
            self._computations[comp.id] = comp
            for inst in comp.instructions:
                assert inst.name not in self._instructions
                assert inst.id not in self._instructions_by_id
                wrapped_inst = HloInstruction(self, inst)
                self._instructions[inst.name] = (comp, wrapped_inst)
                self._instructions_by_id[inst.id] = (comp, wrapped_inst)

    def find_computation(self, id: int):
        return self._computations[id]

    def find_instruction(self, name: str) -> tuple[typing.Any, HloInstruction]:
        """
        Look up an HLO instruction and its associated computation by name in
        the wrapped HloModule.

        Returns: (computation, instruction) tuple
        """
        return self._instructions[name]

    def find_instruction_by_id(self, id: int) -> tuple[typing.Any, HloInstruction]:
        """
        Look up an HLO instruction and its associated computation by id in
        the wrapped HloModule.

        Returns: (computation, instruction) tuple
        """
        return self._instructions_by_id[id]

    def _get_stack_frame(self, frame_id: int) -> tuple[StackFrame, int]:
        """
        Extract a stack frame given an integer ID.
        """
        assert frame_id > 0
        index = self._proto.hlo_module.stack_frame_index
        frame = index.stack_frames[frame_id - 1]
        file_location = index.file_locations[frame.file_location_id - 1]
        return (
            StackFrame(
                column=file_location.column,
                file=index.file_names[file_location.file_name_id - 1],
                function=index.function_names[file_location.function_name_id - 1],
                line=file_location.line,
            ),
            frame.parent_frame_id,
        )

    def get_stack_frames(self, frame_id: int) -> list[StackFrame]:
        """
        Given the ID of the most-nested frame, return a full stack trace,
        sorted from least-nested to most-nested.
        """
        frames = []
        while frame_id > 0:
            frame, frame_id = self._get_stack_frame(frame_id)
            frames.append(frame)
        frames.reverse()
        return frames

    def proto(self):
        """
        Access the HloModule protobuf object directly.
        """
        return self._proto


class HloProtoSet:
    """
    Represents a set of HloProto objects for the same program_id, returned by
    xla_module_metadata with policy="all".
    """

    def __init__(self, protos: dict[typing.Optional[str], HloProto]):
        assert len(protos), f"HloProtoSet got {len(protos)} HloProtos"
        self._protos = protos

    def unique_result(self, callable):
        """
        Apply a callable to all wrapped HloProto objects, and if the same result is
        always returned then return that; otherwise raise an exception.
        """
        values = iter(self._protos.values())
        result = callable(next(values))
        for proto in values:
            new_result = callable(proto)
            if result != new_result:
                raise Exception(
                    f"Inconsistent results of {callable}: {result} and {new_result}"
                )
        return result


@typing.overload
def xla_module_metadata(
    program_id: int,
    policy: typing.Literal["consistent"],
    prefix: pathlib.Path = pathlib.Path("."),
) -> HloProto: ...


@typing.overload
def xla_module_metadata(
    program_id: int,
    policy: typing.Literal["all"],
    prefix: pathlib.Path = pathlib.Path("."),
) -> HloProtoSet: ...


@functools.cache
def xla_module_metadata(
    program_id: int,
    policy: str = "consistent",
    prefix: pathlib.Path = pathlib.Path("."),
) -> typing.Union[HloProto, HloProtoSet]:
    """
    Load the protobuf metadata for module `program_id`. If given, `prefix` is the
    search path. `policy` governs what happens if `nsys-jax-combine` found inconsistent
    protobuf files in different profiles:
      consistent (default): error if the dumps from different profiles did not match
      all: return a dict of {profile_name: protobuf} with all the values that were seen
    """
    assert policy in {"consistent", "all"}
    # First, find the input file. There is a lot more that can be done here,
    # but the lowest-hanging fruit is to look for a filename like:
    #   module_0016.jit_train_step.sm_8.0_gpu_after_optimizations.hlo.pb
    # where 16 is the program id
    dump_dir = prefix / "dump"
    for candidate in dump_dir.glob("*_gpu_after_optimizations.hlo.pb.xz"):
        if program_id == int(
            candidate.name.split(".", maxsplit=1)[0].split("_", maxsplit=1)[1]
        ):
            break
    else:
        raise Exception(
            f"Could not find protobuf input for XlaModule {program_id} in {dump_dir}"
        )

    def _load(file: pathlib.Path) -> HloProto:
        from xla.service import hlo_pb2

        hlo = hlo_pb2.HloProto()
        with lzma.LZMAFile(file, "rb") as f:
            hlo.ParseFromString(f.read())
        assert hlo.hlo_module.id == program_id
        return HloProto(hlo)

    if candidate.is_dir():
        # nsys-jax-combine found different .pb.xz files from different profiles
        if policy == "consistent":
            raise Exception(
                f"program_id={program_id}: multiple protobuf dumps were found but policy demands consistency"
            )
        assert policy == "all"
        return HloProtoSet({file.name: _load(file) for file in candidate.iterdir()})
    else:
        # nsys-jax output, or nsys-jax-combine only saw consistent values
        proto = _load(candidate)
        return HloProtoSet({None: proto}) if policy == "all" else proto
