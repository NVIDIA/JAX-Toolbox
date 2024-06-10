import functools
import glob
import lzma
import pathlib
import typing


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
                self._instructions[inst.name] = (comp, inst)
                self._instructions_by_id[inst.id] = (comp, inst)

    def find_computation(self, id: int):
        return self._computations[id]

    def find_instruction(self, name: str):
        """
        Look up an HLO instruction and its associated computation by name in
        the wrapped HloModule.

        Returns: (computation, instruction) tuple
        """
        return self._instructions[name]

    def find_instruction_by_id(self, id: int):
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


@functools.lru_cache
def xla_module_metadata(program_id: int, prefix: pathlib.Path = pathlib.Path(".")):
    # First, find the input file. There is a lot more that can be done here,
    # but the lowest-hanging fruit is to look for a filename like:
    #   module_0016.jit_train_step.sm_8.0_gpu_after_optimizations.hlo.pb
    # where 16 is the program id
    dump_dir = prefix / "dump"
    for candidate in glob.glob(
        "*_gpu_after_optimizations.hlo.pb.xz", root_dir=dump_dir
    ):
        if program_id == int(
            candidate.split(".", maxsplit=1)[0].split("_", maxsplit=1)[1]
        ):
            break
    else:
        raise Exception(
            f"Could not find protobuf input for XlaModule {program_id} in {dump_dir}"
        )
    from xla.service import hlo_pb2

    hlo = hlo_pb2.HloProto()
    with lzma.LZMAFile(dump_dir / candidate, "rb") as f:
        hlo.ParseFromString(f.read())
    assert hlo.hlo_module.id == program_id
    return HloProto(hlo)
