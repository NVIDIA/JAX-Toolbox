from nsys_jax import (
    ensure_compiled_protos_are_importable,
    load_profiler_data,
    xla_module_metadata,
)
import os
import pathlib
import pytest  # type: ignore
import re
import subprocess
import sys
import tempfile

helper_dir = os.path.join(os.path.dirname(__file__), "nsys_jax_test_helpers")
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)
from nsys_jax_test_helpers import extract, multi_process_nsys_jax  # noqa: E402


@pytest.fixture(scope="module")
def individual_results():
    """
    Fixture that yields the .zip files from individual subprocesses.
    """
    num_processes = 2
    return multi_process_nsys_jax(
        num_processes=num_processes,
        command=lambda rank: [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "multi_process_program.py"),
            "--nranks",
            str(num_processes),
            "--rank",
            str(rank),
        ],
    )


@pytest.fixture(scope="module")
def individual_and_combined_results(individual_results):
    """
    Fixture that yields the extracted .zip files from individual subprocesses and the
    merged result from nsys-jax-combine.
    """
    combined_output = tempfile.NamedTemporaryFile(suffix=".zip")
    subprocess.run(
        ["nsys-jax-combine", "--force-overwrite", "--output", combined_output.name]
        + [output.name for output in individual_results],
        check=True,
    )
    combined_dir = extract(combined_output.name)
    # Make sure the protobuf bindings can be imported, the generated .py will go into
    # a temporary directory that is not currently cleaned up. The bindings cannot be
    # un-imported from the test process, so there is a tacit assumption that in a given
    # test session there will only be one set of .proto files and it doesn't matter
    # which ones are picked up.
    ensure_compiled_protos_are_importable(prefix=pathlib.Path(combined_dir.name))
    return combined_dir, [extract(output.name) for output in individual_results]


def get_module_id(dump_dir, module_name: str) -> int:
    """
    Given an extracted dump/ dir, find the numeric module ID for the named module.
    """
    path = pathlib.Path(dump_dir.name) / "dump"
    pattern = re.compile(
        rf"^module_(\d+)\.jit_{module_name}\.before_optimizations\.hlo\.pb\.xz$"
    )
    for file in path.iterdir():
        if m := pattern.match(file.name):
            return int(m.group(1))
    raise Exception(f"Did not find {module_name} in {path}: {list(path.iterdir())}")


def module_exists(dump_dir, module_name, module_id):
    path = pathlib.Path(dump_dir.name) / "dump"
    # Could be a dir or a file
    return (
        path / f"module_{module_id:04}.jit_{module_name}.before_optimizations.hlo.pb.xz"
    ).exists()


def test_metadata(individual_and_combined_results):
    """
    Make sure that the test case is producing the intended modules/IDs on disk.
    """
    combined, (rank0, rank1) = individual_and_combined_results

    # First check the individual outputs

    # The processes should be in lockstep at this point, so the IDs should match
    rank0_spmd0 = get_module_id(rank0, "distinctively_named_function_with_lhs")
    rank1_spmd0 = get_module_id(rank1, "distinctively_named_function_with_lhs")
    assert rank0_spmd0 == rank1_spmd0, (rank0_spmd0, rank1_spmd0)
    # They should still be in lockstep; this is making sure the remapping does
    # not accidentally depend on the LHS
    rank0_spmd1 = get_module_id(rank0, "distinctively_named_function_without_lhs")
    rank1_spmd1 = get_module_id(rank1, "distinctively_named_function_without_lhs")
    assert rank0_spmd1 == rank1_spmd1, (rank0_spmd1, rank1_spmd1)
    # This should only be executed in the 0th process
    rank0_only1 = get_module_id(rank0, "only_in_process_zero")
    with pytest.raises(Exception, match="Did not find only_in_process_zero"):
        get_module_id(rank1, "only_in_process_zero")
    # This should be executed in both processes, but the IDs should have diverged
    rank0_spmd2 = get_module_id(rank0, "another_distinctively_named_function")
    rank1_spmd2 = get_module_id(rank1, "another_distinctively_named_function")
    assert rank0_spmd2 > rank1_spmd2
    assert rank1_spmd2 == rank0_only1  # might turn out to be too flaky?
    # This should only be executed in the 0th process
    rank0_only2 = get_module_id(rank0, "another_one_only_in_process_zero")
    with pytest.raises(
        Exception, match="Did not find another_one_only_in_process_zero"
    ):
        get_module_id(rank1, "another_one_only_in_process_zero")
    # This should only be executed in the 1st process
    rank1_only = get_module_id(rank1, "only_in_process_one")
    with pytest.raises(Exception, match="Did not find only_in_process_one"):
        get_module_id(rank0, "only_in_process_one")

    # Second, check the combined outputs

    assert module_exists(combined, "distinctively_named_function_with_lhs", rank0_spmd0)
    assert module_exists(
        combined, "distinctively_named_function_without_lhs", rank0_spmd1
    )
    assert module_exists(combined, "only_in_process_zero", rank0_only1)
    assert module_exists(combined, "another_distinctively_named_function", rank0_spmd2)
    assert module_exists(combined, "another_distinctively_named_function", rank1_spmd2)
    assert module_exists(combined, "another_one_only_in_process_zero", rank0_only2)
    assert module_exists(combined, "only_in_process_one", rank1_only)


@pytest.fixture(scope="module")
def combined_data_and_path(individual_and_combined_results):
    combined, _ = individual_and_combined_results
    path = pathlib.Path(combined.name)
    return load_profiler_data(path), path


def get_program_id(module_data, module_name):
    test_func_mask = module_data["Name"] == f"jit_{module_name}"
    program_ids = module_data[test_func_mask].index.get_level_values("ProgramId")
    # All executions should have the same program id
    assert all(program_ids == program_ids[0])
    return program_ids[0]


instances = [
    ("distinctively_named_function_with_lhs", 5, [0, 1]),
    ("distinctively_named_function_without_lhs", 5, [0, 1]),
    ("only_in_process_zero", 1, [0]),
    ("another_distinctively_named_function", 5, [0, 1]),
    ("another_one_only_in_process_zero", 1, [0]),
    ("only_in_process_one", 1, [1]),
]


@pytest.mark.parametrize("module_name,num_executions,execution_devices", instances)
def test_combined_data(
    combined_data_and_path, module_name, num_executions, execution_devices
):
    """
    Make sure the module IDs are remapped correctly
    """
    combined_data, prefix = combined_data_and_path
    remapped_id = get_program_id(combined_data.module, module_name)
    module_data = combined_data.module.loc[remapped_id]
    assert (
        len(module_data.index.get_level_values("ProgramExecution").unique())
        == num_executions
    )
    assert (
        module_data.index.get_level_values("Device")
        == execution_devices * num_executions
    ).all()


@pytest.mark.parametrize("module_name,num_executions,execution_devices", instances)
def test_reloading_proto_data(
    combined_data_and_path, module_name, num_executions, execution_devices
):
    """
    Make sure that calling xla_module_metadata with the remapped IDs present
    in the loaded data works.
    """
    combined_data, prefix = combined_data_and_path
    remapped_id = get_program_id(combined_data.module, module_name)
    hlo_set = xla_module_metadata(program_id=remapped_id, prefix=prefix, policy="all")
    # This might prove a bit fragile. It passes today because:
    # - distinctively_named_function's HLO dump does not get de-duplicated
    #   because the HLO dumps from different SPMD processes are not bitwise
    #   identical
    # - another_distinctively_named_function additionally does not get
    #   de-duplicated because the numerical ID does not match
    assert len(hlo_set._protos) == len(execution_devices)


@pytest.mark.parametrize("recipe", ["summary", "communication"])
def test_analysis_recipes(individual_results, recipe):
    """
    Fixture that yields the extracted .zip files from individual subprocesses and the
    merged result from nsys-jax-combine.
    """
    with tempfile.NamedTemporaryFile(suffix=".zip") as combined_output:
        subprocess.run(
            [
                "nsys-jax-combine",
                "--force-overwrite",
                "--analysis",
                recipe,
                "--output",
                combined_output.name,
            ]
            + [output.name for output in individual_results],
            check=True,
        )
