from collections import defaultdict
import lzma
import numpy as np
import pandas as pd  # type: ignore
import pathlib
import re

from .analysis import calculate_collective_metrics
from .protobuf import xla_module_metadata
from .utils import make_child_mask, ProfilerData

pd.options.mode.copy_on_write = True


def _classify_comms(thunk_df: pd.DataFrame, prefix: pathlib.Path) -> pd.DataFrame:
    # Classify each thunk as either communication or computation, as we only
    # want to attribute non-overlapped communication time to those operations.
    # Use HloInstructionProto.channel_id as a proxy for whether an operation is
    # communication.
    def is_communication(row):
        assert thunk_df.index.names == [
            "ProgramId",
            "ProgramExecution",
            "ThunkIndex",
            "Device",
        ]
        program_id = row.name[0]
        if program_id == -1:
            # Assume this is an autotuning execution.
            return False
        try:
            # this will be a dict of {profile_name | None: proto} to cover the case
            # that we had different protos from different profiles
            protos = xla_module_metadata(program_id, prefix, policy="all")
            results = {
                proto.find_instruction(row["Name"])[1].channel_id != 0
                for proto in protos.values()
            }
            if len(results) != 1:
                raise Exception("Inconsistent results from different profiles")
        except:
            print(f'Failed to get metadata for {row["Name"]} in program #{program_id}')
            raise
        return next(iter(results))

    thunk_df["Communication"] = thunk_df.apply(is_communication, axis=1)
    thunk_df["ProjDurHiddenNs"] = 0.0
    # For convenience when calculating unhidden comms
    thunk_df["ProjEndNs"] = thunk_df["ProjStartNs"] + thunk_df["ProjDurNs"]

    for _, module_exec_df in thunk_df.groupby(
        ["ProgramId", "ProgramExecution", "Device"]
    ):
        # Update the projected duration of each communication kernel to only
        # include the non-overlapped time
        for comm_thunk in module_exec_df[module_exec_df["Communication"]].itertuples():
            # This is a range annotating a communication operation, i.e. NCCL kernel
            # That kernel was active from comm_thunk.ProjStartNs until comm_thunk.ProjEndNs
            # but during that time then other computation was going on. We want to
            # find how much of the time did not overlap with other computation.
            compute_df = module_exec_df[
                (
                    (module_exec_df["ProjEndNs"] > comm_thunk.ProjStartNs)
                    & (module_exec_df["ProjStartNs"] < comm_thunk.ProjEndNs)
                    & ~module_exec_df["Communication"]
                )
            ]
            # The computation kernels should all be serialised, but check that
            assert (
                compute_df["ProjStartNs"].iloc[1:].array
                >= compute_df["ProjEndNs"].iloc[:-1].array
            ).all()
            compute_time = np.sum(
                np.minimum(compute_df["ProjEndNs"], comm_thunk.ProjEndNs)
                - np.maximum(compute_df["ProjStartNs"], comm_thunk.ProjStartNs)
            )
            # Update the projected duration of communication kernels to just be the
            # time that is not hidden.
            unhidden_comm_time = comm_thunk.ProjDurNs - compute_time
            thunk_df.loc[comm_thunk.Index, "ProjDurNs"] = unhidden_comm_time
            thunk_df.loc[comm_thunk.Index, "ProjDurHiddenNs"] = compute_time

    return thunk_df.drop(columns=["ProjEndNs"])


def _load_nvtx_gpu_proj_trace_single(
    prefix: pathlib.Path,
    file: pathlib.Path,
    frames: set[str],
    process_index: int,
):
    # Load input data; rename some columns for convenience with `.itertuples()`; use RangeId as the index
    df = pd.read_parquet(file).drop(columns=["PID", "Rank"])
    # Alternative trace.parquet format
    alt_rename_map = {
        "Text": "Name",
        "Start": "ProjStartNs",
        "End": None,
        "Children Count": "NumChild",
        "Range ID": "RangeId",
        "Parent ID": "ParentId",
        "Range Stack": "RangeStack",
        "Stack Level": "Lvl",
    }
    if set(df.columns) == alt_rename_map.keys():
        df = df.rename(
            columns={k: v for k, v in alt_rename_map.items() if v is not None}
        )
        df["ProjDurNs"] = df.pop("End") - df["ProjStartNs"]
        df["RangeStack"] = df["RangeStack"].map(
            lambda stack: ":" + ":".join(map(str, stack))
        )
        # TODO: add OrigDurNs, OrigStartNs
    else:
        df = df.rename(
            columns={
                "Projected Duration": "ProjDurNs",
                "Projected Start": "ProjStartNs",
                "Orig Duration": "OrigDurNs",
                "Orig Start": "OrigStartNs",
            }
        ).dropna(subset=["RangeId"])
    df = df.set_index(df.pop("RangeId").astype(np.int32), verify_integrity=True)
    # Due to idiosyncracies of how Nsight tracks CUDA graphs, and because
    # thunks can be nested, the NVTX hierarchy generally looks like:
    #  Iteration -> XlaModule:A [-> XlaModule:B] -> Thunk:C [-> Thunk:D ...]
    # and we aim to keep the deepest XlaModule range (B) and the deepest Thunk
    # range (D), and add a link from that Thunk range to that XlaModule range.

    # Get all of the Thunks in the profile. Note that we want to discard some
    # of these, like Thunk:C in the example above, for not being the most
    # deeply nested.
    thunk_prefix = "TSL:Thunk:#"
    all_thunks = df["Name"].str.startswith(thunk_prefix)

    # If profile collection started while an XlaModule was executing, there may
    # be Thunk ranges without XlaModule parents. We treat those as edge effects
    # and ignore them.
    module_prefix = "TSL:XlaModule:"
    all_modules = df["Name"].str.startswith(module_prefix)
    first_module_orig_time = df.loc[all_modules, "OrigStartNs"].min()
    thunks_without_modules = all_thunks & (df["OrigStartNs"] < first_module_orig_time)
    if thunks_without_modules.sum():
        print(f"Ignoring {thunks_without_modules.sum()} thunks without modules")
    all_thunks &= ~thunks_without_modules

    # We will set ModuleId to refer to the RangeId of the closest ancestor
    # XlaModule of each Thunk.
    # Initial state, correct where thunks are direct descendants of modules
    df.loc[:, "ModuleId"] = -1
    df.loc[all_thunks, "ModuleId"] = df.loc[all_thunks, "ParentId"]

    # Convert to a series of indices in the global dataframe
    thunk_ids = df[all_thunks].index

    # Where the ModuleId refers to a Thunk, update ModuleId with that Thunk's
    # ParentId. Iterate until convergence (i.e. when all ModuleId refer to
    # XlaModule ranges).
    while True:
        # thunk_ids are indices (in df) of the Thunks we are checking
        mod_ids = df.loc[thunk_ids, "ModuleId"].astype(np.int32)
        # get the names of the ranges referred to by ModuleId
        mod_id_names = df.loc[mod_ids, "Name"]
        assert mod_ids.shape == mod_id_names.shape
        # Get a mask in mod_id_names of entries where ModuleId in the original
        # Thunk is not referring to a Module. If it's not a module, it should
        # be a thunk.
        mask = ~mod_id_names.str.startswith(module_prefix)
        assert (mask == mod_id_names.str.startswith(thunk_prefix)).all()
        assert mask.shape == mod_ids.shape
        # We want to end up without all_thunks containing thunks with child
        # thunks, as noted above.
        thunk_ids_with_child_thunks = mod_ids.array[mask]
        all_thunks[thunk_ids_with_child_thunks] = False
        # Set thunk_ids to be the (shorter) list of indices (in df) of the
        # Thunks whose ModuleId values need to be updated
        thunk_ids = thunk_ids[mask]
        if thunk_ids.empty:
            break
        # For those Thunks, replace ModuleId with the ParentId of the range
        # that was referred to by the old ModuleId
        df.loc[thunk_ids, "ModuleId"] = df.loc[mod_ids, "ParentId"][mask].array

    # Now all the Thunks should have ModuleId pointing to an XlaModule range.
    mod_ids = sorted(set(df.loc[all_thunks, "ModuleId"].astype(np.int32)))
    assert df.loc[all_thunks, "Name"].str.startswith(thunk_prefix).all()
    assert df.loc[mod_ids, "Name"].str.startswith(module_prefix).all()

    # Somewhat fuzzily try and drop the last occurence of each XlaModule if it
    # looks like profile collection stopped partway through it. For each
    # XlaModule calculate the mean and standard deviation of the number of GPU
    # operations in all but the last occurence, and see if the last occurence
    # is an outlier. TODO: if we processed the SQLite database directly, we
    # would know if the current XlaModule range had actually been closed.
    for mod_name, mod_name_df in df.loc[mod_ids, :].groupby("Name"):
        gpu_ops = mod_name_df["NumGPUOps"].array
        not_last, last = gpu_ops[:-1], gpu_ops[-1]
        if last < np.mean(not_last) - np.std(not_last):
            print(
                "Skipping last occurence of {} because it only had {} GPU operations, compared to {} +/- {} before".format(
                    mod_name, last, np.mean(not_last), np.std(not_last)
                )
            )
            mod_id = mod_name_df.index[-1]
            mod_ids.remove(mod_id)
            # Also remove its thunks from all_thunks
            all_thunks &= df["ModuleId"] != mod_id

    # Parse the numerical program ID out of the name of each XlaModule.
    # program_id is not set in all cases, although this could be fixed in XLA.
    # The classic example where it is not set is during autotuning, where ops
    # to be autotuned are extracted into new HloModule instances, which are not
    # propagated to the GpuExecutable that emits the XlaModule annotation.
    # Those are probably not interesting, so setting the ProgramId to -1 in
    # such cases is acceptable.
    module_re = r"^TSL:XlaModule:#(?:prefix=(.*?),|)hlo_module=([a-z0-9._-]+)(?:,program_id=(\d+)|)#$"
    mod_program_ids = (
        df.loc[mod_ids, "Name"]
        .str.replace(
            pat=module_re,
            repl=lambda m: "-1" if m.group(3) is None else m.group(3),
            n=1,
            regex=True,
        )
        .astype(np.int32)
    )
    # Update each module and thunk row with the program ID it corresponds to
    df.loc[mod_ids, "ProgramId"] = mod_program_ids
    df.loc[all_thunks, "ProgramId"] = mod_program_ids[
        df.loc[all_thunks, "ModuleId"]
    ].array

    # Add a new column describing which (0th, 1st, ...) execution of the module
    # each module/thunk range corresponds to.
    mod_exec_indices = df.loc[mod_ids, :].groupby(["TID", "ProgramId"]).cumcount()
    df.loc[mod_ids, "ProgramExecution"] = mod_exec_indices
    df.loc[all_thunks, "ProgramExecution"] = mod_exec_indices[
        df.loc[all_thunks, "ModuleId"]
    ].array

    # Assume that the thunks are launched from one thread per device, this is probably
    # safe. Also, until https://github.com/openxla/xla/pull/14092 is plumbed through,
    # assume that thread ID order is local device order (FIXME!)
    tid_to_ordinal: dict[int, int] = {}
    for _, module_df in df[all_thunks].groupby("ProgramId"):
        # A given module should have N threads submitting work to N devices, but the
        # thread ID submitting work to device 0 is different for N=1 (main thread) and
        # N>1 (a worker thread)
        for ordinal, tid in enumerate(sorted(module_df["TID"].unique())):
            assert tid_to_ordinal.get(tid, ordinal) == ordinal
            tid_to_ordinal[tid] = ordinal
    # This profile contains devices [process_index*num_devices, (process_index+1)*num_devices]
    num_devices = len(set(tid_to_ordinal.values()))
    df["Device"] = df["TID"].map(
        {k: process_index * num_devices + v for k, v in tid_to_ordinal.items()}
    )

    def clean_data_frame(d):
        return d.drop(
            columns=[
                "Lvl",
                "ModuleId",
                "NumChild",
                "NumGPUOps",
                "ParentId",
                "RangeStack",
                "Style",
                "TID",
            ]
        ).astype({"ProgramExecution": np.int32, "ProgramId": np.int32})

    output = {
        "#devices": num_devices,
    }
    if "thunk" in frames:
        # At this point there should be no need to look beyond the rows for individual thunks + the protobuf data, and we can further clean up the data
        thunk_df = clean_data_frame(df[all_thunks])
        thunk_df["Name"] = thunk_df["Name"].replace(
            to_replace="^TSL:Thunk:#(?:name=(.*?),|)hlo_op=([a-z0-9._-]+)#$",
            value=r"\2",
            regex=True,
        )
        # Add an index of the thunk within the module
        thunk_df["ThunkIndex"] = thunk_df.groupby(
            ["ProgramId", "ProgramExecution", "Device"]
        ).cumcount()
        # Add a new column describing which (0th, 1st, ...) execution of the thunk
        # within the given module execution this is. For example, while loops in the
        # HLO can lead to the same thunk being executed multiple times within the same
        # module execution.
        thunk_df["ThunkExecution"] = thunk_df.groupby(
            ["ProgramId", "ProgramExecution", "Device", "Name"]
        ).cumcount()

        # Classify thunks as communication/computation and save to output
        # TODO: instead of using this sorting here, use a sort bucketed by execution time to speed up overlap detection?
        output["thunk"] = _classify_comms(
            thunk_df.set_index(
                ["ProgramId", "ProgramExecution", "ThunkIndex", "Device"]
            ).sort_index(),
            prefix,
        )

    if "module" in frames:
        # Also create a reduced, module-centric dataframe
        module_df = clean_data_frame(df.loc[mod_ids, :])
        module_df["Name"] = module_df["Name"].replace(
            to_replace=module_re, value=r"\2", regex=True
        )
        module_df["NumThunks"] = module_df.index.to_frame().apply(
            lambda row: sum(df.loc[all_thunks, "ModuleId"] == row["RangeId"]), axis=1
        )
        output["module"] = module_df.set_index(
            ["ProgramId", "ProgramExecution", "Device"]
        )

    return output


def _load_nvtx_gpu_proj_trace(
    prefix: pathlib.Path,
    frames: set[str],
):
    path = prefix / "nvtx_gpu_proj_trace" / "trace.parquet"
    if path.is_dir():
        # We're looking at the output of nsys-jax-combine
        filenames = sorted(path.iterdir())
    else:
        # We're looking at the output of nsys-jax
        filenames = [path]

    tmp = defaultdict(list)
    for n_process, file in enumerate(filenames):
        for k, v in _load_nvtx_gpu_proj_trace_single(
            prefix, file, frames, n_process
        ).items():
            tmp[k].append(v)
    # Make sure all the profiles agree about how many devices per profile there are
    device_counts = set(tmp.pop("#devices"))
    assert len(device_counts) == 1, f"Inconsistent #devices/profile: {device_counts}"
    output = {}
    for k, v in tmp.items():
        output[k] = pd.concat(v, verify_integrity=True).sort_index()
    return output


def _load_nvtx_pushpop_trace_single(name: pathlib.Path) -> pd.DataFrame:
    def keep_column(name):
        return name not in {"PID", "Lvl", "NameTree"}

    compile_df = pd.read_csv(
        lzma.open(name, "rt", newline=""),
        dtype={"RangeId": np.int32},
        index_col="RangeId",
        usecols=keep_column,
    ).rename(
        columns={
            "Start (ns)": "StartNs",
            "End (ns)": "EndNs",
            "Duration (ns)": "DurNs",
            "DurChild (ns)": "DurChildNs",
            "DurNonChild (ns)": "DurNonChildNs",
        }
    )
    # XlaCompileBackend always has the name and program ID, while one code path
    # produces XlaCompile annotations that don't have the program ID. Also, the
    # auto-tuner produces nested XlaCompileBackend ranges with different names
    # and IDs to the ancestor XlaCompile range.
    backend_prefix = "TSL:XlaCompileBackend:#module="
    backend_re = re.compile("^" + backend_prefix + r"(.*?),program_id=(\d+)#$")
    backend_mask = compile_df["Name"].str.startswith(backend_prefix)
    # Populate ProgramId and ProgramName by parsing the XlaCompileBackend name
    compile_df.loc[backend_mask, "ProgramId"] = compile_df.loc[
        backend_mask, "Name"
    ].str.replace(backend_re, r"\2", regex=True)
    compile_df.loc[backend_mask, "ProgramName"] = compile_df.loc[
        backend_mask, "Name"
    ].str.replace(backend_re, r"\1", regex=True)
    # Propagate ProgramId and ProgramName up from XlaCompileBackend to XlaCompile
    backends_with_parents = backend_mask & ~compile_df["ParentId"].isna()
    backend_parent_ids = (
        compile_df.loc[backends_with_parents, "ParentId"].astype(np.int32).array
    )
    compile_prefix = "TSL:XlaCompile:#module="
    assert (
        compile_df.loc[backend_parent_ids, "Name"].str.startswith(compile_prefix).all()
    )
    compile_df.loc[backend_parent_ids, "ProgramId"] = compile_df.loc[
        backends_with_parents, "ProgramId"
    ].array
    compile_df.loc[backend_parent_ids, "ProgramName"] = compile_df.loc[
        backends_with_parents, "ProgramName"
    ].array
    # When parallel compilation is enabled, we end up with worker threads that
    # emit NVTX ranges but which are not accounted for in the RangeStack tree.
    # Splice these in.
    compile_mask = compile_df["Name"].str.startswith(compile_prefix)
    new_parent_ranges = set()
    for compile_range in compile_df[compile_mask].itertuples():
        # Identify top-level ranges in possible worker threads associated to this XlaCompile range
        worker_mask = (
            (compile_df["StartNs"] >= compile_range.StartNs)
            & (compile_df["EndNs"] <= compile_range.EndNs)
            & (compile_df["TID"] != compile_range.TID)
            & (compile_df["ParentId"].isna())
        )
        for worker_range in compile_df[worker_mask].itertuples():
            assert worker_range.Name.startswith("TSL:Xla")
            # Find the deepest still-open range in the main thread
            mask = (
                make_child_mask(compile_df, compile_range.Index)
                & (compile_df["StartNs"] < worker_range.StartNs)
                & (compile_df["EndNs"] > worker_range.EndNs)
                & (compile_df["TID"] == compile_range.TID)
            )
            new_parent_index = mask[mask].index[-1]
            assert compile_df.loc[new_parent_index, "TID"] == compile_range.TID
            # Graft this worker/child range and its children into compile_df as
            # children of `new_parent_index`
            compile_df.loc[worker_range.Index, "ParentId"] = new_parent_index
            compile_df.loc[new_parent_index, "NumChild"] += 1
            range_stack_prefix = compile_df.loc[new_parent_index, "RangeStack"]
            # Get a mask for this worker range and all its children
            mask = make_child_mask(compile_df, worker_range.Index)
            mask.loc[worker_range.Index] = True
            # prefix RangeStack values with `range_stack_prefix`
            compile_df.loc[mask, "RangeStack"] = compile_df.loc[
                mask, "RangeStack"
            ].str.slice_replace(stop=0, repl=range_stack_prefix)
            new_parent_ranges.add(new_parent_index)
            # See TODO below. Set to NaN to avoid meaningless numbers being used.
            compile_df.loc[new_parent_index, ("DurChildNs", "DurNonChildNs")] = (
                np.nan,
                np.nan,
            )
    # TODO: update the DurChildNs and DurNonChildNs values for `new_parent_ranges` to avoid double-counting
    # Mask out anything that's not descended from XlaCompile now
    compile_related_mask = compile_mask
    for compile_range_index in compile_mask[compile_mask].index:
        compile_related_mask |= make_child_mask(compile_df, compile_range_index)
    compile_df = compile_df[compile_related_mask]
    # Fill in all the missing ProgramId and ProgramName values by navigating up
    # to the closest ancestor range that has them set.
    mask = compile_df["ProgramId"].isna()
    rows_to_fill = mask[mask].index
    ancestor_rows = compile_df.loc[rows_to_fill, "ParentId"]
    while len(rows_to_fill):
        # If any ancestor_row values are nan we can't navigate further; mask them
        good_mask = ~ancestor_rows.isna()
        ancestor_rows = ancestor_rows[good_mask]
        rows_to_fill = rows_to_fill[good_mask]
        # For each (row, ancestor) in zip(rows_to_fill, ancestor_rows) see if
        # ProgramId and ProgramName in `row` can be populated from `ancestor`.
        # If not, make ancestor one generation higher.
        ancestor_ids = compile_df.loc[ancestor_rows, "ProgramId"]
        ancestor_names = compile_df.loc[ancestor_rows, "ProgramName"]
        # If yes, copy those values into the row
        good_mask = ~ancestor_ids.isna()
        done_rows = rows_to_fill[good_mask].array
        compile_df.loc[done_rows, "ProgramId"] = ancestor_ids[good_mask].array
        compile_df.loc[done_rows, "ProgramName"] = ancestor_names[good_mask].array
        # If not, look one generation higher above the row
        ancestor_rows = compile_df.loc[ancestor_rows.array[~good_mask], "ParentId"]
        rows_to_fill = rows_to_fill[~good_mask]
    # It makes analysis confusing to have ranges from cuBLAS appear as
    # children of XLA ranges, so fold non-TSL ranges into their TSL parents
    non_xla_mask = ~compile_df["Name"].str.startswith("TSL:")
    for non_xla_range in compile_df[non_xla_mask].itertuples():
        # Do not support nested non-TSL children for now.
        assert non_xla_range.NumChild == 0
        parent_id = int(non_xla_range.ParentId)
        # Pretend `non_xla_range` doesn't exist as a child of `parent_id`
        compile_df.loc[parent_id, "NumChild"] -= 1
        compile_df.loc[parent_id, "DurChildNs"] -= non_xla_range.DurNs
        compile_df.loc[parent_id, "DurNonChildNs"] += non_xla_range.DurNs

    # Because the ProgramId and ProgramName ranges provide the same information,
    # remove those fields from the compilation range names.
    def remove_program_id_and_name(row):
        row.Name = (
            row.Name.removeprefix("TSL:")
            .replace(f",program_id={int(row.ProgramId)}", "")
            .replace(f",module={row.ProgramName}", "")
            .replace(f":#module={row.ProgramName}#", "")
        )
        return row

    return (
        compile_df[~non_xla_mask]
        .drop(columns=["EndNs"])
        .astype({"ProgramId": np.int32})
        .transform(remove_program_id_and_name, axis="columns")
    )


def _load_nvtx_pushpop_trace(prefix: pathlib.Path, frames: set[str]) -> pd.DataFrame:
    path = prefix / "report_nvtx_pushpop_trace.csv.xz"
    if path.is_dir():
        # We're looking at the output of nsys-jax-combine
        filenames = sorted(path.iterdir())
        keys = [file.name for file in filenames]
    else:
        # We're looking at the output of nsys-jax
        filenames = [path]
        keys = [prefix.name]

    return pd.concat(
        [_load_nvtx_pushpop_trace_single(file) for file in filenames],
        keys=keys,
        names=["ProfileName", "RangeId"],
    )


def load_profiler_data(
    prefix: pathlib.Path = pathlib.Path("."),
    frames: set[str] = {"compile", "communication", "thunk", "module"},
) -> ProfilerData:
    """
    Load post-processed Nsight Systems traces and prepare them for analysis.

    Arguments:
     prefix: directory under which to search for input data files (default:
        current directory.
     frames: list of prepared data frames to return.

    Return:
     Dictionary of {frame_name: data_frame} with one entry for each value of
     ``frames``.
    """
    output = ProfilerData()
    # Which prepared data frames currently come from the nvtx_pushpop_trace
    # output file.
    nvtx_pushpop_trace_frames = {"compile"}
    if len(frames & nvtx_pushpop_trace_frames):
        output.compile = _load_nvtx_pushpop_trace(
            prefix, frames & nvtx_pushpop_trace_frames
        )
    # Which prepared data frames currently come from the nvtx_gpu_proj_trace
    # output file.
    nvtx_gpu_proj_trace_frames = {"thunk", "module"}
    if len(frames & nvtx_gpu_proj_trace_frames):
        for k, v in _load_nvtx_gpu_proj_trace(
            prefix,
            frames & nvtx_gpu_proj_trace_frames,
        ).items():
            setattr(output, k, v)

    if "communication" in frames:
        assert (
            output.thunk is not None
        ), "Cannot generate 'communication' frame without 'thunk' frame"
        output.communication = calculate_collective_metrics(output.thunk)

    return output
