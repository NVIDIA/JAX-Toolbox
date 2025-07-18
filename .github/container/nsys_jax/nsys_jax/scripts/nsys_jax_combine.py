import argparse
from collections import defaultdict
import copy
import os
import pathlib
import shutil
import tempfile
import zipfile

from .utils import execute_analysis_script, shuffle_analysis_arg


def main():
    """
    Entrypoint for nsys-jax-combine
    """
    parser = argparse.ArgumentParser(
        description=(
            "`nsys-jax-combine` facilitates distributed profiling of JAX applications "
            "using the `nsys-jax` wrapper. It aggregates multiple .zip outputs from "
            "different `nsys-jax` processes that profiled the same distributed execution "
            "of an application, checking consistency and removing duplicated data."
        ),
    )
    parser.add_argument(
        "--analysis",
        action="append",
        help=(
            "Post-processing analysis script to execute after merging. This can be the "
            "name of a recipe bundled in the inpit files, or the path to a Python script. "
            "The script will be passed any arguments specified via --analysis-arg, "
            "followed by a single positional argument, which is the path to a directory "
            "of the same structure as the extracted output archive."
        ),
        type=lambda x: ("script", x),
    )
    parser.add_argument(
        "--analysis-arg",
        action="append",
        dest="analysis",
        help="Extra arguments to pass to analysis scripts specified via --analysis",
        type=lambda x: ("arg", x),
    )
    parser.add_argument(
        "-f",
        "--force-overwrite",
        action="store_true",
        help="Overwrite the output file if it exists.",
    )
    parser.add_argument(
        "input",
        type=pathlib.Path,
        nargs="+",
        help="Input .zip archives produced by `nsys-jax`",
    )

    def check_keep_nsys_rep(raw):
        assert raw in {"all", "first", "none"}
        return raw

    parser.add_argument(
        "--keep-nsys-rep",
        default="first",
        type=check_keep_nsys_rep,
        help=(
            "How many .nsys-rep files from the input to copy to the output. Supported "
            "values are 'all', 'first' and 'none'."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file name",
        required=True,
        type=pathlib.Path,
    )
    # TODO: derive a default output path from the input paths
    args = parser.parse_args()
    args.analysis = shuffle_analysis_arg(args.analysis)
    if args.output.suffix != ".zip":
        args.output = args.output.with_suffix(".zip")
    if os.path.exists(args.output) and not args.force_overwrite:
        raise Exception(
            f"Output path {args.output} already exists and -f/--force-overwrite was not passed"
        )

    hashes = defaultdict(lambda: defaultdict(set))
    for input in args.input:
        with zipfile.ZipFile(input) as ifile:
            for member in ifile.infolist():
                hashes[member.filename][member.CRC].add(input)

    mirror_dir = pathlib.Path(tempfile.mkdtemp()) if len(args.analysis) else None
    with zipfile.ZipFile(args.output, "w") as ofile:
        for n_input, input in enumerate(args.input):
            keep_this_nsys_rep = args.keep_nsys_rep == "all" or (
                args.keep_nsys_rep == "first" and (n_input == 0)
            )
            with zipfile.ZipFile(input) as ifile:
                for member in ifile.infolist():
                    if member.is_dir():
                        continue
                    filename = member.filename
                    assert filename in hashes
                    seen_hashes = hashes[filename]

                    def write(dst_info):
                        assert dst_info.filename not in set(ofile.namelist())
                        with ifile.open(member) as src:
                            with ofile.open(dst_info, "w") as dst:
                                shutil.copyfileobj(src, dst)
                            if mirror_dir is not None:
                                dst_path = mirror_dir / dst_info.filename
                                os.makedirs(dst_path.parent, exist_ok=True)
                                src.seek(0)
                                with open(dst_path, "wb") as dst:
                                    shutil.copyfileobj(src, dst)

                    if filename.endswith(".nsys-rep"):
                        assert len(seen_hashes) == 1
                        if filename == input.stem + ".nsys-rep" and keep_this_nsys_rep:
                            # `filename`` is the .nsys-rep from `input``
                            write(member)
                    else:
                        if len(seen_hashes) == 1:
                            # This file the same wherever it appeared, which could mean
                            # that it was the same in all inputs, or could mean that it
                            # only appeared in one input.
                            inputs_containing_this = next(iter(seen_hashes.values()))
                            if input == next(iter(sorted(inputs_containing_this))):
                                write(member)
                        else:
                            # This file was not the same in all inputs: copy it to a
                            # modified destination. An input file A/B in reportN.zip will
                            # be saved as A/B/reportN in the output, i.e. A/B will be a
                            # directory instead of a file. TODO: in future instead of using
                            # input.stem use a standardised format showing the device
                            # numbers that were profiled in reportN.zip.
                            dst_info = copy.copy(member)
                            dst_info.filename = filename + "/" + input.stem
                            write(dst_info)
        if len(args.analysis):
            assert mirror_dir is not None
            # Execute post-processing recipes and add any outputs to `ofile`
            for analysis in args.analysis:
                result, output_prefix = execute_analysis_script(
                    data=mirror_dir, script=analysis[0], args=analysis[1:]
                )
                result.check_returncode()
                # Gather output files of the scrpt
                for path in (mirror_dir / output_prefix).rglob("*"):
                    src_path = mirror_dir / output_prefix / path
                    if src_path.is_dir():
                        continue
                    with (
                        open(src_path, "rb") as src,
                        ofile.open(str(path.relative_to(mirror_dir)), "w") as dst,
                    ):
                        # https://github.com/python/mypy/issues/15031 ?
                        shutil.copyfileobj(src, dst)  # type: ignore
