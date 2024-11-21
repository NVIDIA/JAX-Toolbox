import argparse
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import contextmanager
from glob import glob, iglob
import lzma
import os
import os.path as osp
import pandas as pd  # type: ignore
import queue
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

from .utils import shuffle_analysis_arg
from ..version import __sha__ as jax_toolbox_sha_with_prefix

# Expand %q{ENV_VAR} if the variable is defined.
def expand(string: str, skip_missing=True) -> str:
    missing = set()

    def rep(x):
        if len(x.group(1)) % 2 == 0:
            return x.group(0)
        if x.group(2) not in os.environ:
            missing.add(x.group(2))
            return x.group(0)
        return x.group(1)[:-1] + os.environ[x.group(2)]

    expanded = re.sub(r"([%]+)q\{(.*?)\}", rep, string).replace("%%", "%")
    if not skip_missing and missing:
        raise Exception(f"{missing} not defined when expanding '{string}'")
    return expanded

# Use deflate compression
COMPRESS_DEFLATE = {"compress_type": zipfile.ZIP_DEFLATED}
# Do not compress (if the file is already compressed)
COMPRESS_NONE: dict[str, int] = {}

install_script_template = r"""#!/bin/bash
#
# Usage: ./install.sh [optional arguments to virtualenv]
#
# If it doesn't already exist, this creates a virtual environment named
# `nsys_jax_env` in the current directory and installs Jupyter Lab and the
# dependencies of the Analysis.ipynb notebook that is shipped alongside this
# script inside the output archives of the `nsys-jax` wrapper.
#
# The expectation is that those archives will be copied and extracted on a
# laptop or workstation, and this installation script will be run there, while
# the `nsys-jax` wrapper is executed on a remote GPU cluster.
set -ex
SCRIPT_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
VIRTUALENV="${SCRIPT_DIR}/nsys_jax_venv"
if [[ ! -d "${VIRTUALENV}" ]]; then
  # Let `virtualenv` find/choose a Python. Currently >=3.10 is supported.
  virtualenv -p 3.13 -p 3.12 -p 3.11 -p 3.10 "$@" "${VIRTUALENV}"
  "${VIRTUALENV}/bin/pip" install -U pip
  "${VIRTUALENV}/bin/pip" install 'nsys-jax[jupyter] @ git+https://github.com/NVIDIA/JAX-Toolbox.git@{jax_toolbox_commit}#subdirectory=.github/container/nsys_jax'
  "${VIRTUALENV}/bin/install-flamegraph" "${VIRTUALENV}"
  "${VIRTUALENV}/bin/install-protoc" "${VIRTUALENV}"
else
  echo "Virtual environment already exists, not installing anything..."
fi
if [ -z ${NSYS_JAX_INSTALL_SKIP_LAUNCH+x} ]; then
  # TODO: point to nsys_jax/analysis/Analysis.ipynb
  echo "Launching: cd ${SCRIPT_DIR} && ${VIRTUALENV}/bin/python -m jupyterlab Analysis.ipynb"
  cd "${SCRIPT_DIR}" && "${VIRTUALENV}/bin/python" -m jupyterlab Analysis.ipynb
else
  echo "Skipping launch of jupyterlab due to NSYS_JAX_INSTALL_SKIP_LAUNCH"
fi
"""

def create_install_script(output_queue):
    """
    Write an install.sh to the output archive that installs nsys-jax at the same
    version/commit that the current execution is using.
    """
    jax_toolbox_sha = jax_toolbox_sha_with_prefix[1:]
    install_script = install_script_template.format(jax_toolbox_commit=jax_toolbox_sha)
    output_queue.put(("install.sh", install_script.encode(), COMPRESS_DEFLATE))

def main() -> None:
    """
    Entrypoint for nsys-jax
    """
    # Wrapper-specific arguments. This also handles -h and --help.
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        usage=(
            "nsys-jax [-h] [--nsys-jax-condition EXPRESSION] [--nsys-jax-analysis A1 "
            "[--nsys-jax-analysis-arg=A1_ARG1 [--nsys-jax-analysis-arg=A1_ARG2 ...]] "
            "[--nsys-jax-analysis A2 [--nsys-jax-analysis-arg=A2_ARG1 ...]] [-o OUTPUT | "
            "--output OUTPUT] [-f | --force-overwrite] [nsys profile arguments ...] [--] "
            "executable [executable arguments ...]"
        ),
        description=(
            "`nsys-jax` is a wrapper for `nsys profile` that collects additional metadata "
            "that are specific to JAX and XLA, post-processes the profile data, and "
            "produces a compressed .zip archive containing the relevant files."
        ),
        epilog=(
            "NOTE: if the executable arguments include a literal `--` then the optional "
            "`--` shown in the usage message MUST be passed to disambiguate. This is also "
            "required when extra nsys profile arguments are passed."
        ),
    )
    parser.add_argument(
        "--nsys-jax-analysis",
        action="append",
        dest="analysis",
        help=(
            "Post-processing analysis script to execute after report collection. This can "
            "be the name of a bundled recipe, or the path to a Python script. The script "
            "will be passed any arguments specified via --nsys-jax-analysis-arg, followed "
            "by a single positional argument, which is the path to a directory of the "
            "same structure as the extracted output archive."
        ),
        type=lambda x: ("script", x),
    )
    parser.add_argument(
        "--nsys-jax-analysis-arg",
        action="append",
        dest="analysis",
        help="Extra arguments to pass to analysis scripts specified via --nsys-jax-analysis",
        type=lambda x: ("arg", x),
    )
    parser.add_argument(
        "--nsys-jax-condition",
        help=(
            "Bash expression that will be expanded to determine if this instance "
            "of nsys-jax should actually launch nsys. Example: "
            "--nsys-jax-condition='$SLURM_LOCALID == 0' to only profile the first "
            "process on each node. The expression is evaluated inside [[ ... ]]."
        ),
    )
    parser.add_argument(
        "-f",
        "--force-overwrite",
        action="store_true",
        help="This must be passed for nsys-jax to overwrite an existing output archive.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output filename, if this has an .nsys-rep or .zip suffix it will be removed "
            "to yield ROOT, and the output archive will be ROOT.zip, which will contain a "
            "ROOT.nsys-rep."
        ),
    )

    nsys_jax_flags, unknown_args = parser.parse_known_args(sys.argv)
    nsys_jax_flags.analysis = shuffle_analysis_arg(nsys_jax_flags.analysis)
    # Remove the name of the nsys-jax wrapper
    nsys_flags_and_cmd = unknown_args[1:]
    # This can have two forms:
    #   exe [exe args ...]
    #   [nsys args ...] -- exe [exe args ...]
    # where the second one must be used if `exe args` contains `--`, even if no nsys args
    # are passed.
    try:
        limit = nsys_flags_and_cmd.index("--")
        nsys_flags = nsys_flags_and_cmd[:limit]
        application = nsys_flags_and_cmd[limit + 1 :]
    except ValueError:
        # No --, everything is the application
        nsys_flags = []
        application = nsys_flags_and_cmd

    if len(application) == 0:
        parser.print_help()
        raise Exception("No application to profile")

    if shutil.which(application[0]) is None:
        parser.print_help()
        raise Exception(f"{application[0]} not found by shutil.which")

    enable_profiling = True
    if nsys_jax_flags.nsys_jax_condition is not None:
        enable_profiling = (
            subprocess.run(
                ["/bin/bash", "-c", f"[[ {nsys_jax_flags.nsys_jax_condition} ]]"],
                shell=False,
            ).returncode
            == 0
        )

    if nsys_jax_flags.output is None:
        # There was not an explicit output location; generate one. There may be
        # multiple processes racing to do this.
        archive_handle, archive_name = tempfile.mkstemp(
            dir=os.getcwd(), prefix="nsys-jax-report-", suffix=".zip"
        )
        # Re-open it based on name later, mkstemp is just a way of avoiding races
        os.close(archive_handle)
        # No -f / --force-overwrite needed in this case
        archive_name_can_be_overwritten = True
    else:
        # Explicit output location was given in `nsys_jax_flags.output`, transform that
        # into the .zip-suffixed verison of it.
        archive_name = (
            expand(nsys_jax_flags.output.removesuffix(".nsys-rep").removesuffix(".zip"))
            + ".zip"
        )
        if not osp.isabs(archive_name):
            nsys_output = osp.join(os.getcwd(), archive_name)
        archive_name_can_be_overwritten = nsys_jax_flags.force_overwrite

    # We will write /final/output/path/name.zip, and it will contain name.nsys-rep,
    # but we do not instruct nsys to write that to /final/output/path/name.nsys-rep
    # so that more of the processing can happen on a faster, more local filesystem.
    report_name = osp.basename(archive_name).removesuffix(".zip") + ".nsys-rep"
    tmp_dir = tempfile.mkdtemp()
    tmp_rep = osp.join(tmp_dir, report_name)
    nsys_flags += ["--output", tmp_rep]

    # If --nsys-jax-analysis is used, we also construct a local directory mirroring
    # the extracted archive structure. TODO: clean this up
    mirror_dir = None if len(nsys_jax_flags.analysis) == 0 else tempfile.mkdtemp()


    def override_nsys_default(arg, value):
        if any(x.startswith(f"--{arg}=") for x in nsys_flags):
            return
        nsys_flags.append(f"--{arg}={value}")


    # Override some Nsight Systems defaults, but don't block setting them explicitly.
    override_nsys_default("cuda-graph-trace", "node")
    override_nsys_default("cpuctxsw", "none")
    override_nsys_default("python-sampling", "true")
    # TODO: consider dropping osrt from here
    override_nsys_default("trace", "cublas,cuda,cudnn,cusolver,nvtx,osrt")

    # Modified environment in which to run the application
    env = os.environ.copy()

    # Stop stack traces from being truncated in the metadata passed to XLA unless
    # the option was explicitly set.
    if "JAX_TRACEBACK_IN_LOCATIONS_LIMIT" not in env:
        env["JAX_TRACEBACK_IN_LOCATIONS_LIMIT"] = "-1"

    # Disable the compilation cache so that we get the full set of .pb files
    if "JAX_ENABLE_COMPILATION_CACHE" not in env:
        env["JAX_ENABLE_COMPILATION_CACHE"] = "false"

    # Get the existing XLA_FLAGS and parse them into a dictionary.
    xla_flag_list = shlex.split(env.get("XLA_FLAGS", ""))
    xla_flags = {}
    for flag in xla_flag_list:
        assert flag.startswith("--")
        bits = flag[2:].split("=", maxsplit=1)
        name, value = bits[0], bits[1] if len(bits) > 1 else None
        assert name not in xla_flags
        xla_flags[name] = value


    def as_list(flags):
        return [f"--{n}" if v is None else f"--{n}={v}" for n, v in flags.items()]


    assert xla_flag_list == as_list(xla_flags)


    def as_bool(s):
        """String -> bool conversion following XLA's semantics."""
        if s.lower() == "true" or s == "1":
            return True
        if s.lower() == "false" or s == "0":
            return False
        raise Exception("Could not convert '{}' to bool".format(s))


    # Enable dumping protobufs unless it was explicitly disabled
    if "xla_dump_hlo_as_proto" not in xla_flags:
        xla_flags["xla_dump_hlo_as_proto"] = "true"

    proto_dump_enabled = as_bool(xla_flags["xla_dump_hlo_as_proto"])

    # For simplicity, impose our directory structure on the dump from XLA
    if proto_dump_enabled:
        if "xla_dump_to" in xla_flags:
            print(f"WARNING: --xla_dump_to={xla_flags['xla_dump_to']} being overriden")
        xla_flags["xla_dump_to"] = osp.join(tmp_dir, "dump")
    else:
        print("WARNING: protobuf dump explicitly disabled, things will break")

    # Serialise the modified XLA flags. shlex.join is tempting, but doesn't seem to
    # get the right result for --xla_dump_hlo_pass_re=.*, as it adds extra quotes.
    env["XLA_FLAGS"] = " ".join(as_list(xla_flags))

    # Run the application in nsys
    # TODO: consider being more fault-tolerant?
    # The Nsight Systems command prefix
    nsys = [
        "nsys",
        "profile",
    ] + nsys_flags
    subprocess.run((nsys if enable_profiling else []) + application, check=True, env=env)

    # If we skipped profiling the application, there is nothing more to be done.
    if not enable_profiling:
        sys.exit(0)

    # Check the output report was written and is new
    if not osp.exists(tmp_rep):
        raise Exception(f"Could not find output file: {tmp_rep}")


    def copy_proto_files_to_tmp(tmp_dir, xla_dir=os.environ.get("SRC_PATH_XLA", "/opt/xla")):
        """
        Copy .proto files from XLA into a temporary directory under `tmp_dir`.

        TODO: install .proto files as part of `jaxlib`, so this can work without
            the XLA sources being available under `xla_dir` e.g. as part of a
            generic `pip` installation of JAX.

        Returns: (name of temporary directory, list of relative .proto paths)
        """
        start = time.time()
        proto_dir = osp.join(tmp_dir, "protos")
        tsl_dir = osp.join(xla_dir, "third_party", "tsl")
        proto_files = []
        for p, root in [("tsl/**/*.proto", tsl_dir), ("xla/**/*.proto", xla_dir)]:
            for proto in iglob(p, recursive=True, root_dir=root):
                proto_files.append(proto)
                dst_dir = osp.join(proto_dir, osp.dirname(proto))
                if not osp.isdir(dst_dir):
                    os.makedirs(dst_dir)
                shutil.copy(osp.join(root, proto), osp.join(proto_dir, proto))
        print(f"{archive_name}: gathered .proto files in {time.time()-start:.2f}s")
        return proto_dir, proto_files


    def run_nsys_recipe(recipe, report_file, tmp_dir, output_queue):
        """
        Post-process a .nsys-rep file into a .parquet file for offline analysis.
        This is currently implemented using the given nsys recipe.
        """
        start = time.time()
        recipe_output = osp.join(tmp_dir, recipe)
        subprocess.run(
            [
                "nsys",
                "recipe",
                recipe,
                "--input",
                report_file,
                "--output",
                recipe_output,
            ],
            check=True,
        )
        for ofile in iglob(recipe + "/**", recursive=True, root_dir=tmp_dir):
            full_path = osp.join(tmp_dir, ofile)
            # glob("/does-not-exist/**", recursive=True) == ['/does-not-exist/']
            if osp.isdir(full_path) or not osp.exists(full_path):
                continue
            output_queue.put((ofile, full_path, COMPRESS_NONE))
        print(f"{archive_name}: post-processing finished in {time.time()-start:.2f}s")


    def compress_and_archive(prefix, file, output_queue):
        """
        Read prefix+file, compress it, queue the compressed bytes for archival
        without further compression.
        """
        with open(osp.join(prefix, file), "rb") as ifile:
            output_queue.put((file + ".xz", lzma.compress(ifile.read()), COMPRESS_NONE))


    def run_nsys_stats_report(report, report_file, tmp_dir, output_queue):
        """
        Run a stats recipe on an .nsys-rep file (that has probably already been
        exported to .sqlite).
        """
        start = time.time()
        subprocess.run(
            [
                "nsys",
                "stats",
                "--report",
                report,
                "--input",
                report_file,
                # avoid race conditions with other reports/etc.
                "--sqlite",
                osp.splitext(report_file)[0] + "-" + report + ".sqlite",
                "--output",
                osp.join(tmp_dir, "report"),
            ],
            check=True,
        )
        for ofile in iglob("report_" + report + ".csv", root_dir=tmp_dir):
            compress_and_archive(tmp_dir, ofile, output_queue)
        print(f"{archive_name}: post-processing finished in {time.time()-start:.2f}s")


    def save_device_stream_thread_names(tmp_dir, report, output_queue):
        """
        Extract extra information from the SQLite dump that is needed to map projected NVTX
        ranges to global device IDs.
        """
        start = time.time()
        assert report.endswith(".nsys-rep"), f"{report} had an unexpected suffix"
        db_file = report.removesuffix(".nsys-rep") + "-metadata.sqlite"
        subprocess.run(
            [
                "nsys",
                "export",
                "--type",
                "sqlite",
                "--tables",
                "StringIds,TARGET_INFO_GPU,TARGET_INFO_NVTX_CUDA_DEVICE,TARGET_INFO_SYSTEM_ENV,ThreadNames",
                "--output",
                db_file,
                report,
            ],
            check=True,
        )
        assert os.path.exists(db_file)
        con = sqlite3.connect(db_file)
        cur = con.cursor()

        def table_to_parquet(query, index, filename, columns=None, index_name=None):
            res = cur.execute(query)
            if columns is None:
                columns = [x[0] for x in res.description]
            df = pd.DataFrame(res, columns=columns).set_index(index, verify_integrity=True)
            if index_name is not None:
                df.index.name = index_name
            df.to_parquet(osp.join(tmp_dir, filename))
            output_queue.put((filename, osp.join(tmp_dir, filename), COMPRESS_NONE))

        # Extract {(pid, tid): (name, priority)} map; PID/TID arithmetic comes from
        # https://docs.nvidia.com/nsight-systems/UserGuide/index.html#common-sqlite-examples
        table_to_parquet(
            r"""
            SELECT
                StringIds.value AS Name,
                ThreadNames.priority AS Priority,
                ThreadNames.globalTid / 0x1000000 % 0x1000000 AS PID,
                ThreadNames.globalTid % 0x1000000 AS TID
            FROM ThreadNames
            INNER JOIN StringIds ON ThreadNames.nameId=StringIds.id""",
            ["PID", "TID"],
            "thread-metadata.parquet",
        )
        # Extract high level metadata about the profiling session, including the hostname
        table_to_parquet(
            "SELECT name, nameEnum, value FROM TARGET_INFO_SYSTEM_ENV",
            "nameEnum",
            "system-metadata.parquet",
        )

        def table_exists(table_name):
            return (
                cur.execute(
                    f"SELECT 1 FROM sqlite_master WHERE type='table' AND name='{table_name}'"
                ).fetchall()
                != []
            )

        # Cannot write device-metadata.parquet if no device activity was profiled.
        if table_exists("TARGET_INFO_GPU") and table_exists("TARGET_INFO_NVTX_CUDA_DEVICE"):
            # Extract {device_id: metadata_and_name} map, making sure to pick up the name that
            # XLA assigns via NVTX
            def table_columns(table_name):
                return [
                    (table_name, x[0])
                    for x in cur.execute(f"SELECT * FROM {table_name} LIMIT 1").description
                ]

            table_to_parquet(
                """
                SELECT * FROM TARGET_INFO_GPU
                INNER JOIN TARGET_INFO_NVTX_CUDA_DEVICE ON TARGET_INFO_GPU.cuDevice = TARGET_INFO_NVTX_CUDA_DEVICE.deviceId""",
                ("TARGET_INFO_GPU", "cuDevice"),
                "device-metadata.parquet",
                columns=pd.MultiIndex.from_tuples(
                    table_columns("TARGET_INFO_GPU")
                    + table_columns("TARGET_INFO_NVTX_CUDA_DEVICE")
                ),
                index_name="cuDevice",
            )
        else:
            print("WARNING: NOT writing device metadata, no device activity profiled?")
        print(f"{archive_name}: extracted device/thread names in {time.time()-start:.2f}s")


    def find_pb_files_in_tmp(tmp_dir):
        """
        Return a prefix + list of relative paths to Protobuf files dumped by XLA.
        """
        return tmp_dir, glob("dump/*.pb", root_dir=tmp_dir) + glob(
            "dump/*.pbtxt", root_dir=tmp_dir
        )


    def gather_source_files(
        proto_dir, proto_files, pb_file_prefix, pb_file_list, output_queue
    ):
        """
        Given a directory containing the required .proto files (`proto_dir`) and a
        prefix (`pb_file_prefix`) and list of relative paths to .pb files
        (`pb_file_list`), extract a list of source code files referred to by the
        XLA metadata and embed those source code files in the output archive.
        """
        start = time.time()
        # .hlo.pb are used to gather source code to be embedded
        hlo_pb_files = [
            osp.join(pb_file_prefix, x) for x in pb_file_list if x.endswith(".hlo.pb")
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Compile the .proto files
            subprocess.run(
                ["protoc", f"-I={proto_dir}", f"--python_out={tmp_dir}"] + proto_files,
                check=True,
                cwd=proto_dir,
            )
            # Collect the set of referenced source files
            sys.path.insert(0, tmp_dir)
            from xla.service import hlo_pb2

            hlo = hlo_pb2.HloProto()
            src_files = set()
            for hlo_pb_file in hlo_pb_files:
                with open(hlo_pb_file, "rb") as f:
                    hlo.ParseFromString(f.read())
                src_files |= set(hlo.hlo_module.stack_frame_index.file_names)
            sys.path.remove(tmp_dir)
        if len(src_files) == 0:
            print("WARNING: no source files were gathered")
        # Copy these files into the output archive.
        for src_file in src_files:
            if src_file == "<string>":
                # This can appear due to python -c "...", for example.
                continue
            assert osp.isabs(src_file), f"{src_file} is not absolute"
            output_queue.put(("sources" + src_file, src_file, COMPRESS_DEFLATE))
        print(f"{archive_name}: gathered source code in {time.time()-start:.2f}s")


    def execute_analysis_scripts(mirror_dir, analysis_scripts):
        """
        Execute any post-processing scripts passed via --nsys-jax-analysis,
        returning a list of output files that should be added to the output
        archive.
        """
        if len(analysis_scripts) == 0:
            return [], 0

        assert mirror_dir is not None
        output = []
        exit_code = 0
        used_slugs = set()
        for analysis in analysis_scripts:
            script, args = analysis[0], analysis[1:]
            # If --nsys-jax-analysis is the name of a bundled analysis script, use that. Otherwise it should be a file that exists.
            search = [
                osp.join(
                    mirror_dir,
                    "python",
                    "nsys_jax_analysis",
                    script + ".py",
                ),
                script,
            ]
            candidates = list(filter(osp.exists, search))
            assert len(candidates), f"Could not find analysis script, tried {search}"
            args.append(mirror_dir)
            analysis_command = [sys.executable, candidates[0]] + args
            # Derive a unique name slug from the analysis script name
            slug = osp.basename(candidates[0]).removesuffix(".py")
            n, suffix = 1, ""
            while slug + suffix in used_slugs:
                suffix = f"-{n}"
                n += 1
            slug += suffix
            used_slugs.add(slug)
            working_dir = osp.join(mirror_dir, "analysis", slug)
            os.makedirs(working_dir, exist_ok=True)
            print(
                f"Running analysis script: {shlex.join(analysis_command)} in {working_dir}"
            )
            result = subprocess.run(
                analysis_command,
                cwd=working_dir,
            )
            if result.returncode != 0:
                exit_code = result.returncode
            # Gather output files of the scrpt
            for path in iglob("**", recursive=True, root_dir=working_dir):
                output.append(
                    (osp.join("analysis", slug, path), osp.join(working_dir, path))
                )
        return output, exit_code


    def write_output_file(to_process, mirror_dir, analysis_scripts):
        """
        Write the output archive (`archive_name`) by consuming entries from the
        queue until a `None` sentinel value is seen. If `mirror_dir` is not None
        then populate it with symlinks/files as necessary to create a structure
        equivalent to the output archive.
        """
        start = time.time()
        with zipfile.ZipFile(
            archive_name, "w" if archive_name_can_be_overwritten else "x"
        ) as archive:
            while True:
                timeout = 30
                try:
                    item = to_process.get(timeout=timeout)
                    to_process.task_done()
                    if item is None:
                        # This is the sentinel value instructing us to exit.
                        assert to_process.empty()
                        break
                    path_in_archive, content, kwargs = item
                    mirror_path = None
                    if mirror_dir is not None:
                        mirror_path = osp.join(mirror_dir, path_in_archive)
                        os.makedirs(osp.dirname(mirror_path), exist_ok=True)
                    if isinstance(content, bytes):
                        archive.writestr(path_in_archive, content, **kwargs)
                        if mirror_path is not None:
                            with open(mirror_path, "wb") as mfile:
                                mfile.write(content)
                    else:
                        archive.write(content, arcname=path_in_archive, **kwargs)
                        if mirror_path is not None:
                            os.symlink(content, mirror_path)
                except queue.Empty:
                    print(f"{archive_name}: output stalled ({timeout}s heartbeat)")
            # Execute analysis scripts so their outputs can be bundled in the archive
            # before it is closed
            analysis_outputs, exit_code = execute_analysis_scripts(
                mirror_dir, analysis_scripts
            )
            for path_in_archive, local_path in analysis_outputs:
                archive.write(filename=local_path, arcname=path_in_archive)
        os.chmod(archive_name, 0o644)
        print(f"{archive_name}: wrote in {time.time()-start:.2f}s")
        if exit_code != 0:
            print("Exiting due to analysis script errors")
            sys.exit(exit_code)


    def process_pb_files(pb_future):
        """
        Queue .pb and .pbtxt files for inclusion in the output archive.
        """
        pb_file_prefix, pb_file_list = pb_future.result()
        for pb_file in pb_file_list:
            futures.append(
                executor.submit(
                    compress_and_archive, pb_file_prefix, pb_file, files_to_archive
                )
            )


    def process_pb_and_proto_files(pb_future, proto_future, output_queue, futures):
        """
        Queue .proto files for inclusion in the output archive and trigger
        gathering source code files once .pb/.pbtxt/.proto files are available.
        """
        # Block for completion of copy_proto_files_to_tmp
        proto_dir, proto_files = proto_future.result()
        # Queue them for inclusion in the output archive
        for proto_file in proto_files:
            output_queue.put(
                (
                    osp.join("protos", proto_file),
                    osp.join(proto_dir, proto_file),
                    COMPRESS_DEFLATE,
                )
            )
        # Wait to have pb files too
        pb_file_prefix, pb_file_list = pb_future.result()
        # Submit work that depends on the proto directory
        futures.append(
            executor.submit(
                gather_source_files,
                proto_dir,
                proto_files,
                pb_file_prefix,
                pb_file_list,
                files_to_archive,
            )
        )


    # Orchestrate post-processing steps:
    # - collect Python source files:
    #   - collect list of .proto files
    #   - copy them to a temp dir
    #   - extract list of Python source files from .pb/.pbtxt files using that dir
    #   - save those source files to the archive
    #   - save the .proto files in the temp dir to the archive
    # - save .pb/.pbtxt files:
    #   - gather a list of these
    #   - compress them individually
    #   - add the compressed versions to the output archive w/out extra compression
    # - save the .nsys-rep file to the output archive with compression
    # - post-process the .nsys-rep
    #   - convert .nsys-rep -> .parquet in the temp dir with nsys recipe
    #   - save the .parquet file to the output archive w/out extra compression

    # Element format: (path_in_archive, Path or bytes, ZipFile.write* kwargs)
    files_to_archive: queue.Queue = queue.Queue()


    @contextmanager
    def output_thread(executor: ThreadPoolExecutor):
        """
        Launch the output worker on context manager entry, signal that it should
        exit on context manager exit.
        """
        try:
            # Spawn a worker to actually write the output file, consuming entries
            # in output_queue.
            future = executor.submit(
                write_output_file,
                files_to_archive,
                mirror_dir,
                nsys_jax_flags.analysis,
            )
            yield future
        finally:
            # Signal via the output queue that the worker should exit.
            files_to_archive.put(None)
            # Make sure any errors from the output thread are surfaced
            future.result()


    exit_code = 0
    with ThreadPoolExecutor() as executor, output_thread(executor):
        # Track futures so we can wait on them and report errors.
        futures = []
        # Queue the .nsys-rep for compression
        files_to_archive.put(
            (
                report_name,
                tmp_rep,
                COMPRESS_DEFLATE,
            )
        )
        # Convert .nsys-rep -> .parquet and queue the latter for archival
        futures.append(
            executor.submit(
                run_nsys_recipe,
                "nvtx_gpu_proj_trace",
                tmp_rep,
                tmp_dir,
                files_to_archive,
            )
        )
        # Write an installation script into the archive
        futures.append(executor.submit(create_install_script, files_to_archive))
        # Gather the list of .proto files
        proto_future = executor.submit(copy_proto_files_to_tmp, tmp_dir)
        # Gather the list of .pb[txt] files
        pb_future = executor.submit(find_pb_files_in_tmp, tmp_dir)
        futures.append(pb_future)
        futures.append(executor.submit(process_pb_files, pb_future))
        # Wait on pb_future and proto_future and submit dependent work
        futures.append(
            executor.submit(
                process_pb_and_proto_files,
                pb_future,
                proto_future,
                files_to_archive,
                futures,
            )
        )
        futures.append(
            executor.submit(
                run_nsys_stats_report,
                "nvtx_pushpop_trace",
                tmp_rep,
                tmp_dir,
                files_to_archive,
            )
        )
        # Do some custom post-processing of the .sqlite export generated by gpu_proj_future
        futures.append(
            executor.submit(
                save_device_stream_thread_names,
                tmp_dir,
                tmp_rep,
                files_to_archive,
            )
        )
        # Wait for errors/completion of `futures`; note that this does not include
        # the output thread, which is signaled to upon exiting from this block.
        # Also note that the list of futures can still grow at this point.
        retired = 0
        while True:
            results = wait(futures, return_when=FIRST_EXCEPTION, timeout=30)
            # Check if we exited early because of an exception and, if so, print it
            # immediately. Do not abort, so even in case of errors a valid archive
            # containing as much useful information as possible will be written.
            retired += len(results.done)
            for future in results.done:
                futures.remove(future)
                if future.exception() is not None:
                    exit_code = 1
                    traceback.print_exception(future.exception())
            pending = len(futures)
            if pending == 0:
                break
            print(f"{archive_name}: {pending}/{len(futures) + retired} pending")
    if exit_code:
        print(f"{archive_name}: exiting with code {exit_code} due to errors")
    sys.exit(exit_code)
