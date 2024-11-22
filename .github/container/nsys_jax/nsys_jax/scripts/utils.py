import contextlib
import importlib.resources
import os
import pathlib
import shlex
import subprocess
import sys


def shuffle_analysis_arg(analysis):
    """
    Helper for parsing --nsys-jax-analysis[-arg] (nsys-jax) and --analysis[-arg]
    (nsys-jax-combine) command line options.
    """
    if analysis is None:
        return []
    # [Script(A), Arg(A1), Arg(A2), Script(B), Arg(B1)] becomes [[A, A1, A2], [B, B1]]
    out, current = [], []
    for t, x in analysis:
        if t == "script":
            if len(current):
                out.append(current)
            current = [x]
        else:
            assert t == "arg" and len(current)
            current.append(x)
    if len(current):
        out.append(current)
    return out


def analysis_recipe_path(script):
    """
    Return a context manager that yields the path to the analysis script named by
    `script`. This can either be the name of a bundled analysis script from the
    the installed analyses/ directory, or a filesystem path.
    """
    script_file = importlib.resources.files("nsys_jax").joinpath(
        "analyses", script + ".py"
    )
    if script_file.is_file():
        return script_file
    assert os.path.exists(
        script
    ), f"{script} does not exist and is not the name of a built-in analysis script"
    return contextlib.nullcontext(pathlib.Path(script))


def execute_analysis_recipe(
    *, data: pathlib.Path, script: str, args: list[str]
) -> tuple[subprocess.CompletedProcess, pathlib.Path]:
    """
    Run the analysis script named by `script` on the profile data in the `data`
    directory (structure the same as nsys-jax[-combine] output archives), saving any
    output files to subdirectory of `data` named `output_prefix`.
    """
    with analysis_recipe_path(script) as script_path:
        analysis_command = [sys.executable, str(script_path)] + args + [str(data)]

        # Derive a unique name slug from the analysis script name
        def with_suffix(suffix):
            return data / "analysis" / (script_path.stem + suffix)

        n, suffix = 1, ""
        while with_suffix(suffix).exists():
            suffix = f"-{n}"
            n += 1
        working_dir = with_suffix(suffix)
        working_dir.mkdir(parents=True)
        print(
            f"Running analysis script: {shlex.join(analysis_command)} in {working_dir}"
        )
        result = subprocess.run(
            analysis_command,
            cwd=working_dir,
        )
    return result, working_dir.relative_to(data)
