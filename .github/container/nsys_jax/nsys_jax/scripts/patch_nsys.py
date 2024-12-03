import os
import re
import shutil
import subprocess

patch_content = r"""diff --git a/nsys_recipe/lib/nvtx.py b/nsys_recipe/lib/nvtx.py
index 2470043..7abf892 100644
--- a/nsys_recipe/lib/nvtx.py
+++ b/nsys_recipe/lib/nvtx.py
@@ -161,6 +161,7 @@ def _compute_gpu_projection_df(
             "start": list(nvtx_gpu_start_dict.values()) + starts,
             "end": list(nvtx_gpu_end_dict.values()) + ends,
             "pid": nvtx_df.loc[list(nvtx_gpu_end_dict.keys()) + indices, "pid"],
+            "tid": nvtx_df.loc[list(nvtx_gpu_end_dict.keys()) + indices, "tid"],
         }
     )

diff --git a/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py b/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py
index cd60bf4..37e0d0d 100644
--- a/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py
+++ b/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py
@@ -96,6 +96,7 @@ class NvtxGpuProjTrace(recipe.Recipe):
             "start": "Start",
             "end": "End",
             "pid": "PID",
+            "tid": "TID",
             "stackLevel": "Stack Level",
             "childrenCount": "Children Count",
             "rangeId": "Range ID",
"""


def main():
    """
    Entrypoint for nsys-jax-patch-nsys.
    """
    nsys = shutil.which("nsys")
    assert nsys is not None, "nsys-jax-patch-nsys expects nsys to be installed"
    nsys_version = subprocess.check_output([nsys, "--version"], text=True)
    m = re.match(
        r"^NVIDIA Nsight Systems version (\d+\.\d+\.\d+)\.\d+-\d+v\d+$", nsys_version
    )
    assert m is not None, f"Could not parse: {nsys_version}"
    if m.group(1) in {"2024.5.1", "2024.6.1"}:
        print(f"Patching Nsight Systems version {m.group(1)}")
        # e.g. /opt/nvidia/nsight-systems-cli/2024.7.1/target-linux-x64
        tdir = os.path.dirname(os.path.realpath(nsys))
        subprocess.run(
            [shutil.which("git"), "apply"],
            cwd=os.path.join(tdir, "python", "packages"),
            input=patch_content,
            check=True,
            text=True,
        )
