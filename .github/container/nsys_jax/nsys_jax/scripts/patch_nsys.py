import os
import re
import shutil
import subprocess

nsys_2024_5_and_6_patch_content = r"""diff --git a/nsys_recipe/lib/nvtx.py b/nsys_recipe/lib/nvtx.py
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

nsys_2024_7_patch_content = r'''diff --git a/nsys_recipe/lib/nvtx.py b/nsys_recipe/lib/nvtx.py
index 1e958f8..1b6138d 100644
--- a/nsys_recipe/lib/nvtx.py
+++ b/nsys_recipe/lib/nvtx.py
@@ -162,7 +162,7 @@ def _compute_gpu_projection_df(nvtx_df, cuda_df, cuda_nvtx_index_map):
         starts.append(start)
         ends.append(end)
 
-    return (
+    df = (
         pd.DataFrame(
             {
                 "text": nvtx_df.loc[list(nvtx_gpu_end_dict.keys()) + indices, "text"],
@@ -172,10 +172,13 @@ def _compute_gpu_projection_df(nvtx_df, cuda_df, cuda_nvtx_index_map):
                 "tid": nvtx_df.loc[list(nvtx_gpu_end_dict.keys()) + indices, "tid"],
             }
         )
-        .sort_values(by=["start", "end"], ascending=[True, False])
-        .reset_index(drop=True)
+        .reset_index()
     )
 
+    return df.sort_values(
+        by=["start", "end", "index"], ascending=[True, False, True]
+    ).drop(columns=["index"])
+
 
 def _find_cuda_nvtx_ranges(nvtx_df, cuda_df):
     """Find the NVTX ranges that enclose each CUDA operation.
'''


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
    match m.group(1):
        case "2024.5.1" | "2024.6.1":
            patch_content = nsys_2024_5_and_6_patch_content
        case "2024.7.1":
            patch_content = nsys_2024_7_patch_content
        case _:
            patch_content = None
    if patch_content is not None:
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
