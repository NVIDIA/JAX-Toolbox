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
index 1e958f8..d08bb99 100644
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
@@ -172,11 +172,44 @@ def _compute_gpu_projection_df(nvtx_df, cuda_df, cuda_nvtx_index_map):
                 "tid": nvtx_df.loc[list(nvtx_gpu_end_dict.keys()) + indices, "tid"],
             }
         )
-        .sort_values(by=["start", "end"], ascending=[True, False])
+    ).reset_index()
+
+    return (
+        df.sort_values(by=["start", "end", "index"], ascending=[True, False, True])
+        .drop(columns=["index"])
         .reset_index(drop=True)
     )
 
 
+def _compute_grouped_gpu_projection_df(
+    nvtx_df, cuda_df, cuda_nvtx_index_map, per_gpu=False, per_stream=False
+):
+    group_by_elements = []
+    if per_stream:
+        group_by_elements.append("streamId")
+    if per_gpu:
+        group_by_elements.append("deviceId")
+
+    if not group_by_elements:
+        return _compute_gpu_projection_df(nvtx_df, cuda_df, cuda_nvtx_index_map)
+
+    dfs = []
+    cuda_gdf = cuda_df.groupby(group_by_elements)
+
+    for group_keys, cuda_group_df in cuda_gdf:
+        df = _compute_gpu_projection_df(nvtx_df, cuda_group_df, cuda_nvtx_index_map)
+        if df.empty:
+            continue
+
+        if per_stream:
+            df["streamId"] = group_keys[group_by_elements.index("streamId")]
+        if per_gpu:
+            df["deviceId"] = group_keys[group_by_elements.index("deviceId")]
+        dfs.append(df)
+
+    return pd.concat(dfs, ignore_index=True)
+
+
 def _find_cuda_nvtx_ranges(nvtx_df, cuda_df):
     """Find the NVTX ranges that enclose each CUDA operation.

@@ -258,8 +291,8 @@ def project_nvtx_onto_gpu(nvtx_df, cuda_df):

         cuda_tid_df = cuda_gdf.get_group(global_tid)
         cuda_nvtx_index_map = _find_cuda_nvtx_ranges(nvtx_tid_df, cuda_tid_df)
-        df = _compute_gpu_projection_df(
-            filtered_nvtx_df, cuda_tid_df, cuda_nvtx_index_map
+        df = _compute_grouped_gpu_projection_df(
+            filtered_nvtx_df, cuda_tid_df, cuda_nvtx_index_map, True, True
         )
         dfs.append(df)

diff --git a/nsys_recipe/lib/table_config.py b/nsys_recipe/lib/table_config.py
index e412c4f..db9449e 100644
--- a/nsys_recipe/lib/table_config.py
+++ b/nsys_recipe/lib/table_config.py
@@ -48,6 +48,7 @@ def get_cuda_gpu_dict():
             "deviceId",
             "contextId",
             "greenContextId",
+            "streamId",
         ],
         "CUPTI_ACTIVITY_KIND_MEMSET": [
             "correlationId",
@@ -57,6 +58,7 @@ def get_cuda_gpu_dict():
             "deviceId",
             "contextId",
             "greenContextId",
+            "streamId",
         ],
     }

diff --git a/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py b/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py
index 2f05d50..e52dabe 100644
--- a/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py
+++ b/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py
@@ -107,6 +107,8 @@ class NvtxGpuProjTrace(recipe.Recipe):
             "rangeId": "Range ID",
             "parentId": "Parent ID",
             "rangeStack": "Range Stack",
+            "deviceId": "Device ID",
+            "streamId": "Stream ID",
         }

         proj_nvtx_df = proj_nvtx_df.rename(columns=name_dict)[name_dict.values()]
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
