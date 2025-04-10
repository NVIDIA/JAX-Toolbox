import re
import shutil
import subprocess

patch_content_2024_5_1_and_2024_6_1 = r"""diff --git a/nsys_recipe/lib/nvtx.py b/nsys_recipe/lib/nvtx.py
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

patch_content_2024_6_2 = r'''diff --git a/nsys_recipe/lib/data_utils.py b/nsys_recipe/lib/data_utils.py
--- a/nsys_recipe/lib/data_utils.py
+++ b/nsys_recipe/lib/data_utils.py
@@ -265,6 +265,8 @@ class RangeColumnUnifier:
             cuda_df = self._original_df
 
         proj_nvtx_df = nvtx.project_nvtx_onto_gpu(filtered_nvtx_df, cuda_df)
+        if proj_nvtx_df.empty:
+            return
 
         self._filter_by_overlap(proj_nvtx_df)
 
diff --git a/nsys_recipe/lib/nvtx.py b/nsys_recipe/lib/nvtx.py
--- a/nsys_recipe/lib/nvtx.py
+++ b/nsys_recipe/lib/nvtx.py
@@ -7,11 +7,12 @@
 # disclosure or distribution of this material and related documentation
 # without an express license agreement from NVIDIA CORPORATION or
 # its affiliates is strictly prohibited.
+from collections import defaultdict
 
 import numpy as np
 import pandas as pd
 
-from nsys_recipe.lib import data_utils
+from nsys_recipe.lib import data_utils, overlap
 
 DEFAULT_DOMAIN_ID = 0
 EVENT_TYPE_NVTX_DOMAIN_CREATE = 75
@@ -66,180 +67,201 @@ def combine_text_fields(nvtx_df, str_df):
     return nvtx_textId_df.drop(columns=["textStr"])
 
 
-def compute_hierarchy_info(nvtx_df):
-    """Compute the hierarchy information of each NVTX range.
-
-    This function assumes that the input DataFrame is sorted by times. It
-    will add the following columns to the DataFrame:
-    - stackLevel: level of the range in the stack.
-    - parentId: ID of the parent range.
-    - rangeStack: IDs of the ranges that make up the stack.
-    - childrenCount: number of child ranges.
-    - rangeId: arbitrary ID of the range.
-    """
-    hierarchy_df = nvtx_df.copy()
-
-    hierarchy_df["parentId"] = None
-    hierarchy_df["stackLevel"] = 0
-    hierarchy_df["rangeStack"] = None
-
+def _compute_hierarchy_info(proj_nvtx_df, nvtx_stream_map):
+    hierarchy_df = proj_nvtx_df.assign(parentId=None, stackLevel=0, rangeStack=None)
     stack = []
 
     for row in hierarchy_df.itertuples():
-        while stack and stack[-1].end <= row.start:
+        while stack and row.end > stack[-1].end:
             stack.pop()
 
-        parent_index = stack[-1].Index if stack else np.nan
         stack.append(row)
 
-        hierarchy_df.at[row.Index, "parentId"] = parent_index
+        # Exclude ranges from the stack where the GPU operations are run on
+        # different streams by checking if the intersection of their streams is
+        # non-empty.
+        current_stack = [
+            r
+            for r in stack
+            if nvtx_stream_map[r.originalIndex] & nvtx_stream_map[row.originalIndex]
+        ]
+
+        # The current row is the last element of the stack.
+        hierarchy_df.at[row.Index, "parentId"] = (
+            current_stack[-2].Index if len(current_stack) > 1 else np.nan
+        )
         # The stack level starts at 0.
-        hierarchy_df.at[row.Index, "stackLevel"] = len(stack) - 1
-        hierarchy_df.at[row.Index, "rangeStack"] = [r.Index for r in stack]
+        hierarchy_df.at[row.Index, "stackLevel"] = len(current_stack) - 1
+        hierarchy_df.at[row.Index, "rangeStack"] = [r.Index for r in current_stack]
 
     hierarchy_df = hierarchy_df.reset_index().rename(columns={"index": "rangeId"})
-
     children_count = hierarchy_df["parentId"].value_counts()
     hierarchy_df["childrenCount"] = (
         hierarchy_df["rangeId"].map(children_count).fillna(0).astype(int)
     )
+    # Convert to Int64 to support missing values (pd.NA) while keeping
+    # integer type.
+    hierarchy_df["parentId"] = hierarchy_df["parentId"].astype("Int64")
 
     return hierarchy_df
 
 
-def _compute_gpu_projection_df(nvtx_df, cuda_df, cuda_nvtx_index_map):
+def _aggregate_cuda_ranges(
+    cuda_df, cuda_nvtx_index_map, innermost_nvtx_indices, row_offset
+):
     # Each NVTX index will be associated with the minimum start time and the
     # maximum end time of the CUDA operations that the corresponding NVTX range
     # encloses.
     nvtx_gpu_start_dict = {}
     nvtx_gpu_end_dict = {}
-    # list_of_individuals contains NVTX indices that should not be grouped.
-    # These items will be treated individually, using their original
-    # start and end times without aggregation.
-    list_of_individuals = []
+
+    indices = []
+    starts = []
+    ends = []
+
+    nvtx_stream_map = defaultdict(set)
 
     for cuda_row in cuda_df.itertuples():
         if cuda_row.Index not in cuda_nvtx_index_map:
             continue
 
         nvtx_indices = cuda_nvtx_index_map[cuda_row.Index]
+
         for nvtx_index in nvtx_indices:
-            if hasattr(cuda_row, "groupId") and not pd.isna(cuda_row.groupId):
-                list_of_individuals.append(
-                    (nvtx_index, cuda_row.gpu_start, cuda_row.gpu_end)
-                )
+            nvtx_stream_map[nvtx_index].add(cuda_row.streamId)
+
+            start = cuda_row.gpu_start
+            end = cuda_row.gpu_end
+
+            # Handle cases where the innermost NVTX range encloses CUDA events
+            # that result in multiple GPU ranges (e.g. CUDA graphs). In this
+            # case, we don't group them and keep them as separate NVTX ranges.
+            if (
+                hasattr(cuda_row, "groupId")
+                and not pd.isna(cuda_row.groupId)
+                and nvtx_index in innermost_nvtx_indices
+            ):
+                indices.append(nvtx_index)
+                starts.append(start)
+                ends.append(end)
                 continue
+
             if nvtx_index not in nvtx_gpu_start_dict:
-                nvtx_gpu_start_dict[nvtx_index] = cuda_row.gpu_start
-                nvtx_gpu_end_dict[nvtx_index] = cuda_row.gpu_end
+                nvtx_gpu_start_dict[nvtx_index] = start
+                nvtx_gpu_end_dict[nvtx_index] = end
                 continue
-            if cuda_row.gpu_start < nvtx_gpu_start_dict[nvtx_index]:
-                nvtx_gpu_start_dict[nvtx_index] = cuda_row.gpu_start
-            if cuda_row.gpu_end > nvtx_gpu_end_dict[nvtx_index]:
-                nvtx_gpu_end_dict[nvtx_index] = cuda_row.gpu_end
 
-    indices, starts, ends = [], [], []
+            if start < nvtx_gpu_start_dict[nvtx_index]:
+                nvtx_gpu_start_dict[nvtx_index] = start
+            if end > nvtx_gpu_end_dict[nvtx_index]:
+                nvtx_gpu_end_dict[nvtx_index] = end
+
+    indices += list(nvtx_gpu_start_dict.keys())
+    starts += list(nvtx_gpu_start_dict.values())
+    ends += list(nvtx_gpu_end_dict.values())
+
+    df = (
+        pd.DataFrame({"originalIndex": indices, "start": starts, "end": ends})
+        # Preserve original order for rows with identical "start" and "end"
+        # values using the index.
+        .sort_values(
+            by=["start", "end", "originalIndex"], ascending=[True, False, True]
+        ).reset_index(drop=True)
+    )
 
-    for index, start, end in list_of_individuals:
-        if index in nvtx_gpu_start_dict:
-            # The range is already included in an existing range. We skip it.
-            if start >= nvtx_gpu_start_dict[index] and end <= nvtx_gpu_end_dict[index]:
-                continue
-            # The range is partially included in an existing range. We extend
-            # the existing range to include it.
-            elif (
-                start >= nvtx_gpu_start_dict[index]
-                and start <= nvtx_gpu_end_dict[index]
-            ):
-                nvtx_gpu_end_dict[index] = end
-                continue
-            elif end >= nvtx_gpu_start_dict[index] and end <= nvtx_gpu_end_dict[index]:
-                nvtx_gpu_start_dict[index] = start
-                continue
-        # The range is not included in an existing range. We add it as a
-        # new range.
-        indices.append(index)
-        starts.append(start)
-        ends.append(end)
-
-    return (
-        pd.DataFrame(
-            {
-                "text": nvtx_df.loc[list(nvtx_gpu_end_dict.keys()) + indices, "text"],
-                "start": list(nvtx_gpu_start_dict.values()) + starts,
-                "end": list(nvtx_gpu_end_dict.values()) + ends,
-                "pid": nvtx_df.loc[list(nvtx_gpu_end_dict.keys()) + indices, "pid"],
-                "tid": nvtx_df.loc[list(nvtx_gpu_end_dict.keys()) + indices, "tid"],
-            }
+    df.index = range(row_offset, row_offset + len(df))
+    return _compute_hierarchy_info(df, nvtx_stream_map)
+
+
+def _compute_gpu_projection_df(
+    cuda_df, group_columns, cuda_nvtx_index_map, innermost_nvtx_indices, row_offset
+):
+    if group_columns:
+        cuda_gdf = cuda_df.groupby(group_columns)
+    else:
+        cuda_gdf = [(None, cuda_df)]
+
+    dfs = []
+    for group_keys, cuda_group_df in cuda_gdf:
+        df = _aggregate_cuda_ranges(
+            cuda_group_df, cuda_nvtx_index_map, innermost_nvtx_indices, row_offset
         )
-        .sort_values(by=["start", "end"], ascending=[True, False])
-        .reset_index(drop=True)
-    )
+        if df.empty:
+            continue
 
+        row_offset += len(df)
 
-def _find_cuda_nvtx_ranges(nvtx_df, cuda_df):
-    """Find the NVTX ranges that enclose each CUDA operation.
+        for key in group_columns:
+            df[key] = group_keys[group_columns.index(key)]
 
-    Returns
-    -------
-    cuda_nvtx_index_map : dict
-        Dictionary mapping each CUDA operation index to a set of NVTX range
-        indices.
-    """
-    cuda_nvtx_index_map = {}
-    nvtx_active_indices = set()
-
-    cuda_time_df = pd.DataFrame(
-        data={"start": cuda_df["start"], "end": cuda_df["end"]}
-    ).sort_values("start")
-    nvtx_start_df = pd.DataFrame(data={"time": nvtx_df["start"]}).sort_values("time")
-    nvtx_end_df = pd.DataFrame(data={"time": nvtx_df["end"]}).sort_values("time")
-
-    cuda_iter = iter(cuda_time_df.itertuples())
-    nvtx_start_iter = iter(nvtx_start_df.itertuples())
-    nvtx_end_iter = iter(nvtx_end_df.itertuples())
-
-    cuda_row = next(cuda_iter)
-    nvtx_start_row = next(nvtx_start_iter)
-    nvtx_end_row = next(nvtx_end_iter)
-
-    while True:
-        if (
-            nvtx_start_row is not None
-            and nvtx_start_row.time <= nvtx_end_row.time
-            and nvtx_start_row.time <= cuda_row.start
-        ):
-            nvtx_active_indices.add(nvtx_start_row.Index)
-
-            try:
-                nvtx_start_row = next(nvtx_start_iter)
-            except StopIteration:
-                nvtx_start_row = None
-        elif nvtx_end_row.time <= cuda_row.start or nvtx_end_row.time <= cuda_row.end:
-            nvtx_active_indices.remove(nvtx_end_row.Index)
-
-            try:
-                nvtx_end_row = next(nvtx_end_iter)
-            except StopIteration:
-                break
-        else:
-            if nvtx_active_indices:
-                cuda_nvtx_index_map[cuda_row.Index] = nvtx_active_indices.copy()
-
-            try:
-                cuda_row = next(cuda_iter)
-            except StopIteration:
-                break
-
-    return cuda_nvtx_index_map
-
-
-def project_nvtx_onto_gpu(nvtx_df, cuda_df):
+        dfs.append(df)
+
+    if not dfs:
+        return pd.DataFrame()
+
+    return pd.concat(dfs, ignore_index=True)
+
+
+def _validate_group_columns(df, group_columns):
+    if isinstance(group_columns, str):
+        group_columns = [group_columns]
+    elif group_columns is None:
+        group_columns = []
+
+    for col in group_columns:
+        if col not in df.columns:
+            raise ValueError(f"Column '{col}' not found in the DataFrame.")
+
+    return group_columns
+
+
+def _get_innermost_nvtx_indices(nvtx_df):
+    parent_nvtx_df = pd.Series(np.nan, index=nvtx_df.index)
+    stack = []
+
+    for row in nvtx_df.itertuples():
+        while stack and row.end > stack[-1].end:
+            stack.pop()
+
+        if stack:
+            parent_nvtx_df[row.Index] = stack[-1].Index
+
+        stack.append(row)
+
+    parent_ids = parent_nvtx_df.dropna().unique()
+    return set(parent_nvtx_df[~parent_nvtx_df.index.isin(parent_ids)].index)
+
+
+def project_nvtx_onto_gpu(nvtx_df, cuda_df, group_columns=None):
     """Project the NVTX ranges from the CPU onto the GPU.
 
     The projected range will have the start timestamp of the first enclosed GPU
     operation and the end timestamp of the last enclosed GPU operation.
+
+    Parameters
+    ----------
+    nvtx_df : pd.DataFrame
+        DataFrame containing NVTX ranges.
+    cuda_df : pd.DataFrame
+        DataFrame containing CUDA events. It must contain both the runtime and
+        GPU operations.
+    group_columns : str or list of str, optional
+        Column names in the CUDA table by which events should be grouped when
+        the associated NVTX range is projected onto the GPU.
+
+    Returns
+    -------
+    proj_nvtx_df : pd.DataFrame
+        DataFrame with projected NVTX ranges and additional columns for the
+        hierarchy information, including:
+        - stackLevel: level of the range in the stack.
+        - parentId: ID of the parent range.
+        - rangeStack: IDs of the ranges that make up the stack.
+        - childrenCount: number of child ranges.
+        - rangeId: arbitrary ID of the range.
     """
+    group_columns = _validate_group_columns(cuda_df, group_columns)
+
     # Filter ranges that are incomplete or end on a different thread.
     filtered_nvtx_df = nvtx_df[
         nvtx_df["start"].notnull()
@@ -251,20 +273,57 @@ def project_nvtx_onto_gpu(nvtx_df, cuda_df):
     cuda_gdf = cuda_df.groupby("globalTid")
 
     dfs = []
+    total_rows = 0
 
     for global_tid, nvtx_tid_df in nvtx_gdf:
         if global_tid not in cuda_gdf.groups:
             continue
 
         cuda_tid_df = cuda_gdf.get_group(global_tid)
-        cuda_nvtx_index_map = _find_cuda_nvtx_ranges(nvtx_tid_df, cuda_tid_df)
+        cuda_nvtx_index_map = overlap.map_overlapping_ranges(
+            nvtx_tid_df, cuda_tid_df, fully_contained=True
+        )
+
+        innermost_nvtx_indices = _get_innermost_nvtx_indices(nvtx_tid_df)
         df = _compute_gpu_projection_df(
-            filtered_nvtx_df, cuda_tid_df, cuda_nvtx_index_map
+            cuda_tid_df,
+            group_columns,
+            cuda_nvtx_index_map,
+            innermost_nvtx_indices,
+            total_rows,
         )
+        if df.empty:
+            continue
+
+        total_rows += len(df)
+
+        df["text"] = df["originalIndex"].map(nvtx_tid_df["text"])
+        # The values of pid and tid are the same within each group of globalTid.
+        for col in ["pid", "tid"]:
+            df[col] = nvtx_tid_df[col].iat[0]
+
+        df = df.drop(columns=["originalIndex"])
+
         dfs.append(df)
 
-    return pd.concat(dfs, ignore_index=True)
+    if not dfs:
+        return pd.DataFrame(
+            columns=[
+                "text",
+                "start",
+                "end",
+                "pid",
+                "tid",
+                "stackLevel",
+                "parentId",
+                "rangeStack",
+                "childrenCount",
+                "rangeId",
+            ]
+            + group_columns
+        )
 
+    return pd.concat(dfs, ignore_index=True)
 
 def classify_cuda_kernel(nccl_df, cuda_df):
     """Classify CUDA kernels.
diff --git a/nsys_recipe/lib/overlap.py b/nsys_recipe/lib/overlap.py
--- a/nsys_recipe/lib/overlap.py
+++ b/nsys_recipe/lib/overlap.py
@@ -8,6 +8,8 @@
 # without an express license agreement from NVIDIA CORPORATION or
 # its affiliates is strictly prohibited.
 
+from collections import defaultdict
+
 import numpy as np
 import pandas as pd
 
@@ -28,7 +30,7 @@ def group_overlapping_ranges(range_df):
     df = range_df.sort_values("start")
     cumulative_max_end = df["end"].cummax()
     groups = (df["start"] > cumulative_max_end.shift()).cumsum()
-    return groups
+    return groups.reindex(range_df.index)
 
 
 def consolidate_ranges(range_df):
@@ -53,110 +55,131 @@ def consolidate_ranges(range_df):
     return range_df.groupby(groups).agg({"start": "min", "end": "max"})
 
 
-def _calculate_overlap_start_end(df1, df2):
-    """Calculate the start and end positions of the overlap between two
-    dataframes.
+def process_overlapping_ranges(df1, df2, process_func, fully_contained=False):
+    """Process overlapping ranges between two dataframes.
 
     Parameters
     ----------
     df1 : dataframe
-        DataFrame containing ranges to calculate the overlap from, with 'start'
-        and 'end' columns.
+        Dataframe containing ranges with 'start' and 'end' columns. If
+        'fully_contained' is True, this is is checked to see if its ranges
+        fully contain the ranges from 'df2'.
     df2 : dataframe
-        DataFrame containing ranges to calculate the overlap with, with 'start'
-        and 'end' columns.
-
-    Returns
-    -------
-    overlap_start : np.ndarray
-        Start positions in a 2D array.
-    overlap_end : np.ndarray
-        End positions in a 2D array.
+        Dataframe containing ranges with 'start' and 'end' columns. If
+        'fully_contained' is True, this is is checked to see if its ranges
+        are fully contained within the ranges from 'df1'.
+    process_func : callable
+        Function to process overlapping ranges. It takes two arguments:
+        - df1_index: index of a row from 'df1' that overlaps with 'df2_row'.
+        - df2_row: itertuples iterator representing a row from 'df2', containing
+            the index, start, and end values.
+    fully_contained : bool
+        Whether to check if the ranges are fully contained within each other.
+        Fully contained ranges must have their start and end values within the
+        start and end values of the containing range, with the end being
+        exclusive.
     """
-    start1 = df1["start"].values
-    end1 = df1["end"].values
-
-    start2 = df2["start"].values
-    end2 = df2["end"].values
-
-    overlap_start = np.maximum(start1[:, np.newaxis], start2)
-    overlap_end = np.minimum(end1[:, np.newaxis], end2)
-
-    return overlap_start, overlap_end
-
-
-def _flatten_and_filter_invalid_ranges(overlap_start, overlap_end):
-    """Flatten and filter invalid ranges from the given arrays.
+    if df1 is None or df1.empty or df2 is None or df2.empty:
+        return
+
+    df2_time_df = pd.DataFrame(
+        data={"start": df2["start"], "end": df2["end"]}
+    ).sort_values("start")
+
+    df1_active_indices = set()
+    df1_start_df = pd.DataFrame(data={"time": df1["start"]}).sort_values("time")
+    df1_end_df = pd.DataFrame(data={"time": df1["end"]}).sort_values("time")
+
+    df2_iter = iter(df2_time_df.itertuples())
+    df1_start_iter = iter(df1_start_df.itertuples())
+    df1_end_iter = iter(df1_end_df.itertuples())
+
+    df2_row = next(df2_iter)
+    df1_start_row = next(df1_start_iter)
+    df1_end_row = next(df1_end_iter)
+
+    while True:
+        should_include_range = False
+        if df1_start_row is not None:
+            if fully_contained:
+                should_include_range = df1_start_row.time <= df2_row.start
+            else:
+                should_include_range = df1_start_row.time < df2_row.end
+
+        if should_include_range:
+            df1_active_indices.add(df1_start_row.Index)
+
+            try:
+                df1_start_row = next(df1_start_iter)
+            except StopIteration:
+                df1_start_row = None
+        elif df1_end_row.time <= df2_row.start:
+            df1_active_indices.remove(df1_end_row.Index)
+
+            try:
+                df1_end_row = next(df1_end_iter)
+            except StopIteration:
+                break
+        else:
+            for index in df1_active_indices:
+                # Check if the end of the range is contained, as only the start
+                # was checked in the first condition. If the end is not
+                # contained, skip the current 'df2' range.
+                if fully_contained and df2_row.end > df1_end_df.loc[index, "time"]:
+                    continue
+
+                process_func(index, df2_row)
+
+            try:
+                df2_row = next(df2_iter)
+            except StopIteration:
+                break
+
+
+def map_overlapping_ranges(df1, df2, key_df="df2", fully_contained=False):
+    """Map overlapping ranges between two dataframes.
 
     Parameters
     ----------
-    overlap_start : ndarray
-        Array containing the start values of the overlaps.
-    overlap_end : ndarray
-        Array containing the end values of the overlaps.
+    df1 : dataframe
+        Dataframe containing ranges with 'start' and 'end' columns. If
+        'fully_contained' is True, this is is checked to see if its ranges
+        fully contain the ranges from 'df2'.
+    df2 : dataframe
+        Dataframe containing ranges with 'start' and 'end' columns. If
+        'fully_contained' is True, this is is checked to see if its ranges
+        are fully contained within the ranges from 'df1'.
+    key_df : str
+        Whether indices of 'df1' or 'df2' should be used as the key of the
+        resulting mapping. Must be either 'df1' or 'df2'.
+    fully_contained : bool
+        Whether to check if the ranges are fully contained within each other.
+        Fully contained ranges must have their start and end values within the
+        start and end values of the containing range, with the end being
+        exclusive.
 
     Returns
     -------
-    overlap_start_1d : np.ndarray
-        Valid start positions in a 1D array.
-    overlap_end_1d : np.ndarray
-        Valid end positions in a 1D array.
-    row_indices : np.array
-        1D array of row indices matching the resulting arrays to the original
-        ones.
+    overlap_map : dict
+        Dictionary that maps indices of the 'key_df' to the indices of the
+        corresponding ranges in the other dataframe.
     """
-    overlap_start_1d = overlap_start.reshape(-1)
-    overlap_end_1d = overlap_end.reshape(-1)
+    if key_df != "df1" and key_df != "df2":
+        raise ValueError("key_df must be either 'df1' or 'df2'.")
 
-    overlap_duration = overlap_end_1d - overlap_start_1d
-    mask = overlap_duration > 0
+    overlap_map = defaultdict(set)
 
-    overlap_start_1d = overlap_start_1d[mask]
-    overlap_end_1d = overlap_end_1d[mask]
+    def process_func(df1_index, df2_row):
+        if key_df == "df1":
+            overlap_map[df1_index].add(df2_row.Index)
+        else:
+            overlap_map[df2_row.Index].add(df1_index)
 
-    row_indices = np.repeat(np.arange(overlap_start.shape[0]), overlap_start.shape[1])[
-        mask
-    ]
+    process_overlapping_ranges(df1, df2, process_func, fully_contained)
+    return overlap_map
 
-    return overlap_start_1d, overlap_end_1d, row_indices
 
-
-def filter_non_overlapping_from_df(df):
-    """Filter out ranges that have no overlaps with other ranges in the
-    same DataFrame."""
-    group_df = df.assign(group=group_overlapping_ranges(df))
-
-    # Ranges that have no shared groups with the first dataframe are excluded,
-    # as they cannot overlap.
-    group_counts = group_df["group"].value_counts()
-    filtered_groups = group_counts[group_counts >= 2].index
-    filtered_df = group_df[group_df["group"].isin(filtered_groups)]
-
-    return filtered_df.drop(columns=["group"])
-
-
-def filter_non_overlapping_from_dfs(df1, df2):
-    """Filter out ranges that have no overlaps between two DataFrames."""
-    type_df1 = df1.assign(type="df1")
-    type_df2 = df2.assign(type="df2")
-
-    all_df = pd.concat([type_df1, type_df2]).reset_index(drop=True)
-    all_df["group"] = group_overlapping_ranges(all_df)
-
-    group_df1 = all_df[all_df["type"] == "df1"]
-    group_df2 = all_df[all_df["type"] == "df2"]
-
-    filtered_df1 = group_df1[group_df1["group"].isin(group_df2["group"])].drop(
-        columns=["group", "type"]
-    )
-    filtered_df2 = group_df2[group_df2["group"].isin(group_df1["group"])].drop(
-        columns=["group", "type"]
-    )
-
-    return filtered_df1, filtered_df2
-
-
-def calculate_overlapping_ranges(df1, df2):
+def calculate_overlapping_ranges(df1, df2=None):
     """Calculate the overlapping ranges between two dataframes.
 
     Parameters
@@ -164,35 +187,65 @@ def calculate_overlapping_ranges(df1, df2):
     df1 : dataframe
         DataFrame containing ranges to calculate the overlap from, with 'start'
         and 'end' columns.
-    df2 : dataframe
+    df2 : dataframe, optional
         DataFrame containing ranges to calculate the overlap with, with 'start'
-        and 'end' columns.
+        and 'end' columns. If not provided, the function calculates the
+        overlap within df1.
 
     Returns
     -------
     result : dataframe
-        DataFrame containing overlapping ranges, with columns "start" and "end".
-        These ranges may not exactly match the original rows, as they could be
-        created by combining start and end values from different rows.
+        DataFrame containing overlapping ranges, with the following columns:
+        - start: start position of the overlap.
+        - end: end position of the overlap.
+        - original_index: index of the original row in df1.
+        These ranges may not exactly match the original ranges, as they could
+        be created by combining start and end values from different ranges.
     """
-    filtered_df1, filtered_df2 = filter_non_overlapping_from_dfs(df1, df2)
+    overlap_map = defaultdict(set)
+
+    def process_func(df1_index, df2_row):
+        overlap_map[df1_index].add((df2_row.Index, df2_row.start, df2_row.end))
+
+    if df2 is None:
+        process_overlapping_ranges(df1, df1, process_func)
+    else:
+        process_overlapping_ranges(df1, df2, process_func)
+
+    results = []
 
-    overlap_start, overlap_end = _calculate_overlap_start_end(
-        filtered_df1, filtered_df2
-    )
-    overlap_start, overlap_end, _ = _flatten_and_filter_invalid_ranges(
-        overlap_start, overlap_end
-    )
+    for df1_row in df1.itertuples():
+        if df1_row.Index not in overlap_map:
+            continue
 
-    overlap = list(set(zip(overlap_start, overlap_end)))
-    return (
-        pd.DataFrame(overlap, columns=["start", "end"])
-        .sort_values("start")
-        .reset_index(drop=True)
-    )
+        indices, starts, ends = zip(*overlap_map[df1_row.Index])
+        indices = np.array(indices)
+        starts_array = np.array(starts)
+        ends_array = np.array(ends)
+
+        # We don't want to consider the overlap between the same range
+        # instances.
+        if df2 is None:
+            non_self_mask = indices != df1_row.Index
+            starts_array = starts_array[non_self_mask]
+            ends_array = ends_array[non_self_mask]
 
+        overlap_start = np.maximum(df1_row.start, starts_array)
+        overlap_end = np.minimum(df1_row.end, ends_array)
+        overlap_duration = overlap_end - overlap_start
 
-def calculate_overlap_sum(df1, df2=None, divisor=1):
+        valid_overlap_mask = overlap_duration > 0
+        overlap_start = overlap_start[valid_overlap_mask]
+        overlap_end = overlap_end[valid_overlap_mask]
+
+        results.extend(
+            zip(overlap_start, overlap_end, [df1_row.Index] * len(overlap_start))
+        )
+
+    return pd.DataFrame(results, columns=["start", "end", "original_index"])
+
+
+def calculate_overlap_sum(df1, df2=None, consolidate=True):
     """Calculate the sum of overlapping durations between two dataframes.
 
     Parameters
@@ -215,86 +268,15 @@ def calculate_overlap_sum(df1, df2=None, divisor=1):
         Series containing the sum of overlapping durations for each row in df1.
         Non overlapping ranges will have a sum of 0.
     """
-    if df2 is None:
-        filtered_ranges_df = filter_non_overlapping_from_df(df1)
-        filtered_compare_df = filtered_ranges_df
-    else:
-        filtered_ranges_df, filtered_compare_df = filter_non_overlapping_from_dfs(
-            df1, df2
-        )
-
-    # The 'original_index' will map to df1.
-    filtered_ranges_df = filtered_ranges_df.reset_index().rename(
-        columns={"index": "original_index"}
-    )
-
-    if filtered_ranges_df.empty:
-        return pd.Series(0, index=df1.index)
-
-    # We split the array into chunks of size 'divisor' to break down the
-    # computation of the overlapping kernel truth table. This increases the
-    # computation time but helps reduce memory usage.
-    divisor = 1 if len(filtered_ranges_df) < divisor else divisor
-    chunks = len(filtered_ranges_df) / divisor
-    split_dfs = np.array_split(filtered_ranges_df, chunks)
+    overlap_df = calculate_overlapping_ranges(df1, df2)
 
-    for i, split_df in enumerate(split_dfs):
-        # The 'filtered_index' index will map to filtered_df, while the current
-        # index becomes local to split_df.
-        split_df = split_df.reset_index().rename(columns={"index": "filtered_index"})
-
-        overlap_start, overlap_end = _calculate_overlap_start_end(
-            split_df, filtered_compare_df
-        )
-
-        if df2 is None:
-            # We don't want to consider the overlap between the same range
-            # instances.
-            row_indices = split_df.index.tolist()
-            column_indices = split_df["filtered_index"].tolist()
-
-            overlap_start[row_indices, column_indices] = 0
-            overlap_end[row_indices, column_indices] = 0
-
-        (
-            overlap_start,
-            overlap_end,
-            original_indices,
-        ) = _flatten_and_filter_invalid_ranges(overlap_start, overlap_end)
-
-        overlap_df = pd.DataFrame(
-            {
-                "start": overlap_start,
-                "end": overlap_end,
-                "local_index": original_indices,
-            }
+    if consolidate:
+        overlap_df = (
+            overlap_df.assign(groups=group_overlapping_ranges(overlap_df))
+            .groupby(["original_index", "groups"])
+            .agg({"start": "min", "end": "max"})
         )
 
-        # If there are multiple overlapping ranges, we consolidate them to
-        # ensure that only one overlap is considered at a time, preventing the
-        # percentage from exceeding 100%.
-        overlap_df["groups"] = group_overlapping_ranges(overlap_df)
-        overlap_df = overlap_df.groupby(["local_index", "groups"]).agg(
-            {"start": "min", "end": "max"}
-        )
-
-        # We drop the group's index since we only use the first index
-        # to map to the index of the first dataframe.
-        if overlap_df.index.nlevels > 1:
-            overlap_df = overlap_df.reset_index(level=1, drop=True)
-
-        overlap_df["duration"] = overlap_df["end"] - overlap_df["start"]
-
-        split_df["Sum"] = overlap_df.groupby(overlap_df.index)["duration"].sum()
-        split_dfs[i] = split_df
-
-    split_df = pd.concat(split_dfs).reset_index(drop=True)
-
-    merged_df = df1.merge(
-        split_df.set_index("original_index"),
-        left_index=True,
-        right_index=True,
-        how="left",
-    )
-
-    return merged_df["Sum"].fillna(0)
+    overlap_df["duration"] = overlap_df["end"] - overlap_df["start"]
+    total_duration = overlap_df.groupby("original_index")["duration"].sum()
+    return total_duration.reindex(df1.index, fill_value=0.0).astype(float).round(1)
diff --git a/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py b/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py
--- a/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py
+++ b/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py
@@ -74,9 +74,12 @@ class NvtxGpuProjTrace(recipe.Recipe):
             graph_df = cuda.derive_graph_df(runtime_df, graph_events_df, cuda_gpu_df)
             cuda_df = pd.concat([cuda_df, graph_df], ignore_index=True)
 
-        service.filter_and_adjust_time(
+        err_msg = service.filter_and_adjust_time(
             cuda_df, start_column="gpu_start", end_column="gpu_end"
         )
+        if err_msg is not None:
+            logger.error(f"{report_path}: {err_msg}")
+            return None
 
         if cuda_df.empty or nvtx_df.empty:
             logger.info(
@@ -91,8 +94,6 @@ class NvtxGpuProjTrace(recipe.Recipe):
             )
             return None
 
-        proj_nvtx_df = nvtx.compute_hierarchy_info(proj_nvtx_df)
-
         name_dict = {
             "text": "Text",
             "start": "Start",
'''

patch_content_2025_1_1_65 = r'''diff --git a/nsys_recipe/lib/args.py b/nsys_recipe/lib/args.py
--- a/nsys_recipe/lib/args.py
+++ b/nsys_recipe/lib/args.py
@@ -40,6 +40,8 @@ class Option(Enum):
     FILTER_TIME = 13
     FILTER_PROJECTED_NVTX = 14
     HIDE_INACTIVE = 15
+    PER_GPU = 16
+    PER_STREAM = 17
 
 
 def _replace_range(name, start_index, end_index, value):
@@ -489,6 +491,22 @@ class ArgumentParser(argparse.ArgumentParser):
                 "By default, all devices are shown.",
                 **kwargs,
             )
+        elif option == Option.PER_GPU:
+            group.add_argument(
+                "--per-gpu",
+                action="store_const",
+                const=["deviceId"],
+                default=[],
+                help="Group events by GPU.",
+            )
+        elif option == Option.PER_STREAM:
+            group.add_argument(
+                "--per-stream",
+                action="store_const",
+                const=["deviceId", "streamId"],
+                default=[],
+                help="Group events by stream within each GPU.",
+            )
         else:
             raise NotImplementedError("Invalid option.")
 
diff --git a/nsys_recipe/lib/data_utils.py b/nsys_recipe/lib/data_utils.py
--- a/nsys_recipe/lib/data_utils.py
+++ b/nsys_recipe/lib/data_utils.py
@@ -265,7 +265,7 @@ class RangeColumnUnifier:
             cuda_df = self._original_df
 
         proj_nvtx_df = nvtx.project_nvtx_onto_gpu(filtered_nvtx_df, cuda_df)
-        if proj_nvtx_df is None:
+        if proj_nvtx_df.empty:
             return
 
         self._filter_by_overlap(proj_nvtx_df)
diff --git a/nsys_recipe/lib/nvtx.py b/nsys_recipe/lib/nvtx.py
--- a/nsys_recipe/lib/nvtx.py
+++ b/nsys_recipe/lib/nvtx.py
@@ -7,6 +7,7 @@
 # disclosure or distribution of this material and related documentation
 # without an express license agreement from NVIDIA CORPORATION or
 # its affiliates is strictly prohibited.
+from collections import defaultdict
 
 import numpy as np
 import pandas as pd
@@ -66,43 +67,41 @@ def combine_text_fields(nvtx_df, str_df):
     return nvtx_textId_df.drop(columns=["textStr"])
 
 
-def compute_hierarchy_info(nvtx_df):
-    """Compute the hierarchy information of each NVTX range.
-
-    This function assumes that the input DataFrame is sorted by times. It
-    will add the following columns to the DataFrame:
-    - stackLevel: level of the range in the stack.
-    - parentId: ID of the parent range.
-    - rangeStack: IDs of the ranges that make up the stack.
-    - childrenCount: number of child ranges.
-    - rangeId: arbitrary ID of the range.
-    """
-    hierarchy_df = nvtx_df.copy()
-
-    hierarchy_df["parentId"] = None
-    hierarchy_df["stackLevel"] = 0
-    hierarchy_df["rangeStack"] = None
-
+def _compute_hierarchy_info(proj_nvtx_df, nvtx_stream_map):
+    hierarchy_df = proj_nvtx_df.assign(parentId=None, stackLevel=0, rangeStack=None)
     stack = []
 
     for row in hierarchy_df.itertuples():
-        while stack and stack[-1].end <= row.start:
+        while stack and row.end > stack[-1].end:
             stack.pop()
 
-        parent_index = stack[-1].Index if stack else np.nan
         stack.append(row)
 
-        hierarchy_df.at[row.Index, "parentId"] = parent_index
+        # Exclude ranges from the stack where the GPU operations are run on
+        # different streams by checking if the intersection of their streams is
+        # non-empty.
+        current_stack = [
+            r
+            for r in stack
+            if nvtx_stream_map[r.originalIndex] & nvtx_stream_map[row.originalIndex]
+        ]
+
+        # The current row is the last element of the stack.
+        hierarchy_df.at[row.Index, "parentId"] = (
+            current_stack[-2].Index if len(current_stack) > 1 else np.nan
+        )
         # The stack level starts at 0.
-        hierarchy_df.at[row.Index, "stackLevel"] = len(stack) - 1
-        hierarchy_df.at[row.Index, "rangeStack"] = [r.Index for r in stack]
+        hierarchy_df.at[row.Index, "stackLevel"] = len(current_stack) - 1
+        hierarchy_df.at[row.Index, "rangeStack"] = [r.Index for r in current_stack]
 
     hierarchy_df = hierarchy_df.reset_index().rename(columns={"index": "rangeId"})
-
     children_count = hierarchy_df["parentId"].value_counts()
     hierarchy_df["childrenCount"] = (
         hierarchy_df["rangeId"].map(children_count).fillna(0).astype(int)
     )
+    # Convert to Int64 to support missing values (pd.NA) while keeping
+    # integer type.
+    hierarchy_df["parentId"] = hierarchy_df["parentId"].astype("Int64")
 
     return hierarchy_df
 
@@ -139,94 +138,162 @@ def _add_individual_events(nvtx_gpu_start_dict, nvtx_gpu_end_dict, list_of_indiv
     return indices, starts, ends
 
 
-def _compute_gpu_projection_df(nvtx_df, cuda_df, cuda_nvtx_index_map):
+def _aggregate_cuda_ranges(
+    cuda_df, cuda_nvtx_index_map, innermost_nvtx_indices, row_offset
+):
     # Each NVTX index will be associated with the minimum start time and the
     # maximum end time of the CUDA operations that the corresponding NVTX range
     # encloses.
     nvtx_gpu_start_dict = {}
     nvtx_gpu_end_dict = {}
-    # list_of_individuals contains NVTX indices that should not be grouped.
-    # These items will be treated individually, using their original
-    # start and end times without aggregation.
-    list_of_individuals = []
+
+    indices = []
+    starts = []
+    ends = []
+
+    nvtx_stream_map = defaultdict(set)
 
     for cuda_row in cuda_df.itertuples():
         if cuda_row.Index not in cuda_nvtx_index_map:
             continue
 
         nvtx_indices = cuda_nvtx_index_map[cuda_row.Index]
+
         for nvtx_index in nvtx_indices:
-            if hasattr(cuda_row, "groupId") and not pd.isna(cuda_row.groupId):
-                list_of_individuals.append(
-                    (nvtx_index, cuda_row.gpu_start, cuda_row.gpu_end)
-                )
+            nvtx_stream_map[nvtx_index].add(cuda_row.streamId)
+
+            start = cuda_row.gpu_start
+            end = cuda_row.gpu_end
+
+            # Handle cases where the innermost NVTX range encloses CUDA events
+            # that result in multiple GPU ranges (e.g. CUDA graphs). In this
+            # case, we don't group them and keep them as separate NVTX ranges.
+            if (
+                hasattr(cuda_row, "groupId")
+                and not pd.isna(cuda_row.groupId)
+                and nvtx_index in innermost_nvtx_indices
+            ):
+                indices.append(nvtx_index)
+                starts.append(start)
+                ends.append(end)
                 continue
+
             if nvtx_index not in nvtx_gpu_start_dict:
-                nvtx_gpu_start_dict[nvtx_index] = cuda_row.gpu_start
-                nvtx_gpu_end_dict[nvtx_index] = cuda_row.gpu_end
+                nvtx_gpu_start_dict[nvtx_index] = start
+                nvtx_gpu_end_dict[nvtx_index] = end
                 continue
-            if cuda_row.gpu_start < nvtx_gpu_start_dict[nvtx_index]:
-                nvtx_gpu_start_dict[nvtx_index] = cuda_row.gpu_start
-            if cuda_row.gpu_end > nvtx_gpu_end_dict[nvtx_index]:
-                nvtx_gpu_end_dict[nvtx_index] = cuda_row.gpu_end
 
-    indices, starts, ends = _add_individual_events(
-        nvtx_gpu_start_dict, nvtx_gpu_end_dict, list_of_individuals
-    )
+            if start < nvtx_gpu_start_dict[nvtx_index]:
+                nvtx_gpu_start_dict[nvtx_index] = start
+            if end > nvtx_gpu_end_dict[nvtx_index]:
+                nvtx_gpu_end_dict[nvtx_index] = end
 
-    df = pd.DataFrame(
-        {"text": nvtx_df.loc[indices, "text"], "start": starts, "end": ends}
-    ).reset_index()
+    indices += list(nvtx_gpu_start_dict.keys())
+    starts += list(nvtx_gpu_start_dict.values())
+    ends += list(nvtx_gpu_end_dict.values())
 
-    # Preserve original order for rows with identical "start" and "end" values
-    # using the index.
-    return (
-        df.sort_values(by=["start", "end", "index"], ascending=[True, False, True])
-        .drop(columns=["index"])
-        .reset_index(drop=True)
+    df = (
+        pd.DataFrame({"originalIndex": indices, "start": starts, "end": ends})
+        # Preserve original order for rows with identical "start" and "end"
+        # values using the index.
+        .sort_values(
+            by=["start", "end", "originalIndex"], ascending=[True, False, True]
+        ).reset_index(drop=True)
     )
 
+    df.index = range(row_offset, row_offset + len(df))
+    return _compute_hierarchy_info(df, nvtx_stream_map)
 
-def _compute_grouped_gpu_projection_df(
-    nvtx_df, cuda_df, cuda_nvtx_index_map, per_gpu=False, per_stream=False
-):
-    group_by_elements = []
-    if per_stream:
-        group_by_elements.append("streamId")
-    if per_gpu:
-        group_by_elements.append("deviceId")
 
-    if not group_by_elements:
-        return _compute_gpu_projection_df(nvtx_df, cuda_df, cuda_nvtx_index_map)
+def _compute_gpu_projection_df(
+    cuda_df, group_columns, cuda_nvtx_index_map, innermost_nvtx_indices, row_offset
+):
+    if group_columns:
+        cuda_gdf = cuda_df.groupby(group_columns)
+    else:
+        cuda_gdf = [(None, cuda_df)]
 
     dfs = []
-    cuda_gdf = cuda_df.groupby(group_by_elements)
-
     for group_keys, cuda_group_df in cuda_gdf:
-        df = _compute_gpu_projection_df(nvtx_df, cuda_group_df, cuda_nvtx_index_map)
+        df = _aggregate_cuda_ranges(
+            cuda_group_df, cuda_nvtx_index_map, innermost_nvtx_indices, row_offset
+        )
         if df.empty:
             continue
 
-        if per_stream:
-            df["streamId"] = group_keys[group_by_elements.index("streamId")]
-        if per_gpu:
-            df["deviceId"] = group_keys[group_by_elements.index("deviceId")]
+        row_offset += len(df)
+
+        for key in group_columns:
+            df[key] = group_keys[group_columns.index(key)]
+
         dfs.append(df)
 
+    if not dfs:
+        return pd.DataFrame()
+
     return pd.concat(dfs, ignore_index=True)
 
 
-def project_nvtx_onto_gpu(nvtx_df, cuda_df, per_gpu=False, per_stream=False):
+def _validate_group_columns(df, group_columns):
+    if isinstance(group_columns, str):
+        group_columns = [group_columns]
+    elif group_columns is None:
+        group_columns = []
+
+    for col in group_columns:
+        if col not in df.columns:
+            raise ValueError(f"Column '{col}' not found in the DataFrame.")
+
+    return group_columns
+
+
+def _get_innermost_nvtx_indices(nvtx_df):
+    parent_nvtx_df = pd.Series(np.nan, index=nvtx_df.index)
+    stack = []
+
+    for row in nvtx_df.itertuples():
+        while stack and row.end > stack[-1].end:
+            stack.pop()
+
+        if stack:
+            parent_nvtx_df[row.Index] = stack[-1].Index
+
+        stack.append(row)
+
+    parent_ids = parent_nvtx_df.dropna().unique()
+    return set(parent_nvtx_df[~parent_nvtx_df.index.isin(parent_ids)].index)
+
+
+def project_nvtx_onto_gpu(nvtx_df, cuda_df, group_columns=None):
     """Project the NVTX ranges from the CPU onto the GPU.
 
     The projected range will have the start timestamp of the first enclosed GPU
     operation and the end timestamp of the last enclosed GPU operation.
 
+    Parameters
+    ----------
+    nvtx_df : pd.DataFrame
+        DataFrame containing NVTX ranges.
+    cuda_df : pd.DataFrame
+        DataFrame containing CUDA events. It must contain both the runtime and
+        GPU operations.
+    group_columns : str or list of str, optional
+        Column names in the CUDA table by which events should be grouped when
+        the associated NVTX range is projected onto the GPU.
+
     Returns
     -------
-    proj_nvtx_df : pd.DataFrame or None
-        DataFrame with projected NVTX ranges, or None if none are found.
+    proj_nvtx_df : pd.DataFrame
+        DataFrame with projected NVTX ranges and additional columns for the
+        hierarchy information, including:
+        - stackLevel: level of the range in the stack.
+        - parentId: ID of the parent range.
+        - rangeStack: IDs of the ranges that make up the stack.
+        - childrenCount: number of child ranges.
+        - rangeId: arbitrary ID of the range.
     """
+    group_columns = _validate_group_columns(cuda_df, group_columns)
+
     # Filter ranges that are incomplete or end on a different thread.
     filtered_nvtx_df = nvtx_df[
         nvtx_df["start"].notnull()
@@ -238,6 +305,7 @@ def project_nvtx_onto_gpu(nvtx_df, cuda_df, per_gpu=False, per_stream=False):
     cuda_gdf = cuda_df.groupby("globalTid")
 
     dfs = []
+    total_rows = 0
 
     for global_tid, nvtx_tid_df in nvtx_gdf:
         if global_tid not in cuda_gdf.groups:
@@ -248,20 +316,44 @@ def project_nvtx_onto_gpu(nvtx_df, cuda_df, per_gpu=False, per_stream=False):
             nvtx_tid_df, cuda_tid_df, fully_contained=True
         )
 
-        df = _compute_grouped_gpu_projection_df(
-            filtered_nvtx_df, cuda_tid_df, cuda_nvtx_index_map, per_gpu, per_stream
+        innermost_nvtx_indices = _get_innermost_nvtx_indices(nvtx_tid_df)
+        df = _compute_gpu_projection_df(
+            cuda_tid_df,
+            group_columns,
+            cuda_nvtx_index_map,
+            innermost_nvtx_indices,
+            total_rows,
         )
         if df.empty:
             continue
 
+        total_rows += len(df)
+
+        df["text"] = df["originalIndex"].map(nvtx_tid_df["text"])
         # The values of pid and tid are the same within each group of globalTid.
-        df["pid"] = nvtx_tid_df["pid"].iat[0]
-        df["tid"] = nvtx_tid_df["tid"].iat[0]
+        for col in ["pid", "tid"]:
+            df[col] = nvtx_tid_df[col].iat[0]
+
+        df = df.drop(columns=["originalIndex"])
 
         dfs.append(df)
 
     if not dfs:
-        return None
+        return pd.DataFrame(
+            columns=[
+                "text",
+                "start",
+                "end",
+                "pid",
+                "tid",
+                "stackLevel",
+                "parentId",
+                "rangeStack",
+                "childrenCount",
+                "rangeId",
+            ]
+            + group_columns
+        )
 
     return pd.concat(dfs, ignore_index=True)
 
diff --git a/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py b/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py
--- a/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py
+++ b/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py
@@ -87,17 +87,14 @@ class NvtxGpuProjTrace(recipe.Recipe):
             )
             return None
 
-        proj_nvtx_df = nvtx.project_nvtx_onto_gpu(
-            nvtx_df, cuda_df, parsed_args.per_gpu, parsed_args.per_stream
-        )
-        if proj_nvtx_df is None:
+        group_columns = parsed_args.per_gpu or parsed_args.per_stream
+        proj_nvtx_df = nvtx.project_nvtx_onto_gpu(nvtx_df, cuda_df, group_columns)
+        if proj_nvtx_df.empty:
             logger.info(
                 f"{report_path}: Report does not contain any NVTX data that can be projected onto the GPU."
             )
             return None
 
-        proj_nvtx_df = nvtx.compute_hierarchy_info(proj_nvtx_df)
-
         name_dict = {
             "text": "Text",
             "start": "Start",
@@ -111,9 +108,9 @@ class NvtxGpuProjTrace(recipe.Recipe):
             "rangeStack": "Range Stack",
         }
 
-        if parsed_args.per_gpu:
+        if "deviceId" in group_columns:
             name_dict["deviceId"] = "Device ID"
-        if parsed_args.per_stream:
+        if "streamId" in group_columns:
             name_dict["streamId"] = "Stream ID"
 
         proj_nvtx_df = proj_nvtx_df.rename(columns=name_dict)[name_dict.values()]
@@ -179,16 +176,10 @@ class NvtxGpuProjTrace(recipe.Recipe):
         parser.add_recipe_argument(Option.START)
         parser.add_recipe_argument(Option.END)
         parser.add_recipe_argument(Option.CSV)
-        parser.add_recipe_argument(
-            "--per-gpu",
-            action="store_true",
-            help="Give the results per GPU.",
-        )
-        parser.add_recipe_argument(
-            "--per-stream",
-            action="store_true",
-            help="Give the results per stream.",
-        )
+
+        per_group = parser.recipe_group.add_mutually_exclusive_group()
+        parser.add_argument_to_group(per_group, Option.PER_GPU)
+        parser.add_argument_to_group(per_group, Option.PER_STREAM)
 
         filter_group = parser.recipe_group.add_mutually_exclusive_group()
         parser.add_argument_to_group(filter_group, Option.FILTER_TIME)
'''

patch_content_2025_1_1_110 = r'''
diff --git a/nsys_recipe/lib/args.py b/nsys_recipe/lib/args.py
--- a/nsys_recipe/lib/args.py
+++ b/nsys_recipe/lib/args.py
@@ -40,6 +40,8 @@ class Option(Enum):
     FILTER_TIME = 13
     FILTER_PROJECTED_NVTX = 14
     HIDE_INACTIVE = 15
+    PER_GPU = 16
+    PER_STREAM = 17
 
 
 def _replace_range(name, start_index, end_index, value):
@@ -489,6 +491,22 @@ class ArgumentParser(argparse.ArgumentParser):
                 "By default, all devices are shown.",
                 **kwargs,
             )
+        elif option == Option.PER_GPU:
+            group.add_argument(
+                "--per-gpu",
+                action="store_const",
+                const=["deviceId"],
+                default=[],
+                help="Group events by GPU.",
+            )
+        elif option == Option.PER_STREAM:
+            group.add_argument(
+                "--per-stream",
+                action="store_const",
+                const=["deviceId", "streamId"],
+                default=[],
+                help="Group events by stream within each GPU.",
+            )
         else:
             raise NotImplementedError("Invalid option.")
 
diff --git a/nsys_recipe/lib/data_utils.py b/nsys_recipe/lib/data_utils.py
--- a/nsys_recipe/lib/data_utils.py
+++ b/nsys_recipe/lib/data_utils.py
@@ -265,7 +265,7 @@ class RangeColumnUnifier:
             cuda_df = self._original_df
 
         proj_nvtx_df = nvtx.project_nvtx_onto_gpu(filtered_nvtx_df, cuda_df)
-        if proj_nvtx_df is None:
+        if proj_nvtx_df.empty:
             return
 
         self._filter_by_overlap(proj_nvtx_df)
diff --git a/nsys_recipe/lib/nvtx.py b/nsys_recipe/lib/nvtx.py
--- a/nsys_recipe/lib/nvtx.py
+++ b/nsys_recipe/lib/nvtx.py
@@ -7,6 +7,7 @@
 # disclosure or distribution of this material and related documentation
 # without an express license agreement from NVIDIA CORPORATION or
 # its affiliates is strictly prohibited.
+from collections import defaultdict
 
 import numpy as np
 import pandas as pd
@@ -66,43 +67,41 @@ def combine_text_fields(nvtx_df, str_df):
     return nvtx_textId_df.drop(columns=["textStr"])
 
 
-def compute_hierarchy_info(nvtx_df):
-    """Compute the hierarchy information of each NVTX range.
-
-    This function assumes that the input DataFrame is sorted by times. It
-    will add the following columns to the DataFrame:
-    - stackLevel: level of the range in the stack.
-    - parentId: ID of the parent range.
-    - rangeStack: IDs of the ranges that make up the stack.
-    - childrenCount: number of child ranges.
-    - rangeId: arbitrary ID of the range.
-    """
-    hierarchy_df = nvtx_df.copy()
-
-    hierarchy_df["parentId"] = None
-    hierarchy_df["stackLevel"] = 0
-    hierarchy_df["rangeStack"] = None
-
+def _compute_hierarchy_info(proj_nvtx_df, nvtx_stream_map):
+    hierarchy_df = proj_nvtx_df.assign(parentId=None, stackLevel=0, rangeStack=None)
     stack = []
 
     for row in hierarchy_df.itertuples():
-        while stack and stack[-1].end <= row.start:
+        while stack and row.end > stack[-1].end:
             stack.pop()
 
-        parent_index = stack[-1].Index if stack else np.nan
         stack.append(row)
 
-        hierarchy_df.at[row.Index, "parentId"] = parent_index
+        # Exclude ranges from the stack where the GPU operations are run on
+        # different streams by checking if the intersection of their streams is
+        # non-empty.
+        current_stack = [
+            r
+            for r in stack
+            if nvtx_stream_map[r.originalIndex] & nvtx_stream_map[row.originalIndex]
+        ]
+
+        # The current row is the last element of the stack.
+        hierarchy_df.at[row.Index, "parentId"] = (
+            current_stack[-2].Index if len(current_stack) > 1 else np.nan
+        )
         # The stack level starts at 0.
-        hierarchy_df.at[row.Index, "stackLevel"] = len(stack) - 1
-        hierarchy_df.at[row.Index, "rangeStack"] = [r.Index for r in stack]
+        hierarchy_df.at[row.Index, "stackLevel"] = len(current_stack) - 1
+        hierarchy_df.at[row.Index, "rangeStack"] = [r.Index for r in current_stack]
 
     hierarchy_df = hierarchy_df.reset_index().rename(columns={"index": "rangeId"})
-
     children_count = hierarchy_df["parentId"].value_counts()
     hierarchy_df["childrenCount"] = (
         hierarchy_df["rangeId"].map(children_count).fillna(0).astype(int)
     )
+    # Convert to Int64 to support missing values (pd.NA) while keeping
+    # integer type.
+    hierarchy_df["parentId"] = hierarchy_df["parentId"].astype("Int64")
 
     return hierarchy_df
 
@@ -139,98 +138,165 @@ def _add_individual_events(nvtx_gpu_start_dict, nvtx_gpu_end_dict, list_of_indiv
     return indices, starts, ends
 
 
-def _compute_gpu_projection_df(nvtx_df, cuda_df, cuda_nvtx_index_map):
+def _aggregate_cuda_ranges(
+    cuda_df, cuda_nvtx_index_map, innermost_nvtx_indices, row_offset
+):
     # Each NVTX index will be associated with the minimum start time and the
     # maximum end time of the CUDA operations that the corresponding NVTX range
     # encloses.
     nvtx_gpu_start_dict = {}
     nvtx_gpu_end_dict = {}
-    # list_of_individuals contains NVTX indices that should not be grouped.
-    # These items will be treated individually, using their original
-    # start and end times without aggregation.
-    list_of_individuals = []
+
+    indices = []
+    starts = []
+    ends = []
+
+    nvtx_stream_map = defaultdict(set)
 
     for cuda_row in cuda_df.itertuples():
         if cuda_row.Index not in cuda_nvtx_index_map:
             continue
 
         nvtx_indices = cuda_nvtx_index_map[cuda_row.Index]
+
         for nvtx_index in nvtx_indices:
-            if hasattr(cuda_row, "groupId") and not pd.isna(cuda_row.groupId):
-                list_of_individuals.append(
-                    (nvtx_index, cuda_row.gpu_start, cuda_row.gpu_end)
-                )
+            nvtx_stream_map[nvtx_index].add(cuda_row.streamId)
+
+            start = cuda_row.gpu_start
+            end = cuda_row.gpu_end
+
+            # Handle cases where the innermost NVTX range encloses CUDA events
+            # that result in multiple GPU ranges (e.g. CUDA graphs). In this
+            # case, we don't group them and keep them as separate NVTX ranges.
+            if (
+                hasattr(cuda_row, "groupId")
+                and not pd.isna(cuda_row.groupId)
+                and nvtx_index in innermost_nvtx_indices
+            ):
+                indices.append(nvtx_index)
+                starts.append(start)
+                ends.append(end)
                 continue
+
             if nvtx_index not in nvtx_gpu_start_dict:
-                nvtx_gpu_start_dict[nvtx_index] = cuda_row.gpu_start
-                nvtx_gpu_end_dict[nvtx_index] = cuda_row.gpu_end
+                nvtx_gpu_start_dict[nvtx_index] = start
+                nvtx_gpu_end_dict[nvtx_index] = end
                 continue
-            if cuda_row.gpu_start < nvtx_gpu_start_dict[nvtx_index]:
-                nvtx_gpu_start_dict[nvtx_index] = cuda_row.gpu_start
-            if cuda_row.gpu_end > nvtx_gpu_end_dict[nvtx_index]:
-                nvtx_gpu_end_dict[nvtx_index] = cuda_row.gpu_end
 
-    indices, starts, ends = _add_individual_events(
-        nvtx_gpu_start_dict, nvtx_gpu_end_dict, list_of_individuals
-    )
+            if start < nvtx_gpu_start_dict[nvtx_index]:
+                nvtx_gpu_start_dict[nvtx_index] = start
+            if end > nvtx_gpu_end_dict[nvtx_index]:
+                nvtx_gpu_end_dict[nvtx_index] = end
 
-    df = pd.DataFrame(
-        {"text": nvtx_df.loc[indices, "text"], "start": starts, "end": ends}
-    ).reset_index()
+    indices += list(nvtx_gpu_start_dict.keys())
+    starts += list(nvtx_gpu_start_dict.values())
+    ends += list(nvtx_gpu_end_dict.values())
 
-    # Preserve original order for rows with identical "start" and "end" values
-    # using the index.
-    return (
-        df.sort_values(by=["start", "end", "index"], ascending=[True, False, True])
-        .drop(columns=["index"])
-        .reset_index(drop=True)
+    df = (
+        pd.DataFrame({"originalIndex": indices, "start": starts, "end": ends})
+        # Preserve original order for rows with identical "start" and "end"
+        # values using the index.
+        .sort_values(
+            by=["start", "end", "originalIndex"], ascending=[True, False, True]
+        ).reset_index(drop=True)
     )
 
+    df.index = range(row_offset, row_offset + len(df))
+    return _compute_hierarchy_info(df, nvtx_stream_map)
 
-def _compute_grouped_gpu_projection_df(
-    nvtx_df, cuda_df, cuda_nvtx_index_map, per_gpu=False, per_stream=False
-):
-    group_by_elements = []
-    if per_stream:
-        group_by_elements.append("streamId")
-    if per_gpu:
-        group_by_elements.append("deviceId")
 
-    if not group_by_elements:
-        df = _compute_gpu_projection_df(nvtx_df, cuda_df, cuda_nvtx_index_map)
-        return df if not df.empty else None
+def _compute_gpu_projection_df(
+    cuda_df, group_columns, cuda_nvtx_index_map, innermost_nvtx_indices, row_offset
+):
+    if group_columns:
+        cuda_gdf = cuda_df.groupby(group_columns)
+    else:
+        cuda_gdf = [(None, cuda_df)]
 
     dfs = []
-    cuda_gdf = cuda_df.groupby(group_by_elements)
-
     for group_keys, cuda_group_df in cuda_gdf:
-        df = _compute_gpu_projection_df(nvtx_df, cuda_group_df, cuda_nvtx_index_map)
+        df = _aggregate_cuda_ranges(
+            cuda_group_df, cuda_nvtx_index_map, innermost_nvtx_indices, row_offset
+        )
         if df.empty:
             continue
 
-        if per_stream:
-            df["streamId"] = group_keys[group_by_elements.index("streamId")]
-        if per_gpu:
-            df["deviceId"] = group_keys[group_by_elements.index("deviceId")]
+        row_offset += len(df)
+
+        for key in group_columns:
+            df[key] = group_keys[group_columns.index(key)]
+
         dfs.append(df)
 
+    if not dfs:
+        return pd.DataFrame()
+
     if not dfs:
         return None
 
     return pd.concat(dfs, ignore_index=True)
 
 
-def project_nvtx_onto_gpu(nvtx_df, cuda_df, per_gpu=False, per_stream=False):
+def _validate_group_columns(df, group_columns):
+    if isinstance(group_columns, str):
+        group_columns = [group_columns]
+    elif group_columns is None:
+        group_columns = []
+
+    for col in group_columns:
+        if col not in df.columns:
+            raise ValueError(f"Column '{col}' not found in the DataFrame.")
+
+    return group_columns
+
+
+def _get_innermost_nvtx_indices(nvtx_df):
+    parent_nvtx_df = pd.Series(np.nan, index=nvtx_df.index)
+    stack = []
+
+    for row in nvtx_df.itertuples():
+        while stack and row.end > stack[-1].end:
+            stack.pop()
+
+        if stack:
+            parent_nvtx_df[row.Index] = stack[-1].Index
+
+        stack.append(row)
+
+    parent_ids = parent_nvtx_df.dropna().unique()
+    return set(parent_nvtx_df[~parent_nvtx_df.index.isin(parent_ids)].index)
+
+
+def project_nvtx_onto_gpu(nvtx_df, cuda_df, group_columns=None):
     """Project the NVTX ranges from the CPU onto the GPU.
 
     The projected range will have the start timestamp of the first enclosed GPU
     operation and the end timestamp of the last enclosed GPU operation.
 
+    Parameters
+    ----------
+    nvtx_df : pd.DataFrame
+        DataFrame containing NVTX ranges.
+    cuda_df : pd.DataFrame
+        DataFrame containing CUDA events. It must contain both the runtime and
+        GPU operations.
+    group_columns : str or list of str, optional
+        Column names in the CUDA table by which events should be grouped when
+        the associated NVTX range is projected onto the GPU.
+
     Returns
     -------
-    proj_nvtx_df : pd.DataFrame or None
-        DataFrame with projected NVTX ranges, or None if none are found.
+    proj_nvtx_df : pd.DataFrame
+        DataFrame with projected NVTX ranges and additional columns for the
+        hierarchy information, including:
+        - stackLevel: level of the range in the stack.
+        - parentId: ID of the parent range.
+        - rangeStack: IDs of the ranges that make up the stack.
+        - childrenCount: number of child ranges.
+        - rangeId: arbitrary ID of the range.
     """
+    group_columns = _validate_group_columns(cuda_df, group_columns)
+
     # Filter ranges that are incomplete or end on a different thread.
     filtered_nvtx_df = nvtx_df[
         nvtx_df["start"].notnull()
@@ -242,6 +308,7 @@ def project_nvtx_onto_gpu(nvtx_df, cuda_df, per_gpu=False, per_stream=False):
     cuda_gdf = cuda_df.groupby("globalTid")
 
     dfs = []
+    total_rows = 0
 
     for global_tid, nvtx_tid_df in nvtx_gdf:
         if global_tid not in cuda_gdf.groups:
@@ -252,20 +319,44 @@ def project_nvtx_onto_gpu(nvtx_df, cuda_df, per_gpu=False, per_stream=False):
             nvtx_tid_df, cuda_tid_df, fully_contained=True
         )
 
-        df = _compute_grouped_gpu_projection_df(
-            filtered_nvtx_df, cuda_tid_df, cuda_nvtx_index_map, per_gpu, per_stream
+        innermost_nvtx_indices = _get_innermost_nvtx_indices(nvtx_tid_df)
+        df = _compute_gpu_projection_df(
+            cuda_tid_df,
+            group_columns,
+            cuda_nvtx_index_map,
+            innermost_nvtx_indices,
+            total_rows,
         )
         if df is None:
             continue
 
+        total_rows += len(df)
+
+        df["text"] = df["originalIndex"].map(nvtx_tid_df["text"])
         # The values of pid and tid are the same within each group of globalTid.
-        df["pid"] = nvtx_tid_df["pid"].iat[0]
-        df["tid"] = nvtx_tid_df["tid"].iat[0]
+        for col in ["pid", "tid"]:
+            df[col] = nvtx_tid_df[col].iat[0]
+
+        df = df.drop(columns=["originalIndex"])
 
         dfs.append(df)
 
     if not dfs:
-        return None
+        return pd.DataFrame(
+            columns=[
+                "text",
+                "start",
+                "end",
+                "pid",
+                "tid",
+                "stackLevel",
+                "parentId",
+                "rangeStack",
+                "childrenCount",
+                "rangeId",
+            ]
+            + group_columns
+        )
 
     return pd.concat(dfs, ignore_index=True)
 
diff --git a/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py b/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py
--- a/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py
+++ b/nsys_recipe/recipes/nvtx_gpu_proj_trace/nvtx_gpu_proj_trace.py
@@ -87,17 +87,14 @@ class NvtxGpuProjTrace(recipe.Recipe):
             )
             return None
 
-        proj_nvtx_df = nvtx.project_nvtx_onto_gpu(
-            nvtx_df, cuda_df, parsed_args.per_gpu, parsed_args.per_stream
-        )
-        if proj_nvtx_df is None:
+        group_columns = parsed_args.per_gpu or parsed_args.per_stream
+        proj_nvtx_df = nvtx.project_nvtx_onto_gpu(nvtx_df, cuda_df, group_columns)
+        if proj_nvtx_df.empty:
             logger.info(
                 f"{report_path}: Report does not contain any NVTX data that can be projected onto the GPU."
             )
             return None
 
-        proj_nvtx_df = nvtx.compute_hierarchy_info(proj_nvtx_df)
-
         name_dict = {
             "text": "Text",
             "start": "Start",
@@ -111,9 +108,9 @@ class NvtxGpuProjTrace(recipe.Recipe):
             "rangeStack": "Range Stack",
         }
 
-        if parsed_args.per_gpu:
+        if "deviceId" in group_columns:
             name_dict["deviceId"] = "Device ID"
-        if parsed_args.per_stream:
+        if "streamId" in group_columns:
             name_dict["streamId"] = "Stream ID"
 
         proj_nvtx_df = proj_nvtx_df.rename(columns=name_dict)[name_dict.values()]
@@ -179,16 +176,10 @@ class NvtxGpuProjTrace(recipe.Recipe):
         parser.add_recipe_argument(Option.START)
         parser.add_recipe_argument(Option.END)
         parser.add_recipe_argument(Option.CSV)
-        parser.add_recipe_argument(
-            "--per-gpu",
-            action="store_true",
-            help="Give the results per GPU.",
-        )
-        parser.add_recipe_argument(
-            "--per-stream",
-            action="store_true",
-            help="Give the results per stream.",
-        )
+
+        per_group = parser.recipe_group.add_mutually_exclusive_group()
+        parser.add_argument_to_group(per_group, Option.PER_GPU)
+        parser.add_argument_to_group(per_group, Option.PER_STREAM)
 
         filter_group = parser.recipe_group.add_mutually_exclusive_group()
         parser.add_argument_to_group(filter_group, Option.FILTER_TIME)
'''


def main():
    """
    Entrypoint for nsys-jax-patch-nsys.
    """
    nsys = shutil.which("nsys")
    assert nsys is not None, "nsys-jax-patch-nsys expects nsys to be installed"
    nsys_version = subprocess.check_output([nsys, "--version"], text=True)
    m = re.match(
        r"^NVIDIA Nsight Systems version (\d+\.\d+\.\d+)\.(\d+)-\d+v\d+$", nsys_version
    )
    assert m is not None, f"Could not parse: {nsys_version}"
    match m.group(1):
        case "2024.5.1" | "2024.6.1":
            patch_content = patch_content_2024_5_1_and_2024_6_1
        case "2024.6.2":
            patch_content = patch_content_2024_6_2
        case "2025.1.1":
            match m.group(2):
                case "65":
                    patch_content = patch_content_2025_1_1_65
                case "110":
                    patch_content = patch_content_2025_1_1_110
                case _:
                    raise Exception(f"{m.group(1)}.{m.group(2)} patch not known")
        case _:
            patch_content = None
    if patch_content is not None:
        print(f"Patching Nsight Systems version {m.group(1)}.{m.group(2)}")
        nsys_recipe_help = subprocess.check_output(
            [nsys, "recipe", "--help"], text=True
        )
        m = re.search(
            r"List of required Python packages: '(.*?)/nsys_recipe/requirements/common.txt'",
            nsys_recipe_help,
        )
        assert m is not None, (
            f"Could not determine target directory from: {nsys_recipe_help}"
        )
        # e.g. /opt/nvidia/nsight-systems-cli/2024.7.1/target-linux-x64/python/packages
        subprocess.run(
            [shutil.which("git"), "apply"],
            cwd=m.group(1),
            input=patch_content,
            check=True,
            text=True,
        )
