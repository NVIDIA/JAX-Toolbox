import pandas as pd
from typing import Optional

def remove_child_ranges(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """
    Return a new data frame with the children of ``df[mask]`` removed.

    This can be useful to erase excessive detail from a compilation trace, or make sure
    that a certain class of operation is accounted for as a higher-level concept (e.g.
    autotuning compilation) instead of as lower-level operations (emitting IR,
    optimizing IR, ...).
    """
    to_remove: Optional[pd.Series] = None
    mask &= df["NumChild"] != 0
    for row in df[mask].itertuples():
        child_mask = df["RangeStack"].str.startswith(f"{row.RangeStack}:")
        to_remove = child_mask if to_remove is None else child_mask | to_remove
        df.loc[row.Index, ["NumChild", "DurChildNs"]] = 0
        df.loc[row.Index, "DurNonChildNs"] = row.DurNs
    if to_remove is None:
        return df
    else:
        return df[~to_remove].copy()
