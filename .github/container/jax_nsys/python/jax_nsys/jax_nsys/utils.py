import pandas as pd  # type: ignore
from typing import Optional

pd.options.mode.copy_on_write = True


def make_child_mask(df: pd.DataFrame, parent_row: int) -> pd.Series:
    """
    Return a mask of descendants of the given range.
    """
    return df["RangeStack"].str.startswith(df.loc[parent_row, "RangeStack"] + ":")


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
        child_mask = make_child_mask(df, row.Index)
        to_remove = child_mask if to_remove is None else child_mask | to_remove
        df.loc[row.Index, ["NumChild", "DurChildNs"]] = 0
        df.loc[row.Index, "DurNonChildNs"] = row.DurNs
    return df if to_remove is None else df[~to_remove]
