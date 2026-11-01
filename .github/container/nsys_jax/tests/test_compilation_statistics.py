import pandas as pd  # type: ignore
import pytest  # type: ignore

from nsys_jax.analysis import generate_compilation_statistics


def _compile_df(profile_names):
    # Each profile: XlaCompile (main thread 1) contains a launcher range, which runs
    # "work" on worker threads 2 and 3 over [2, 6) ms and "main_tail" on the main thread.
    rows = [
        # RangeId, Name, TID, ParentId, StartMs, DurMs, DurChildMs, DurNonChildMs, RangeStack
        (1, "XlaCompile", 1, -1, 0.0, 10.0, 8.0, 2.0, "1"),
        (2, "launcher", 1, 1, 1.0, 8.0, 9.0, -1.0, "1:2"),
        (3, "work", 2, 2, 2.0, 4.0, 0.0, 4.0, "1:2:3"),
        (4, "work", 3, 2, 2.0, 4.0, 0.0, 4.0, "1:2:4"),
        (5, "main_tail", 1, 2, 7.0, 1.0, 0.0, 1.0, "1:2:5"),
    ]
    columns = [
        "RangeId",
        "Name",
        "TID",
        "ParentId",
        "StartMs",
        "DurMs",
        "DurChildMs",
        "DurNonChildMs",
        "RangeStack",
    ]
    frames = {}
    for i, name in enumerate(profile_names):
        df = pd.DataFrame.from_records(rows, columns=columns)
        # Each profile comes from a different process, so thread IDs differ.
        df["TID"] += 100 * i
        df["RangeId"] = df["RangeId"].astype("int32")
        frames[name] = df.set_index("RangeId")
    df = pd.concat(frames, names=["ProfileName", "RangeId"])
    return df


@pytest.mark.parametrize("profile_names", [["p0"], ["p0", "p1"]])
def test_parallel_compilation_attribution(profile_names):
    stats = generate_compilation_statistics(_compile_df(profile_names))
    n = len(profile_names)
    # The parallel region lasts 4 ms and both workers spend all of it on "work".
    assert stats.loc["work", "DurNonChildMs"] == pytest.approx(4.0 * n)
    # The launcher's own time excludes the parallel region and its main-thread child.
    assert stats.loc["launcher", "DurNonChildMs"] == pytest.approx(
        (8.0 - 4.0 - 1.0) * n
    )
    assert stats.loc["main_tail", "DurNonChildMs"] == pytest.approx(1.0 * n)
