from nsys_jax.data_loaders import _find_overlapped as find_overlapped
import pandas as pd
import pytest

@pytest.mark.parametrize("records,expected", [
    # no overlap
    ([], []),
    [[(0, 1)], []],
    [[(0, 1), (1, 2)], []],
    # overlap
    [[(0, 1), (0.5, 1.5)], [0, 1]],
    ([(0, 1), (0.5, 1.5), (8, 9), (9, 10)], [0, 1]),
    ([(0, 1), (2, 3), (2.5, 3.5), (3, 4), (5, 6)], [1, 2, 3]),
    ([(0, 1), (0.5, 1.5), (2, 3), (4.5, 5.5), (5, 6)], [0, 1, 3, 4]),
    # overlap between non-neighbouring ranges
    ([(0, 3), (0.1, 1), (2, 4), (5, 6)], [0, 1, 2]), # 0-1, 0-2 overlap but not 1-2
])
def test_find_overlapped(records, expected):
    df = pd.DataFrame.from_records(records, columns=["start", "end"])
    result = find_overlapped(df["start"], df["end"])
    assert list(result) == expected
