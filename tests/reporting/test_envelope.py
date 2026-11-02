"""Display reduction: the pixel envelope keeps every bucket's first, last, minimum and maximum
point (and the start of any gap), so short excursions survive downsampling."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.reporting.charts import extrema_envelope


def test_a_sub_bucket_spike_survives() -> None:
    times = np.arange(0, 10_000, dtype=np.int64)
    values = np.zeros(10_000)
    values[5_000] = 1_000_001.0  # lasts 1 ns of a 10,000 ns axis drawn on 10 buckets
    values[-1] = 1.0
    keep = extrema_envelope(times, values, 0, 10_000, 10)
    assert 5_000 in keep.tolist()
    assert keep[0] == 0 and keep[-1] == 9_999
    assert len(keep) <= 40
    assert values[keep].max() == 1_000_001.0
    assert not keep.flags.writeable


def test_gaps_are_kept_visible() -> None:
    times = np.arange(10, dtype=np.int64)
    values = np.array([0.0, 1.0, 2.0, np.nan, 2.5, 3.0, 2.0, 1.0, 0.5, 0.0])
    keep = extrema_envelope(times, values, 0, 10, 1).tolist()
    assert 3 in keep and 4 in keep  # start of the gap and the first point after it


def test_small_inputs_are_not_reduced() -> None:
    times = np.array([5, 7, 7, 9], np.int64)
    values = np.array([1.0, -1.0, 2.0, 0.0])
    assert extrema_envelope(times, values, 0, 100, 50).tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("times", "values", "error"),
    [
        (np.array([3, 1], np.int64), np.zeros(2), InvalidDataError),
        (np.array([1.0, 2.0]), np.zeros(2), InvalidDataError),
        (np.array([1, 2], np.int64), np.zeros(3), InvalidDataError),
    ],
)
def test_malformed_series_are_rejected(times: np.ndarray, values: np.ndarray, error: type[Exception]) -> None:
    with pytest.raises(error):
        extrema_envelope(times, values, 0, 10, 2)


@pytest.mark.parametrize(("start", "end", "buckets"), [(0, 10, 0), (10, 10, 3), (0, 10, True)])
def test_bad_display_geometry_is_rejected(start: int, end: int, buckets: int) -> None:
    with pytest.raises(InvalidConfigError):
        extrema_envelope(np.array([1], np.int64), np.zeros(1), start, end, buckets)


@st.composite
def series(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    steps = draw(st.lists(st.integers(0, 50), min_size=1, max_size=400))
    times = np.cumsum(np.array(steps, np.int64))
    values = np.array(
        draw(
            st.lists(
                st.one_of(st.floats(-1e6, 1e6, allow_nan=False), st.just(float("nan"))),
                min_size=len(steps),
                max_size=len(steps),
            )
        )
    )
    return times, values


@given(data=series(), buckets=st.integers(1, 40))
@settings(max_examples=200, deadline=None)
def test_every_bucket_keeps_its_first_last_min_and_max(data: tuple[np.ndarray, np.ndarray], buckets: int) -> None:
    times, values = data
    start, end = int(times[0]), int(times[-1]) + 1
    keep = set(extrema_envelope(times, values, start, end, buckets).tolist())
    width = -(-(end - start) // buckets)
    bucket = (times - start) // width
    for b in np.unique(bucket):
        idx = np.flatnonzero(bucket == b)
        assert idx[0] in keep and idx[-1] in keep
        finite = idx[np.isfinite(values[idx])]
        if finite.size:
            kept = [i for i in finite.tolist() if i in keep]
            assert values[kept].min() == values[finite].min()
            assert values[kept].max() == values[finite].max()
        if finite.size < idx.size:
            assert any(np.isnan(values[i]) for i in keep if bucket[i] == b)
    assert len(keep) <= 6 * np.unique(bucket).size  # first, last, min, max, gap start + after
