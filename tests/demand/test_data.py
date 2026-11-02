"""Synthetic demand generator and series: determinism, seasonality, intermittency and delayed availability."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.core.serialization import canonical_json, loads_strict
from financeai.domains.demand import DemandSeries, DemandSpec, generate_demand

HOUR = 3_600 * 10**9
DAY = 24 * HOUR
START = 1_704_067_200 * 10**9  # 2024-01-01T00:00:00Z, a Monday


def _spec(**overrides: Any) -> DemandSpec:
    fields: dict[str, Any] = {"streams": ("a", "b"), "n_periods": 24 * 7 * 8, "seed": 11}
    fields.update(overrides)
    return DemandSpec(**fields)


def test_generation_is_deterministic_and_seeded_per_stream() -> None:
    first, second = generate_demand(_spec()), generate_demand(_spec())
    assert [s.stream for s in first] == ["a", "b"]
    for x, y in zip(first, second, strict=True):
        assert x.values.tobytes() == y.values.tobytes()
        assert x.content_hash == y.content_hash
    assert first[0].values.tobytes() != first[1].values.tobytes()
    assert generate_demand(_spec(seed=12))[0].values.tobytes() != first[0].values.tobytes()
    # adding a stream never changes the existing streams (seeds derive from the stream position, not a hash)
    more = generate_demand(_spec(streams=("a", "b", "c")))
    assert more[0].values.tobytes() == first[0].values.tobytes()
    assert more[1].values.tobytes() == first[1].values.tobytes()


def test_values_show_daily_and_weekly_seasonality_and_intermittency() -> None:
    spec = _spec(streams=("a",), n_periods=24 * 7 * 30, zero_probability=0.3, weekend_effect=-0.6)
    series = generate_demand(spec)[0]
    v = series.values
    assert bool(np.all(v >= 0)) and bool(np.all(v == np.round(v)))  # non-negative counts
    hours = np.arange(len(v)) % 24
    days = (np.arange(len(v)) // 24) % 7  # the series starts on a Monday
    assert v[hours == 12].mean() > 2.0 * v[hours == 0].mean()
    assert v[days >= 5].mean() < 0.8 * v[days < 5].mean()
    zeros = float(np.mean(v == 0))
    assert 0.25 <= zeros <= 0.5  # zero-inflation (p = 0.3) plus Poisson zeros at night


def test_level_shift_scales_demand_after_the_shift_period() -> None:
    n = 24 * 7 * 20
    base = generate_demand(_spec(streams=("a",), n_periods=n))[0].values
    shifted = generate_demand(_spec(streams=("a",), n_periods=n, shift_period=n // 2, shift_factor=2.0))[0].values
    assert base[: n // 2].tobytes() == shifted[: n // 2].tobytes()
    assert shifted[n // 2 :].mean() > 1.6 * base[n // 2 :].mean()


def test_labels_are_released_a_fixed_delay_after_their_period_ends() -> None:
    series = DemandSeries("s", START, HOUR, 3, np.arange(10, dtype=np.float64))
    assert series.n == 10
    assert series.cutoff_ns(4) == START + 4 * HOUR
    assert series.available_at_ns(0) == START + 4 * HOUR  # period [0, 1h) reported 3 h after it ends
    assert [series.visible_count(c) for c in range(0, 15)] == [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]
    for cutoff in range(15):
        released = series.visible(cutoff)
        assert all(series.available_at_ns(i) <= series.cutoff_ns(cutoff) for i in range(len(released)))
        if len(released) < series.n:
            assert series.available_at_ns(len(released)) > series.cutoff_ns(cutoff)
        assert not released.flags.writeable


def test_series_copies_and_freezes_values() -> None:
    values = np.array([1.0, 2.0, 3.0])
    series = DemandSeries("s", START, HOUR, 0, values)
    values[0] = 100.0
    assert series.values[0] == 1.0
    with pytest.raises(ValueError):
        series.values.setflags(write=True)
    changed = series.with_values(np.array([1.0, 2.0, 4.0]))
    assert changed.content_hash != series.content_hash and changed.stream == series.stream


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"values": np.array([1.0, -1.0])}, InvalidDataError),
        ({"values": np.array([1.0, np.nan])}, InvalidDataError),
        ({"values": np.array([], dtype=np.float64)}, InvalidDataError),
        ({"values": np.ones((2, 2))}, InvalidDataError),
        ({"period_ns": 7 * 10**9}, InvalidConfigError),  # does not divide a day
        ({"period_ns": 0}, InvalidConfigError),
        ({"label_delay_periods": -1}, InvalidConfigError),
        ({"start_ns": -5}, InvalidConfigError),
        ({"stream": ""}, InvalidConfigError),
    ],
)
def test_series_validation(kwargs: dict[str, Any], error: type[Exception]) -> None:
    fields: dict[str, Any] = {
        "stream": "s",
        "start_ns": START,
        "period_ns": HOUR,
        "label_delay_periods": 1,
        "values": np.array([1.0, 2.0]),
    }
    fields.update(kwargs)
    with pytest.raises(error):
        DemandSeries(**fields)


@pytest.mark.parametrize(
    "overrides",
    [
        {"streams": ()},
        {"streams": ("a", "a")},
        {"n_periods": 0},
        {"period_ns": 7 * 10**9},
        {"base_level": 0.0},
        {"zero_probability": 1.0},
        {"zero_probability": -0.1},
        {"daily_amplitude": float("nan")},
        {"shift_period": 10**9},
        {"shift_factor": 0.0},
        {"label_delay_periods": -1},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_spec_validation(overrides: dict[str, Any]) -> None:
    with pytest.raises(InvalidConfigError):
        _spec(**overrides)


def test_spec_mapping_round_trip_is_strict() -> None:
    spec = _spec(shift_period=100, shift_factor=1.5)
    doc = loads_strict(canonical_json(spec.to_mapping()))
    assert DemandSpec.from_mapping(doc) == spec
    assert doc["period_ns"] == str(HOUR) and doc["start_ns"] == str(START)
    with pytest.raises(InvalidConfigError):
        DemandSpec.from_mapping({**doc, "venue": "venue_a"})  # no market fields in a demand spec
    with pytest.raises(InvalidConfigError):
        DemandSpec.from_mapping({**doc, "period_ns": HOUR})
    assert dataclasses.replace(spec, seed=3) != spec
