"""Causal demand features: each row uses only values released by its own cutoff (oracle and future mutation)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.domains.demand import (
    DemandSeries,
    DemandSpec,
    FeatureSpec,
    compute_features,
    demand_target,
    generate_demand,
    make_query,
    mature_batch,
)

HOUR = 3_600 * 10**9
START = 1_704_067_200 * 10**9  # Monday 00:00 UTC
SPEC = FeatureSpec(lags=(24, 48, 168), windows=(24, 168))


def _series(delay: int = 3, weeks: int = 4) -> DemandSeries:
    return generate_demand(DemandSpec(streams=("s",), n_periods=24 * 7 * weeks, label_delay_periods=delay, seed=5))[0]


def test_feature_names_follow_the_spec_and_reject_unreleased_lags() -> None:
    assert SPEC.names(1, 3) == (
        "lag_4",
        "lag_24",
        "lag_48",
        "lag_168",
        "mean_24",
        "mean_168",
        "hod_sin",
        "hod_cos",
        "dow_sin",
        "dow_cos",
        "weekend",
        "phase",
    )
    assert SPEC.names(2, 0)[:4] == ("lag_2", "lag_24", "lag_48", "lag_168")
    with pytest.raises(InvalidConfigError, match="released"):
        FeatureSpec(lags=(2, 24)).names(1, 3)  # lag 2 would need a value reported after the cutoff
    for bad in ({"lags": (0,)}, {"lags": (24, 24)}, {"windows": (0,)}, {"lags": (True,)}, {"windows": (2.0,)}):
        with pytest.raises(InvalidConfigError):
            FeatureSpec(**bad)


def _oracle_row(series: DemandSeries, cutoff: int, horizon: int) -> list[float]:
    """Per-row reference built only from the released prefix (no index tricks)."""
    released = series.visible(cutoff)
    target = cutoff + horizon - 1
    row: list[float] = []
    for lag in (horizon + series.label_delay_periods, 24, 48, 168):
        index = target - lag
        assert index < len(released)  # the lag is released at the cutoff
        row.append(float(released[index]))
    row.extend(math.fsum(released[-window:].tolist()) / window for window in (24, 168))
    t = series.period_start_ns(target)
    day = (t % (24 * HOUR)) / (24 * HOUR)
    dow = (t // (24 * HOUR) + 3) % 7  # 1970-01-01 was a Thursday; 0 = Monday
    hour = (t % (24 * HOUR)) // HOUR
    row += [
        math.sin(2 * math.pi * day),
        math.cos(2 * math.pi * day),
        math.sin(2 * math.pi * dow / 7),
        math.cos(2 * math.pi * dow / 7),
        float(dow >= 5),
        float(dow * 24 + hour),
    ]
    return row


@pytest.mark.parametrize(("delay", "horizon"), [(3, 1), (0, 1), (5, 2)])
def test_vectorised_rows_equal_a_prefix_only_oracle(delay: int, horizon: int) -> None:
    series = _series(delay)
    features = compute_features(series, horizon_periods=horizon, spec=SPEC)
    assert features.first_cutoff == max(168 - horizon + 1, 168 + delay, delay + 1)
    assert features.last_cutoff == series.n - horizon
    for cutoff in (
        features.first_cutoff,
        features.first_cutoff + 37,
        (features.first_cutoff + series.n) // 2,
        features.last_cutoff,
    ):
        np.testing.assert_allclose(features.row(cutoff), _oracle_row(series, cutoff, horizon), rtol=1e-12, atol=1e-12)
    with pytest.raises(InvalidDataError):
        features.row(features.first_cutoff - 1)
    with pytest.raises(InvalidDataError):
        features.row(features.last_cutoff + 1)


def test_rows_do_not_change_when_unreleased_values_change() -> None:
    series = _series()
    features = compute_features(series, horizon_periods=1, spec=SPEC)
    rng = np.random.default_rng(0)
    for cutoff in (features.first_cutoff, 400, features.last_cutoff - 1):
        released = series.visible_count(cutoff)
        future = series.values.copy()
        future[released:] = rng.uniform(0, 1_000, size=series.n - released)
        mutated = compute_features(series.with_values(future), horizon_periods=1, spec=SPEC)
        assert mutated.row(cutoff).tobytes() == features.row(cutoff).tobytes()
        assert mutated.row(cutoff + 1).tobytes() != features.row(cutoff + 1).tobytes()  # next row sees one more


def test_series_too_short_for_the_features_is_rejected() -> None:
    short = DemandSeries("s", START, HOUR, 3, np.ones(100))
    with pytest.raises(InvalidDataError, match="too short"):
        compute_features(short, horizon_periods=1, spec=SPEC)


def test_transform_version_identifies_the_transform() -> None:
    series = _series()
    base = compute_features(series, horizon_periods=1, spec=SPEC).transform_version
    assert base.startswith("demand.features/1:")
    assert base == compute_features(series, horizon_periods=1, spec=SPEC).transform_version
    assert base != compute_features(series, horizon_periods=2, spec=SPEC).transform_version
    assert base != compute_features(series, horizon_periods=1, spec=FeatureSpec(lags=(24, 168))).transform_version


def test_queries_freeze_the_issue_time_snapshot() -> None:
    series = _series()
    features = compute_features(series, horizon_periods=1, spec=SPEC)
    target = demand_target(series, horizon_periods=1)
    query = make_query(series, features, 500, target=target, query_id="q500", quantile_levels=(0.5,))
    assert query.issued_at_ns == query.feature_cutoff_ns == series.cutoff_ns(500)
    assert query.features.tobytes() == features.row(500).tobytes()
    assert query.feature_names == features.feature_names and query.transform_version == features.transform_version
    assert target.horizon_ns == HOUR and target.availability.min_delay_ns == 3 * HOUR
    assert target.domain == "demand"
    assert target.earliest_available_ns(series.cutoff_ns(500)) == series.available_at_ns(500)
    assert {"venue", "usd", "btc"}.isdisjoint(" ".join(map(str, target.to_mapping().values())).lower().split())


def test_mature_batches_hold_only_labels_released_by_the_fit_cutoff() -> None:
    series = _series()
    features = compute_features(series, horizon_periods=1, spec=SPEC)
    target = demand_target(series, horizon_periods=1)
    batch = mature_batch(series, features, as_of_cutoff=400, target=target)
    assert batch.as_of_ns == series.cutoff_ns(400)
    assert int(batch.available_at_ns.max()) == series.cutoff_ns(400)  # the newest label is released exactly now
    assert len(batch) == 400 - 1 - 3 - features.first_cutoff + 1
    first = features.first_cutoff
    assert batch.x[0].tobytes() == features.row(first).tobytes() and batch.y[0] == series.values[first]
    assert len(mature_batch(series, features, as_of_cutoff=401, target=target)) == len(batch) + 1
    with pytest.raises(InvalidDataError, match="no matured"):
        mature_batch(series, features, as_of_cutoff=first + 3, target=target)
