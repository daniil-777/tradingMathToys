"""Demand domain protocol (strict JSON, own validator, no market fields) and demand metrics."""

from __future__ import annotations

import itertools
import json
import math
from typing import Any

import pytest

from financeai.core.errors import IncompatibleSchemaError, InvalidConfigError, InvalidDataError
from financeai.domains.demand import (
    PROTOCOL_SCHEMA,
    AdapterSpec,
    ChronologicalSplit,
    DemandMetrics,
    DemandProtocol,
    DemandSpec,
    FeatureSpec,
    default_protocol,
    summarize,
)
from financeai.online import ScoreRecord

HOUR = 3_600 * 10**9


def _protocol(**overrides: Any) -> DemandProtocol:
    fields: dict[str, Any] = {
        "name": "p",
        "data": DemandSpec(streams=("a",), n_periods=24 * 7 * 6),
        "adapters": (AdapterSpec("naive", "seasonal_naive"), AdapterSpec("ridge", "ridge_lags", {"alpha": 2.0})),
        "train_periods": 24 * 7 * 3,
        "n_folds": 2,
    }
    fields.update(overrides)
    return DemandProtocol(**fields)


def test_protocol_round_trips_through_strict_json() -> None:
    protocol = _protocol()
    text = protocol.to_json()
    doc = json.loads(text)
    assert doc["schema"] == PROTOCOL_SCHEMA and doc["schema_version"] == "1.0"
    restored = DemandProtocol.from_json(text)
    assert restored == protocol and restored.identity == protocol.identity
    assert restored.adapters[1].params == {"alpha": 2.0}
    assert _protocol(n_folds=1).identity != protocol.identity
    assert default_protocol().to_json() == default_protocol().to_json()


def test_protocol_is_a_demand_protocol_without_market_fields() -> None:
    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {k for v in value.values() for k in keys(v)}
        if isinstance(value, list):
            return {k for v in value for k in keys(v)}
        return set()

    found = keys(_protocol().to_mapping())
    assert found.isdisjoint(
        {
            "fees",
            "scenario",
            "scenarios",
            "instrument",
            "instruments",
            "venue",
            "size_units",
            "net_pnl",
            "timing",
            "latency_ms",
        }
    )
    assert _protocol().target().domain == "demand"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(extra=1),
        lambda d: d["data"].update(fees={"x": "0.1"}),
        lambda d: d["adapters"].append({"name": "naive", "kind": "seasonal_naive", "params": {}}),  # duplicate name
        lambda d: d["adapters"].append({"name": "x", "kind": "inverse_perpetual", "params": {}}),
        lambda d: d["adapters"][0]["params"].update(season_lag=-1),
        lambda d: d.update(adapters=[]),
        lambda d: d.update(horizon_periods=0),
        lambda d: d.update(n_folds=0),
        lambda d: d.update(n_folds=10_000),  # more folds than test cutoffs
        lambda d: d.update(train_periods=100),  # no complete matured training row yet
        lambda d: d.update(train_periods=24 * 7 * 6),  # no test cutoff left
        lambda d: d.update(quantile_levels=[0.9, 0.1]),
        lambda d: d.update(online_update="yes"),
        lambda d: d["features"].update(lags=[2]),  # lag 2 is not released at a horizon-1 cutoff with delay 3
    ],
)
def test_protocol_validation_rejects_invalid_documents(mutate: Any) -> None:
    doc = json.loads(_protocol().to_json())
    mutate(doc)
    with pytest.raises(InvalidConfigError):
        DemandProtocol.from_json(json.dumps(doc))


def test_protocol_rejects_unknown_schemas_duplicate_keys_and_non_finite_numbers() -> None:
    doc = json.loads(_protocol().to_json())
    for key, value in (("schema", "financeai.strategy"), ("schema_version", "2.0")):
        with pytest.raises(IncompatibleSchemaError):
            DemandProtocol.from_json(json.dumps({**doc, key: value}))
    text = json.dumps(doc)
    with pytest.raises(InvalidConfigError):
        DemandProtocol.from_json(text[:-1] + ', "name": "again"}')
    with pytest.raises(InvalidConfigError):
        DemandProtocol.from_json(text.replace('"zero_probability": 0.2', '"zero_probability": NaN'))


def test_chronological_splits_cover_the_test_region_without_overlap() -> None:
    protocol = _protocol(n_folds=3)
    splits = protocol.splits()
    assert [s.fold for s in splits] == [0, 1, 2]
    assert splits[0].fit_cutoff == protocol.train_periods
    assert splits[-1].stop_cutoff == protocol.data.n_periods - protocol.horizon_periods + 1
    for a, b in itertools.pairwise(splits):
        assert a.stop_cutoff == b.fit_cutoff and a.fit_cutoff < a.stop_cutoff
    assert protocol.splits(n_periods=24 * 7 * 5)[-1].stop_cutoff == 24 * 7 * 5
    with pytest.raises(InvalidConfigError):
        ChronologicalSplit(fold=0, fit_cutoff=10, stop_cutoff=10)
    with pytest.raises(InvalidConfigError):
        protocol.splits(n_periods=protocol.train_periods)


def _record(i: int, value: float, mean: float, pinball: tuple[float, ...] = ()) -> ScoreRecord:
    error = value - mean
    levels = (0.1, 0.5, 0.9) if pinball else ()
    return ScoreRecord(
        ticket_id=f"t{i}",
        stream="s",
        outcome_version=0,
        issued_at_ns=i * HOUR,
        available_at_ns=(i + 4) * HOUR,
        released_at_ns=(i + 4) * HOUR,
        value=value,
        mean=mean,
        error=error,
        abs_error=abs(error),
        squared_error=error * error,
        quantile_levels=levels,
        pinball=pinball,
    )


def test_metrics_are_mae_rmse_bias_and_pinball_over_scored_tickets() -> None:
    records = [_record(0, 3.0, 1.0, (0.2, 1.0, 0.1)), _record(1, 0.0, 1.0, (0.1, 0.5, 0.9)), _record(2, 5.0, 5.0)]
    metrics = summarize(records, n_issued=4)
    assert metrics.n_issued == 4 and metrics.n_scored == 3 and metrics.n_unscored == 1
    assert metrics.mae == pytest.approx((2 + 1 + 0) / 3)
    assert metrics.rmse == pytest.approx(math.sqrt((4 + 1 + 0) / 3))
    assert metrics.bias == pytest.approx((2 - 1 + 0) / 3)
    assert metrics.pinball == pytest.approx((0.2 + 1.0 + 0.1 + 0.1 + 0.5 + 0.9) / 6)
    assert metrics.n_quantile_scored == 2
    point_only = summarize([_record(0, 3.0, 1.0)], n_issued=1)
    assert point_only.pinball is None and point_only.n_quantile_scored == 0
    json.dumps(metrics.to_mapping(), allow_nan=False)


def test_metrics_never_fill_missing_scores_with_zero() -> None:
    empty = summarize([], n_issued=5)
    assert empty == DemandMetrics(
        n_issued=5, n_scored=0, mae=None, rmse=None, bias=None, pinball=None, n_quantile_scored=0
    )
    with pytest.raises(InvalidDataError):
        summarize([_record(0, 1.0, 1.0), _record(1, 1.0, 1.0)], n_issued=1)
    with pytest.raises(InvalidDataError):
        summarize([_record(0, 1.0, 1.0), _record(0, 2.0, 1.0)], n_issued=3)  # a ticket scored twice


def test_adapter_spec_validates_eagerly() -> None:
    with pytest.raises(InvalidConfigError):
        AdapterSpec("x", "seasonal_naive", {"season_lag": "168"})
    with pytest.raises(InvalidConfigError):
        AdapterSpec("", "seasonal_naive")
    spec = AdapterSpec("ewma", "seasonal_ewma", {"alpha": 0.3})
    assert spec.build().model_id == "seasonal_ewma"
    with pytest.raises(TypeError):
        spec.params["alpha"] = 0.4  # type: ignore[index]
    assert FeatureSpec().names(1, 3)[0] == "lag_4"
