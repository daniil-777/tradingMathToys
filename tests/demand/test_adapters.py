"""Demand baseline adapters: contract conformance, exact formulas, direct/wrapped ridge parity, state round trips."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from financeai.core.errors import (
    IncompatibleSchemaError,
    InvalidConfigError,
    InvalidDataError,
    UnsupportedCapabilityError,
)
from financeai.core.forecast import Capability, ForecastAdapter, ForecastQuery, MatureBatch, Outcome
from financeai.core.serialization import canonical_json, loads_strict
from financeai.domains.demand import (
    ADAPTER_KINDS,
    DemandFeatures,
    DemandSeries,
    DemandSpec,
    FeatureSpec,
    RidgeLagAdapter,
    SeasonalEWMAAdapter,
    SeasonalNaiveAdapter,
    build_adapter,
    compute_features,
    demand_target,
    generate_demand,
    make_query,
    mature_batch,
)
from financeai.models.ridge import RidgeModel

LEVELS = (0.1, 0.5, 0.9)
FIT_CUTOFF = 600


@pytest.fixture(scope="module")
def setup() -> tuple[DemandSeries, DemandFeatures, MatureBatch]:
    series = generate_demand(DemandSpec(streams=("s",), n_periods=24 * 7 * 6, seed=3))[0]
    features = compute_features(series, horizon_periods=1, spec=FeatureSpec())
    batch = mature_batch(series, features, as_of_cutoff=FIT_CUTOFF, target=demand_target(series, horizon_periods=1))
    return series, features, batch


def _query(
    setup: tuple[DemandSeries, DemandFeatures, MatureBatch], cutoff: int, levels: tuple[float, ...] = ()
) -> ForecastQuery:
    series, features, _ = setup
    return make_query(
        series,
        features,
        cutoff,
        target=demand_target(series, horizon_periods=1),
        query_id=f"q{cutoff}",
        quantile_levels=levels,
    )


def _outcome(setup: tuple[DemandSeries, DemandFeatures, MatureBatch], cutoff: int) -> Outcome:
    series = setup[0]
    return Outcome(f"q{cutoff}", float(series.values[cutoff]), series.available_at_ns(cutoff))


EXPECTED_CAPABILITIES = {
    "seasonal_naive": {"mean", "marginal_quantiles", "online_update", "delayed_feedback"},
    "seasonal_ewma": {"mean", "variance", "marginal_quantiles", "online_update", "delayed_feedback"},
    "ridge_lags": {"mean", "online_update", "delayed_feedback"},
}


@pytest.mark.parametrize("kind", sorted(ADAPTER_KINDS))
def test_adapters_conform_to_the_generic_protocol(kind: str) -> None:
    adapter = build_adapter(kind)
    assert isinstance(adapter, ForecastAdapter)
    assert adapter.model_id == kind
    assert {c.value for c in adapter.capabilities} == EXPECTED_CAPABILITIES[kind]
    assert adapter.model_version == "f0.u0"


@pytest.mark.parametrize("kind", sorted(ADAPTER_KINDS))
def test_unfitted_adapters_refuse_to_predict_or_update(kind: str, setup: Any) -> None:
    adapter = build_adapter(kind)
    with pytest.raises(UnsupportedCapabilityError, match="not fitted"):
        adapter.predict(_query(setup, FIT_CUTOFF))
    fitted = build_adapter(kind)
    fitted.fit(setup[2])
    ticket = fitted.predict(_query(setup, FIT_CUTOFF))
    with pytest.raises(UnsupportedCapabilityError, match="not fitted"):
        adapter.update(ticket, _outcome(setup, FIT_CUTOFF))


def test_seasonal_naive_issues_the_seasonal_lag_and_residual_quantiles(setup: Any) -> None:
    _, features, batch = setup
    adapter = SeasonalNaiveAdapter(season_lag=168, residual_window=200, min_residuals=24)
    adapter.fit(batch)
    col = features.feature_names.index("lag_168")
    residuals = (batch.y - batch.x[:, col])[-200:]
    query = _query(setup, FIT_CUTOFF, LEVELS)
    ticket = adapter.predict(query)
    assert ticket.mean == query.features[col]
    assert ticket.quantiles is not None
    assert ticket.quantiles.tobytes() == (ticket.mean + np.quantile(residuals, LEVELS)).tobytes()
    assert ticket.capabilities == frozenset({Capability.MEAN, Capability.MARGINAL_QUANTILES})
    adapter.update(ticket, _outcome(setup, FIT_CUTOFF))
    state = adapter.get_state()
    assert state["model"]["residuals"][-1] == _outcome(setup, FIT_CUTOFF).value - ticket.mean  # the issued mean
    assert len(state["model"]["residuals"]) == 200 and adapter.model_version == "f1.u1"


def test_seasonal_ewma_follows_the_per_phase_recursion(setup: Any) -> None:
    _, features, batch = setup
    alpha, beta = 0.3, 0.1
    adapter = SeasonalEWMAAdapter(alpha=alpha, variance_alpha=beta, n_phases=168, residual_window=500)
    adapter.fit(batch)
    phase_col = features.feature_names.index("phase")
    levels: list[float | None] = [None] * 168
    variance: float | None = None
    for p, y in zip(batch.x[:, phase_col].astype(int).tolist(), batch.y.tolist(), strict=True):
        level = levels[p]
        if level is None:
            levels[p] = y
            continue
        r = y - level
        variance = r * r if variance is None else (1 - beta) * variance + beta * r * r
        levels[p] = level + alpha * (y - level)
    state = adapter.get_state()["model"]
    assert state["levels"] == levels and state["variance"] == variance
    query = _query(setup, FIT_CUTOFF, LEVELS)
    ticket = adapter.predict(query)
    p = int(query.features[phase_col])
    assert ticket.mean == levels[p] and ticket.variance == variance
    assert ticket.capabilities == frozenset({Capability.MEAN, Capability.VARIANCE, Capability.MARGINAL_QUANTILES})
    outcome = _outcome(setup, FIT_CUTOFF)
    adapter.update(ticket, outcome)
    after = adapter.get_state()["model"]
    assert after["levels"][p] == levels[p] + alpha * (outcome.value - levels[p])  # type: ignore[operator]
    r = outcome.value - ticket.mean
    assert after["variance"] == (1 - beta) * variance + beta * r * r  # type: ignore[operator]


def test_seasonal_ewma_needs_every_phase_in_training(setup: Any) -> None:
    series, features, _ = setup
    small = mature_batch(
        series, features, as_of_cutoff=features.first_cutoff + 100, target=demand_target(series, horizon_periods=1)
    )
    with pytest.raises(InvalidDataError, match="phase"):
        SeasonalEWMAAdapter().fit(small)
    with pytest.raises(InvalidDataError, match="residual"):
        SeasonalNaiveAdapter(min_residuals=200, residual_window=300).fit(small)


def test_ridge_adapter_equals_a_direct_ridge_fit_and_refits_on_matured_feedback(setup: Any) -> None:
    series, features, batch = setup
    names = ("lag_4", "lag_24", "lag_168", "mean_24", "hod_sin", "hod_cos")
    adapter = RidgeLagAdapter(alpha=2.0, features=names, window=300, refit_every=5)
    adapter.fit(batch)
    cols = [features.feature_names.index(n) for n in names]
    direct = RidgeModel(alpha=2.0).fit(batch.x[-300:, cols], batch.y[-300:])
    query = _query(setup, FIT_CUTOFF)
    assert adapter.predict(query).mean == float(direct.predict(query.features[cols][None, :])[0])
    with pytest.raises(UnsupportedCapabilityError):
        adapter.predict(_query(setup, FIT_CUTOFF, LEVELS))
    coef = adapter.get_state()["model"]["coef"]
    rows_x, rows_y = [list(r) for r in batch.x[-300:, cols]], list(batch.y[-300:])
    for k in range(5):
        cutoff = FIT_CUTOFF + k
        ticket = adapter.predict(_query(setup, cutoff))
        adapter.update(ticket, _outcome(setup, cutoff))
        rows_x.append(list(ticket.features[cols]))
        rows_y.append(float(series.values[cutoff]))
        if k < 4:
            assert adapter.get_state()["model"]["coef"] == coef  # no refit before refit_every updates
    refit = RidgeModel(alpha=2.0).fit(np.array(rows_x[-300:]), np.array(rows_y[-300:]))
    assert refit.coef_ is not None
    assert adapter.get_state()["model"]["coef"] == refit.coef_.tolist()
    assert adapter.model_version == "f2.u5"


def test_default_ridge_features_exclude_the_phase_index(setup: Any) -> None:
    adapter = RidgeLagAdapter()
    adapter.fit(setup[2])
    assert "phase" not in adapter.get_state()["model"]["features"]
    assert len(adapter.get_state()["model"]["coef"]) == len(setup[1].feature_names) - 1


@pytest.mark.parametrize("kind", sorted(ADAPTER_KINDS))
def test_queries_from_another_transform_are_rejected(kind: str, setup: Any) -> None:
    series = setup[0]
    adapter = build_adapter(kind)
    adapter.fit(setup[2])
    other = compute_features(series, horizon_periods=1, spec=FeatureSpec(lags=(24, 168), windows=(24,)))
    query = make_query(series, other, FIT_CUTOFF, target=demand_target(series, horizon_periods=1), query_id="x")
    with pytest.raises(InvalidDataError, match="transform"):
        adapter.predict(query)
    ticket = adapter.predict(_query(setup, FIT_CUTOFF))
    with pytest.raises(InvalidDataError):
        adapter.update(ticket, Outcome("someone-else", 1.0, series.available_at_ns(FIT_CUTOFF)))


@pytest.mark.parametrize("kind", sorted(ADAPTER_KINDS))
def test_state_round_trip_reproduces_predictions_exactly(kind: str, setup: Any) -> None:
    adapter = build_adapter(kind)
    adapter.fit(setup[2])
    levels = LEVELS if Capability.MARGINAL_QUANTILES in adapter.capabilities else ()
    for cutoff in range(FIT_CUTOFF, FIT_CUTOFF + 30):
        adapter.update(adapter.predict(_query(setup, cutoff, levels)), _outcome(setup, cutoff))
    text = canonical_json(adapter.get_state())
    restored = build_adapter(kind)
    restored.set_state(loads_strict(text))
    assert canonical_json(restored.get_state()) == text
    for cutoff in range(FIT_CUTOFF + 30, FIT_CUTOFF + 40):
        assert restored.predict(_query(setup, cutoff, levels)) == adapter.predict(_query(setup, cutoff, levels))
    other_params: dict[str, dict[str, Any]] = {
        "seasonal_naive": {"season_lag": 24},
        "seasonal_ewma": {"alpha": 0.5},
        "ridge_lags": {"alpha": 3.0},
    }
    with pytest.raises(IncompatibleSchemaError):
        build_adapter(kind, other_params[kind]).set_state(loads_strict(text))
    corrupt = loads_strict(text)
    corrupt["fits"] = -1
    with pytest.raises(InvalidDataError):
        build_adapter(kind).set_state(corrupt)


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("prophet", {}),
        ("seasonal_naive", {"season_lag": 0}),
        ("seasonal_naive", {"unknown": 1}),
        ("seasonal_naive", {"min_residuals": 500, "residual_window": 100}),
        ("seasonal_ewma", {"alpha": 0.0}),
        ("seasonal_ewma", {"alpha": 1.5}),
        ("seasonal_ewma", {"variance_alpha": float("nan")}),
        ("seasonal_ewma", {"n_phases": 0}),
        ("ridge_lags", {"alpha": -1.0}),
        ("ridge_lags", {"window": 1}),
        ("ridge_lags", {"refit_every": 0}),
        ("ridge_lags", {"features": []}),
        ("ridge_lags", {"features": ["lag_24", "lag_24"]}),
        ("ridge_lags", {"alpha": True}),
    ],
)
def test_build_adapter_validates_kind_and_parameters(kind: str, params: dict[str, Any]) -> None:
    with pytest.raises(InvalidConfigError):
        build_adapter(kind, params)


def test_ridge_rejects_features_the_transform_does_not_provide(setup: Any) -> None:
    with pytest.raises(InvalidConfigError, match="feature"):
        RidgeLagAdapter(features=("lag_999",)).fit(setup[2])
    with pytest.raises(InvalidConfigError, match="feature"):
        SeasonalNaiveAdapter(season_lag=336).fit(setup[2])
