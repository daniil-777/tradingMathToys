"""End-to-end demand evaluation: causal maturity, future mutation, exact resume, chronological reports, boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from financeai.core.errors import IncompatibleSchemaError, InvalidDataError
from financeai.core.forecast import Capability, ForecastAdapter, ForecastQuery, ForecastTicket, MatureBatch, Outcome
from financeai.core.serialization import canonical_json
from financeai.domains.demand import (
    ADAPTER_KINDS,
    AdapterSpec,
    ChronologicalSplit,
    DemandFeatures,
    DemandProtocol,
    DemandRun,
    DemandSeries,
    DemandSpec,
    FeatureSpec,
    RunResult,
    build_adapter,
    compute_features,
    default_protocol,
    evaluate,
    generate_demand,
)
from financeai.online import DelayedFeedbackLearner

FIT, STOP = 600, 700
SPLIT = ChronologicalSplit(fold=0, fit_cutoff=FIT, stop_cutoff=STOP)


@pytest.fixture(scope="module")
def series() -> DemandSeries:
    return generate_demand(DemandSpec(streams=("s",), n_periods=24 * 7 * 6, seed=21))[0]


def _features(s: DemandSeries) -> DemandFeatures:
    return compute_features(s, horizon_periods=1, spec=FeatureSpec())


def _run(
    s: DemandSeries,
    adapter: ForecastAdapter,
    *,
    levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    split: ChronologicalSplit = SPLIT,
    online_update: bool | None = None,
    unit: str = "units",
) -> DemandRun:
    return DemandRun(
        s,
        _features(s),
        adapter,
        name=adapter.model_id,
        split=split,
        quantile_levels=levels,
        online_update=online_update,
        unit=unit,
    )


def _complete(run: DemandRun) -> RunResult:
    run.start()
    run.run()
    return run.finish()


class Spy:
    """Delegating adapter that logs the clock of every prediction and the availability of every update."""

    def __init__(self, inner: ForecastAdapter) -> None:
        self.inner = inner
        self.log: list[tuple[str, int]] = []

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    @property
    def model_version(self) -> str:
        return self.inner.model_version

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self.inner.capabilities

    def fit(self, batch: MatureBatch) -> None:
        self.log.append(("fit", int(batch.available_at_ns.max())))
        self.inner.fit(batch)

    def predict(self, query: ForecastQuery) -> ForecastTicket:
        self.log.append(("predict", query.issued_at_ns))
        return self.inner.predict(query)

    def update(self, ticket: ForecastTicket, outcome: Outcome) -> None:
        self.log.append(("update", outcome.available_at_ns))
        self.inner.update(ticket, outcome)

    def get_state(self) -> dict[str, Any]:
        return self.inner.get_state()

    def set_state(self, state: Mapping[str, Any]) -> None:
        self.inner.set_state(state)


@pytest.mark.parametrize("kind", sorted(ADAPTER_KINDS))
def test_adapters_receive_labels_only_after_release_and_before_the_next_forecast(
    kind: str, series: DemandSeries
) -> None:
    spy = Spy(build_adapter(kind))
    result = _complete(_run(series, spy))
    log = spy.log
    assert log[0] == ("fit", series.cutoff_ns(FIT))  # the newest training label is released exactly at the fit
    predictions = [t for event, t in log if event == "predict"]
    assert predictions == [series.cutoff_ns(c) for c in range(FIT, STOP)]
    for k, (event, available) in enumerate(log):
        if event != "update":
            continue
        assert all(t < available for e, t in log[:k] if e == "predict")  # never trained before availability
        assert all(t >= available for e, t in log[k + 1 :] if e == "predict")  # delivered before the next forecast
    assert sum(1 for e, _ in log if e == "update") == len(result.tickets) == STOP - FIT


@pytest.mark.parametrize("kind", sorted(ADAPTER_KINDS))
def test_each_ticket_is_scored_once_against_its_reported_value(kind: str, series: DemandSeries) -> None:
    result = _complete(_run(series, build_adapter(kind)))
    assert [t.ticket_id for t in result.tickets] == [f"{kind}/s/f0/c{c}" for c in range(FIT, STOP)]
    assert sorted(s.ticket_id for s in result.scores) == sorted(t.ticket_id for t in result.tickets)
    for ticket, score in zip(result.tickets, sorted(result.scores, key=lambda s: s.issued_at_ns), strict=True):
        cutoff = (ticket.issued_at_ns - series.start_ns) // series.period_ns
        assert score.value == series.values[cutoff] and score.mean == ticket.mean
        assert score.released_at_ns >= score.available_at_ns == series.available_at_ns(cutoff)
    assert result.metrics.n_scored == result.metrics.n_issued == STOP - FIT
    expects_pinball = Capability.MARGINAL_QUANTILES in ADAPTER_KINDS[kind].CAPABILITIES
    assert (result.metrics.pinball is not None) == expects_pinball


@pytest.mark.parametrize("kind", sorted(ADAPTER_KINDS))
def test_mutating_unreleased_values_never_changes_earlier_tickets(kind: str, series: DemandSeries) -> None:
    reference = _complete(_run(series, build_adapter(kind)))
    horizon_cutoff = FIT + 40
    released = series.visible_count(horizon_cutoff)
    future = series.values.copy()
    future[released:] = future[released:] * 3.0 + 1.0
    mutated = _complete(_run(series.with_values(future), build_adapter(kind)))
    clock = series.cutoff_ns(horizon_cutoff)
    early = [(a, b) for a, b in zip(reference.tickets, mutated.tickets, strict=True) if a.issued_at_ns <= clock]
    assert len(early) == horizon_cutoff - FIT + 1
    assert all(a == b for a, b in early)  # identical tickets, including model versions and quantiles
    assert any(a != b for a, b in zip(reference.tickets, mutated.tickets, strict=True) if a.issued_at_ns > clock)


@pytest.mark.parametrize("kind", sorted(ADAPTER_KINDS))
@pytest.mark.parametrize("cut", [FIT, FIT + 1, FIT + 37, STOP - 1, STOP])
def test_interrupted_and_resumed_run_equals_the_uninterrupted_run(kind: str, cut: int, series: DemandSeries) -> None:
    uninterrupted = _run(series, build_adapter(kind))
    reference = _complete(uninterrupted)
    first = _run(series, build_adapter(kind))
    first.start()
    first.run(until=cut)
    checkpoint = first.to_json()
    resumed = _run(series, build_adapter(kind))  # a fresh process would rebuild exactly this
    resumed.load_json(checkpoint)
    assert resumed.cursor == cut and resumed.to_json() == checkpoint
    resumed.run()
    result = resumed.finish()
    assert resumed.to_json() == uninterrupted.to_json()
    assert result.tickets == reference.tickets and result.scores == reference.scores
    assert result.metrics == reference.metrics


def test_checkpoints_only_resume_the_run_they_belong_to(series: DemandSeries) -> None:
    run = _run(series, build_adapter("seasonal_ewma"))
    run.start()
    run.run(until=FIT + 5)
    checkpoint = run.to_json()
    other_series = series.with_values(series.values + 1.0)
    for other in (
        _run(other_series, build_adapter("seasonal_ewma")),
        _run(series, build_adapter("seasonal_ewma"), split=ChronologicalSplit(0, FIT, STOP + 1)),
        _run(series, build_adapter("seasonal_ewma"), levels=(0.5,)),
    ):
        with pytest.raises(IncompatibleSchemaError):
            other.load_json(checkpoint)
    with pytest.raises(IncompatibleSchemaError):
        _run(series, build_adapter("seasonal_ewma", {"alpha": 0.5})).load_json(checkpoint)
    started = _run(series, build_adapter("seasonal_ewma"))
    started.start()
    with pytest.raises(InvalidDataError):
        started.load_json(checkpoint)
    with pytest.raises(InvalidDataError):
        started.finish()  # unissued cutoffs remain


CUT = FIT + 10  # tickets at cutoffs 600..609; outcomes of 600..605 released (horizon 1 + delay 3), 606..609 pending
Edit = Callable[[dict[str, Any]], None]


@pytest.fixture(scope="module")
def ewma_reference(series: DemandSeries) -> RunResult:
    return _complete(_run(series, build_adapter("seasonal_ewma")))


def _checkpoint(series: DemandSeries, *, cut: int = CUT) -> dict[str, Any]:
    run = _run(series, build_adapter("seasonal_ewma"))
    run.start()
    run.run(until=cut)
    doc: dict[str, Any] = json.loads(run.to_json())
    return doc


def _book(doc: dict[str, Any]) -> dict[str, Any]:
    book: dict[str, Any] = doc["learner"]["book"]
    return book


def _cursor(value: object) -> Edit:
    return lambda doc: doc.update(cursor=value)


def _newest(kind: str, field: str, change: Callable[[Any], Any]) -> Edit:
    """Edit ``field`` of the newest ticket or outcome (still pending at ``CUT``, so no score pins it)."""

    def edit(doc: dict[str, Any]) -> None:
        item = _book(doc)[kind][-1]
        item[field] = change(item[field])

    return edit


def _swap_first_tickets(doc: dict[str, Any]) -> None:
    tickets = _book(doc)["tickets"]
    tickets[0], tickets[1] = tickets[1], tickets[0]


def _rename_tickets(doc: dict[str, Any]) -> None:
    book = _book(doc)
    for item in (*book["tickets"], *book["outcomes"], *book["scores"]):
        prefix, cutoff = item["ticket_id"].rsplit("/c", 1)
        item["ticket_id"] = f"{prefix}/c{int(cutoff) + 1000}"


def _nudge_clock(doc: dict[str, Any]) -> None:
    book = _book(doc)
    book["clock_ns"] = str(int(book["clock_ns"]) + 1)  # before the next pending availability: nothing new released


def _freeze_learner(doc: dict[str, Any]) -> None:
    doc["learner"].update(updating=False, n_updates=0)  # a frozen learner has applied no update


TAMPERS: dict[str, tuple[Edit, str]] = {
    "cursor_at_stop": (_cursor(STOP), "cutoffs before its cursor"),
    "cursor_ahead": (_cursor(CUT + 40), "cutoffs before its cursor"),
    "cursor_behind": (_cursor(CUT - 1), "cutoffs before its cursor"),
    "tickets_reordered": (_swap_first_tickets, "cutoffs before its cursor"),
    "tickets_of_other_cutoffs": (_rename_tickets, "cutoffs before its cursor"),
    "ticket_target_unit": (_newest("tickets", "target", lambda target: {**target, "unit": "kWh"}), "does not answer"),
    "ticket_features": (
        _newest("tickets", "features", lambda xs: [*xs[:-1], float((xs[-1] + 1) % 168)]),
        "does not answer",
    ),
    "outcome_value": (_newest("outcomes", "value", lambda value: value + 1.0), "outcomes differ"),
    "outcome_available_at": (_newest("outcomes", "available_at_ns", lambda ns: str(int(ns) + 1)), "outcomes differ"),
    "book_clock": (_nudge_clock, "clock disagrees"),
    "update_mode": (_freeze_learner, "update mode"),
}


@pytest.mark.parametrize("tamper", sorted(TAMPERS))
def test_resume_rejects_a_checkpoint_this_run_could_not_have_written(
    tamper: str, series: DemandSeries, ewma_reference: RunResult
) -> None:
    edit, message = TAMPERS[tamper]
    doc = _checkpoint(series)
    edit(doc)
    DelayedFeedbackLearner.from_state(doc["learner"], build_adapter("seasonal_ewma"))  # consistent on its own
    adapter = build_adapter("seasonal_ewma")
    run = _run(series, adapter)
    with pytest.raises(InvalidDataError, match=message):
        run.load_json(canonical_json(doc))
    assert run.cursor is None and adapter.model_version == "f0.u0"  # neither the run nor its adapter was restored
    result = _complete(run)  # the rejected checkpoint leaves a fresh run that still completes exactly
    assert result.tickets == ewma_reference.tickets and result.metrics == ewma_reference.metrics


def test_resume_rejects_a_release_before_the_first_issue(series: DemandSeries) -> None:
    doc = _checkpoint(series, cut=FIT)
    assert _book(doc)["tickets"] == [] and _book(doc)["clock_ns"] is None
    _book(doc)["clock_ns"] = str(series.cutoff_ns(FIT))
    DelayedFeedbackLearner.from_state(doc["learner"], build_adapter("seasonal_ewma"))  # consistent on its own
    with pytest.raises(InvalidDataError, match="clock disagrees"):
        _run(series, build_adapter("seasonal_ewma")).load_json(canonical_json(doc))


@pytest.mark.parametrize("cursor", [FIT - 1, STOP + 1, str(CUT), float(CUT), True])
def test_resume_rejects_a_cursor_outside_the_split(cursor: object, series: DemandSeries) -> None:
    doc = _checkpoint(series)
    doc["cursor"] = cursor
    with pytest.raises(InvalidDataError, match="outside the split"):
        _run(series, build_adapter("seasonal_ewma")).load_json(canonical_json(doc))


def test_checkpoints_pin_the_target_identity(series: DemandSeries) -> None:
    doc = _checkpoint(series)
    assert doc["target"]["unit"] == "units"
    with pytest.raises(IncompatibleSchemaError) as info:
        _run(series, build_adapter("seasonal_ewma"), unit="kWh").load_json(canonical_json(doc))
    assert info.value.context["fields"] == ["target"]


def test_a_finished_run_resumes_to_the_same_result(series: DemandSeries, ewma_reference: RunResult) -> None:
    finished = _run(series, build_adapter("seasonal_ewma"))
    _complete(finished)
    resumed = _run(series, build_adapter("seasonal_ewma"))
    resumed.load_json(finished.to_json())
    assert resumed.done and resumed.finish() == ewma_reference
    assert resumed.to_json() == finished.to_json()


def test_point_only_adapters_are_never_asked_for_quantiles(series: DemandSeries) -> None:
    result = _complete(_run(series, build_adapter("ridge_lags")))
    assert all(t.capabilities == frozenset({Capability.MEAN}) for t in result.tickets)
    assert result.metrics.pinball is None and result.metrics.n_quantile_scored == 0


def test_frozen_evaluation_scores_without_updating(series: DemandSeries) -> None:
    adapter = build_adapter("seasonal_ewma")
    result = _complete(_run(series, adapter, online_update=False))
    assert adapter.model_version == "f1.u0"
    assert len(result.scores) == STOP - FIT


def _small_protocol(**overrides: Any) -> DemandProtocol:
    fields: dict[str, Any] = {
        "name": "small",
        "data": DemandSpec(streams=("a", "b"), n_periods=24 * 7 * 6, seed=4),
        "adapters": (
            AdapterSpec("naive", "seasonal_naive"),
            AdapterSpec("ewma", "seasonal_ewma"),
            AdapterSpec("ridge", "ridge_lags"),
        ),
        "train_periods": 24 * 7 * 3,
        "n_folds": 2,
    }
    fields.update(overrides)
    return DemandProtocol(**fields)


def test_evaluate_reports_every_adapter_stream_and_fold() -> None:
    protocol = _small_protocol()
    report = evaluate(protocol)
    assert [(r.fold, r.adapter, r.stream) for r in report.runs] == [
        (f, a, s) for f in (0, 1) for a in ("naive", "ewma", "ridge") for s in ("a", "b")
    ]
    pooled = report.pooled()
    assert list(pooled) == ["naive", "ewma", "ridge"]
    total = 2 * (24 * 7 * 6 - 1 + 1 - 24 * 7 * 3)  # two streams x every test cutoff
    for name, metrics in pooled.items():
        assert metrics.n_issued == metrics.n_scored == total
        assert metrics.mae is not None and metrics.rmse is not None and metrics.rmse >= metrics.mae > 0
        assert (metrics.pinball is None) == (name == "ridge")
    doc = report.to_mapping()
    assert json.loads(canonical_json(doc)) == doc
    assert doc["protocol_id"] == protocol.identity
    assert evaluate(protocol).to_mapping() == doc  # deterministic end to end


def test_online_updates_track_a_level_shift_that_a_frozen_fit_misses() -> None:
    data = DemandSpec(streams=("a",), n_periods=24 * 7 * 10, shift_period=24 * 7 * 6, shift_factor=2.5, seed=8)
    adapters = (AdapterSpec("ewma", "seasonal_ewma", {"alpha": 0.5}), AdapterSpec("ridge", "ridge_lags"))
    common: dict[str, Any] = {"data": data, "adapters": adapters, "train_periods": 24 * 7 * 5, "n_folds": 1}
    adaptive = evaluate(_small_protocol(**common)).pooled()
    frozen = evaluate(_small_protocol(**common, online_update=False)).pooled()
    for name in ("ewma", "ridge"):
        a, f = adaptive[name], frozen[name]
        assert a.mae is not None and f.mae is not None and a.bias is not None and f.bias is not None
        assert a.mae < 0.9 * f.mae
        assert f.bias > a.bias > 0  # the frozen fit keeps under-forecasting the higher post-shift level


def test_evaluate_accepts_external_series_without_a_generator() -> None:
    protocol = _small_protocol(n_folds=1)
    rng = np.random.default_rng(1)
    hours = np.arange(24 * 7 * 6)
    values = rng.poisson(5 + 4 * np.sin(2 * np.pi * hours / 24) ** 2).astype(np.float64)
    external = DemandSeries("meter-17", protocol.data.start_ns, protocol.data.period_ns, 2, values)
    report = evaluate(protocol, [external])
    assert {r.stream for r in report.runs} == {"meter-17"}
    assert report.dataset_id != evaluate(protocol).dataset_id
    with pytest.raises(InvalidDataError):
        evaluate(protocol, [external, external])
    with pytest.raises(InvalidDataError):
        evaluate(protocol, [])


def test_default_protocol_is_valid_and_small_enough_for_the_example() -> None:
    protocol = default_protocol(weeks=6, streams=1, n_folds=1)
    assert [a.name for a in protocol.adapters] == ["seasonal_naive", "seasonal_ewma", "ridge_lags"]
    assert protocol.splits()[0].fit_cutoff == 24 * 7 * 3


def test_demand_and_online_never_import_market_service_or_plotting_code() -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    code = (
        "import sys; import financeai.domains.demand, financeai.online, financeai.core.forecast; "
        "bad = sorted(m for m in sys.modules if m.startswith(('financeai.domains.market', 'financeai.application',"
        " 'financeai.strategies', 'financeai.cli', 'fastapi', 'pydantic', 'uvicorn', 'matplotlib', 'sklearn')));"
        " print(','.join(bad))"
    )
    env = {**os.environ, "PYTHONPATH": str(src), "FINANCEAI_DISABLE_JIT": "1"}
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120, env=env
    )
    assert done.stdout.strip() == ""
