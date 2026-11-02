"""Prequential evaluation of demand adapters with delayed feedback over chronological splits.

A :class:`DemandRun` owns one adapter on one stream for one split. At each cutoff it releases every label that has
matured (updating the adapter), issues one ticket from the features released by that cutoff, and schedules the
ticket's outcome, which the ticket book keeps invisible until its reporting time. ``finish`` advances to the
reporting time of the last target so every issued ticket is scored exactly once. Runs checkpoint to strict JSON
(run identity including the target, cursor, learner, adapter and book) and resume exactly; runs are independent, so
they can be distributed freely. A resume is accepted only if the restored learner is exactly what this run builds
before the cursor: one ticket per earlier cutoff answering this run's query, the series' outcome of each, this run's
update mode and the book clock of the cursor. Issued payloads (means, variances, quantiles, model versions) and the
adapter state are model output the run cannot re-derive; the book's score replay pins those of released tickets.

Timing is HISTORICAL_REPLAY: computation is instantaneous at the cutoff (issue time = feature cutoff).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from financeai.core.errors import IncompatibleSchemaError, InvalidConfigError, InvalidDataError
from financeai.core.forecast import (
    Capability,
    ForecastAdapter,
    ForecastQuery,
    ForecastTicket,
    Outcome,
    require_keys,
    require_schema,
)
from financeai.core.hashing import sha256_json
from financeai.core.serialization import canonical_json, loads_strict
from financeai.domains.demand.data import DemandSeries, generate_demand
from financeai.domains.demand.features import DemandFeatures, compute_features, demand_target, make_query, mature_batch
from financeai.domains.demand.metrics import DemandMetrics, summarize
from financeai.domains.demand.protocol import ChronologicalSplit, DemandProtocol
from financeai.online.book import TicketBook
from financeai.online.learner import DelayedFeedbackLearner, answer_mismatches
from financeai.online.scoring import ScoreRecord

RUN_SCHEMA: Final = "financeai.demand.run"
RUN_SCHEMA_VERSION: Final = "1.0"
_RUN_KEYS: Final = (
    "schema",
    "schema_version",
    "name",
    "stream",
    "series_hash",
    "transform_version",
    "target",
    "split",
    "quantile_levels",
    "online_update",
    "cursor",
    "learner",
)


@dataclass(frozen=True, slots=True)
class RunResult:
    adapter: str
    stream: str
    fold: int
    tickets: tuple[ForecastTicket, ...]
    scores: tuple[ScoreRecord, ...]
    metrics: DemandMetrics


class DemandRun:
    """One adapter, one stream, one chronological split; the run is the single owner of its learner."""

    def __init__(
        self,
        series: DemandSeries,
        features: DemandFeatures,
        adapter: ForecastAdapter,
        *,
        name: str,
        split: ChronologicalSplit,
        quantile_levels: tuple[float, ...] = (),
        online_update: bool | None = None,
        unit: str = "units",
    ) -> None:
        if features.series_hash != series.content_hash:
            raise InvalidDataError("features were computed from another series", context={"stream": series.stream})
        if split.fit_cutoff < features.first_cutoff or split.stop_cutoff - 1 > features.last_cutoff:
            raise InvalidConfigError(
                "split cutoffs fall outside the complete feature rows",
                context={"split": split.to_mapping(), "first": features.first_cutoff, "last": features.last_cutoff},
            )
        self._series, self._features, self._adapter = series, features, adapter
        self._name, self._split, self._online_update = name, split, online_update
        self._target = demand_target(series, horizon_periods=features.horizon_periods, unit=unit)
        quantiles = Capability.MARGINAL_QUANTILES in adapter.capabilities
        self._levels = tuple(quantile_levels) if quantiles else ()
        self._learner: DelayedFeedbackLearner | None = None
        self._cursor: int | None = None

    @property
    def cursor(self) -> int | None:
        """Next cutoff to issue at (``None`` before :meth:`start`)."""
        return self._cursor

    @property
    def done(self) -> bool:
        return self._cursor == self._split.stop_cutoff

    @property
    def learner(self) -> DelayedFeedbackLearner:
        if self._learner is None:
            raise InvalidDataError("run has not started")
        return self._learner

    def start(self) -> None:
        """Fit on every label released by the split's fit cutoff (nothing later)."""
        if self._learner is not None:
            raise InvalidDataError("run already started")
        batch = mature_batch(self._series, self._features, as_of_cutoff=self._split.fit_cutoff, target=self._target)
        self._adapter.fit(batch)
        self._learner = DelayedFeedbackLearner(self._adapter, online_update=self._online_update)
        self._cursor = self._split.fit_cutoff

    def _ticket_id(self, cutoff: int) -> str:
        return f"{self._name}/{self._series.stream}/f{self._split.fold}/c{cutoff}"

    def _query(self, cutoff: int) -> ForecastQuery:
        return make_query(
            self._series,
            self._features,
            cutoff,
            target=self._target,
            query_id=self._ticket_id(cutoff),
            quantile_levels=self._levels,
        )

    def _outcome(self, cutoff: int) -> Outcome:
        """The reported value of the target of ``cutoff`` and the clock at which it becomes available."""
        target_period = cutoff + self._features.horizon_periods - 1
        return Outcome(
            self._ticket_id(cutoff),
            float(self._series.values[target_period]),
            self._series.available_at_ns(target_period),
        )

    def _final_clock(self) -> int:
        """Reporting time of the split's last target (``finish`` waits until then)."""
        last_target = self._split.stop_cutoff - 1 + self._features.horizon_periods - 1
        return self._series.available_at_ns(last_target)

    def step(self) -> ForecastTicket:
        """Release matured labels, issue the ticket of the current cutoff and schedule its (hidden) outcome."""
        learner, cutoff = self.learner, self._cursor
        if cutoff is None or self.done:
            raise InvalidDataError("no cutoff left in this split")
        ticket = learner.predict(self._query(cutoff))
        learner.submit(self._outcome(cutoff))
        self._cursor = cutoff + 1
        return ticket

    def run(self, until: int | None = None) -> None:
        """Issue at every cutoff before ``until`` (default: the end of the split)."""
        stop = self._split.stop_cutoff if until is None else min(until, self._split.stop_cutoff)
        while self._cursor is not None and self._cursor < stop:
            self.step()

    def finish(self) -> RunResult:
        """Wait until the last target is reported, score every ticket once and summarise."""
        if not self.done:
            raise InvalidDataError("run has unissued cutoffs", context={"cursor": self._cursor})
        learner = self.learner
        learner.advance(self._final_clock())
        tickets, scores = learner.book.tickets, learner.book.scores
        return RunResult(
            adapter=self._name,
            stream=self._series.stream,
            fold=self._split.fold,
            tickets=tickets,
            scores=scores,
            metrics=summarize(scores, n_issued=len(tickets)),
        )

    def _identity(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "stream": self._series.stream,
            "series_hash": self._series.content_hash,
            "transform_version": self._features.transform_version,
            "target": self._target.to_mapping(),
            "split": self._split.to_mapping(),
            "quantile_levels": list(self._levels),
            "online_update": self._online_update,
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            **self._identity(),
            "cursor": self._cursor,
            "learner": None if self._learner is None else self._learner.get_state(),
        }

    def to_json(self) -> str:
        return canonical_json(self.get_state())

    def _reconcile(self, book: TicketBook, updating: bool, cursor: int) -> None:
        """Reject a restored learner that is not exactly what this run builds before ``cursor``."""
        mode = DelayedFeedbackLearner(self._adapter, online_update=self._online_update).updating  # as start() sets it
        if updating != mode:
            raise InvalidDataError(
                "checkpoint learner update mode differs from this run", context={"checkpoint": updating, "run": mode}
            )
        cutoffs = range(self._split.fit_cutoff, cursor)
        tickets = book.tickets
        if [t.ticket_id for t in tickets] != [self._ticket_id(c) for c in cutoffs]:
            raise InvalidDataError(
                "checkpoint tickets are not this run's cutoffs before its cursor, in order",
                context={"cursor": cursor, "tickets": len(tickets)},
            )
        for cutoff, ticket in zip(cutoffs, tickets, strict=True):
            fields = answer_mismatches(ticket, self._query(cutoff), model_id=self._adapter.model_id)
            if fields:
                raise InvalidDataError(
                    "checkpoint ticket does not answer this run's query",
                    context={"ticket_id": ticket.ticket_id, "fields": list(fields)},
                )
        if book.get_state()["outcomes"] != [self._outcome(c).to_mapping() for c in cutoffs]:
            raise InvalidDataError("checkpoint outcomes differ from this run's series", context={"cursor": cursor})
        # nothing is released before the first issue; afterwards the clock is the last issue time (the final
        # reporting time once ``finish`` has run)
        clocks: tuple[int | None, ...] = (
            (None,) if cursor == self._split.fit_cutoff else (self._series.cutoff_ns(cursor - 1),)
        )
        if cursor == self._split.stop_cutoff:
            clocks += (self._final_clock(),)
        if book.clock_ns not in clocks:
            raise InvalidDataError(
                "checkpoint clock disagrees with its cursor", context={"cursor": cursor, "clock_ns": book.clock_ns}
            )

    def load_state(self, doc: object) -> None:
        """Resume a checkpoint into this fresh run (same series, features, target, split and adapter configuration).

        The restored learner must be exactly what this run builds before the checkpoint's cursor. These checks run
        before the adapter state is applied, so a checkpoint they reject leaves the run unstarted and its adapter
        unchanged.
        """
        if self._learner is not None:
            raise InvalidDataError("load_state needs a run that has not started")
        d = require_keys(
            require_schema(doc, schema=RUN_SCHEMA, version=RUN_SCHEMA_VERSION, where="demand run"),
            _RUN_KEYS,
            where="demand run",
        )
        mismatched = sorted(k for k, v in self._identity().items() if d[k] != v)
        if mismatched:
            raise IncompatibleSchemaError("checkpoint belongs to another run", context={"fields": mismatched})
        cursor = d["cursor"]
        if cursor is None or d["learner"] is None:
            if cursor is not None or d["learner"] is not None:
                raise InvalidDataError("checkpoint mixes started and unstarted fields")
            return
        if (
            isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or not self._split.fit_cutoff <= cursor <= self._split.stop_cutoff
        ):
            raise InvalidDataError("checkpoint cursor lies outside the split", context={"cursor": cursor})

        def reconcile(book: TicketBook, updating: bool) -> None:
            self._reconcile(book, updating, cursor)

        self._learner = DelayedFeedbackLearner.from_state(d["learner"], self._adapter, reconcile=reconcile)
        self._cursor = cursor

    def load_json(self, text: str | bytes) -> None:
        self.load_state(loads_strict(text))


@dataclass(frozen=True, slots=True)
class DemandReport:
    """Every run of a protocol evaluation, plus per-adapter metrics pooled over streams and folds."""

    protocol_name: str
    protocol_id: str
    dataset_id: str
    adapters: tuple[str, ...]
    runs: tuple[RunResult, ...]

    def pooled(self) -> dict[str, DemandMetrics]:
        out: dict[str, DemandMetrics] = {}
        for name in self.adapters:
            runs = [r for r in self.runs if r.adapter == name]
            out[name] = summarize([s for r in runs for s in r.scores], n_issued=sum(len(r.tickets) for r in runs))
        return out

    def to_mapping(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol_name,
            "protocol_id": self.protocol_id,
            "dataset_id": self.dataset_id,
            "runs": [
                {"adapter": r.adapter, "stream": r.stream, "fold": r.fold, "metrics": r.metrics.to_mapping()}
                for r in self.runs
            ],
            "pooled": {name: m.to_mapping() for name, m in self.pooled().items()},
        }


def evaluate(protocol: DemandProtocol, dataset: Sequence[DemandSeries] | None = None) -> DemandReport:
    """Run every adapter of ``protocol`` on every stream and fold (synthetic data unless ``dataset`` is given)."""
    series_list = generate_demand(protocol.data) if dataset is None else tuple(dataset)
    if not series_list or not all(isinstance(s, DemandSeries) for s in series_list):
        raise InvalidDataError("evaluation needs at least one DemandSeries")
    streams = [s.stream for s in series_list]
    if len(set(streams)) != len(streams):
        raise InvalidDataError("dataset streams must be distinct", context={"streams": streams})
    runs: list[RunResult] = []
    for series in series_list:
        features = compute_features(series, horizon_periods=protocol.horizon_periods, spec=protocol.features)
        for split in protocol.splits(series.n):
            for spec in protocol.adapters:
                run = DemandRun(
                    series,
                    features,
                    spec.build(),
                    name=spec.name,
                    split=split,
                    quantile_levels=protocol.quantile_levels,
                    online_update=protocol.online_update,
                    unit=protocol.unit,
                )
                run.start()
                run.run()
                runs.append(run.finish())
    order = {
        (name, stream): i for i, (name, stream) in enumerate((a.name, s) for a in protocol.adapters for s in streams)
    }
    runs.sort(key=lambda r: (r.fold, order[(r.adapter, r.stream)]))
    return DemandReport(
        protocol_name=protocol.name,
        protocol_id=protocol.identity,
        dataset_id=sha256_json([s.content_hash for s in series_list]),
        adapters=tuple(a.name for a in protocol.adapters),
        runs=tuple(runs),
    )
