"""Small deterministic adapters and fixtures for the online machinery tests (invented values)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from financeai.core.forecast import (
    AvailabilityRule,
    Capability,
    ForecastQuery,
    ForecastTicket,
    MatureBatch,
    Outcome,
    TargetSpec,
    require_capability,
)

HOUR = 3_600 * 10**9
T0 = 1_704_067_200 * 10**9  # 2024-01-01T00:00:00Z
TARGET = TargetSpec(
    domain="toy",
    schema="toy.value/1",
    unit="units",
    horizon_ns=HOUR,
    labeler="toy.value@1",
    availability=AvailabilityRule("after_horizon"),
)


def query(i: int, *, issued_at_ns: int | None = None, levels: tuple[float, ...] = (), x: float = 0.0) -> ForecastQuery:
    t = T0 + i * HOUR if issued_at_ns is None else issued_at_ns
    return ForecastQuery(
        query_id=f"q{i:04d}",
        stream="s0",
        target=TARGET,
        issued_at_ns=t,
        feature_cutoff_ns=t,
        features=np.array([float(i), x]),
        feature_names=("i", "x"),
        transform_version="toy@1",
        quantile_levels=levels,
    )


def ticket(
    i: int, *, mean: float = 1.0, levels: tuple[float, ...] = (), quantiles: list[float] | None = None
) -> ForecastTicket:
    return ForecastTicket.from_query(
        query(i, levels=levels), model_id="toy", model_version="1", mean=mean, quantiles=quantiles
    )


class RunningMean:
    """Welford mean/variance of every matured outcome; records the order of updates it received."""

    model_id = "running_mean"
    capabilities = frozenset(
        {Capability.MEAN, Capability.VARIANCE, Capability.ONLINE_UPDATE, Capability.DELAYED_FEEDBACK}
    )

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.updated: list[str] = []

    @property
    def model_version(self) -> str:
        return f"u{self.n}"

    def _add(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    def fit(self, batch: MatureBatch) -> None:
        for value in batch.y.tolist():
            self._add(value)

    def predict(self, q: ForecastQuery) -> ForecastTicket:
        require_capability(
            self.capabilities,
            Capability.MARGINAL_QUANTILES if q.quantile_levels else Capability.MEAN,
            component=self.model_id,
        )
        variance = self.m2 / (self.n - 1) if self.n > 1 else 0.0
        return ForecastTicket.from_query(
            q, model_id=self.model_id, model_version=self.model_version, mean=self.mean, variance=variance
        )

    def update(self, t: ForecastTicket, outcome: Outcome) -> None:
        self.updated.append(t.ticket_id)
        self._add(outcome.value)

    def get_state(self) -> dict[str, Any]:
        return {"n": self.n, "mean": self.mean, "m2": self.m2, "updated": list(self.updated)}

    def set_state(self, state: Mapping[str, Any]) -> None:
        self.n, self.mean, self.m2 = int(state["n"]), float(state["mean"]), float(state["m2"])
        self.updated = list(state["updated"])


class Flaky(RunningMean):
    """Raises inside the ``fail_at``-th update (1-based) to inject a crash between scoring and state commit."""

    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self.fail_at = fail_at
        self.calls = 0

    def update(self, t: ForecastTicket, outcome: Outcome) -> None:
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("injected crash")
        super().update(t, outcome)


class Constant:
    """Frozen point forecaster (no online update)."""

    model_id = "constant"
    model_version = "1"
    capabilities = frozenset({Capability.MEAN})

    def fit(self, batch: MatureBatch) -> None:
        return None

    def predict(self, q: ForecastQuery) -> ForecastTicket:
        require_capability(
            self.capabilities,
            Capability.MARGINAL_QUANTILES if q.quantile_levels else Capability.MEAN,
            component=self.model_id,
        )
        return ForecastTicket.from_query(q, model_id=self.model_id, model_version=self.model_version, mean=2.0)

    def update(self, t: ForecastTicket, outcome: Outcome) -> None:
        require_capability(self.capabilities, Capability.ONLINE_UPDATE, component=self.model_id)

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: Mapping[str, Any]) -> None:
        return None


class ImmediateOnly(RunningMean):
    """Declares online updates but not delayed feedback (assumes each outcome arrives before the next query)."""

    model_id = "immediate_only"
    capabilities = frozenset({Capability.MEAN, Capability.VARIANCE, Capability.ONLINE_UPDATE})


class Misbehaving(RunningMean):
    """Returns a ticket for a different query ID (an adapter bug the learner must catch)."""

    model_id = "misbehaving"

    def predict(self, q: ForecastQuery) -> ForecastTicket:
        other = ForecastQuery(
            query_id=q.query_id + "-x",
            stream=q.stream,
            target=q.target,
            issued_at_ns=q.issued_at_ns,
            feature_cutoff_ns=q.feature_cutoff_ns,
            features=q.features,
            feature_names=q.feature_names,
            transform_version=q.transform_version,
        )
        return ForecastTicket.from_query(other, model_id=self.model_id, model_version="1", mean=0.0)
