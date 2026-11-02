"""The demand domain protocol: data, target, features, chronological splits, adapters and metrics (strict JSON).

This is the nonfinancial domain's own validator. It never passes through the inverse-perpetual experiment template,
fee/latency scenarios or a BTC ledger, and a demand target has no venue or USD-face fields. Adapters are selected by
registered kind (allowlist) with JSON parameters; no module path or code is ever taken from a document.

Splits are chronological and purged by availability: fold ``k`` fits on labels released by its fit cutoff only and
issues at the following cutoffs up to the next fold, learning online from matured feedback inside the fold. Every
fold restarts from a fresh fit; no learned state is carried between folds.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from financeai.core.errors import InvalidConfigError
from financeai.core.forecast import ForecastAdapter, TargetSpec, require_keys, require_schema
from financeai.core.hashing import sha256_json
from financeai.core.serialization import canonical_json, loads_strict
from financeai.domains.demand.adapters import build_adapter
from financeai.domains.demand.data import DemandSpec
from financeai.domains.demand.features import FeatureSpec, demand_target_for

PROTOCOL_SCHEMA: Final = "financeai.demand.protocol"
PROTOCOL_SCHEMA_VERSION: Final = "1.0"
_KEYS: Final = (
    "schema",
    "schema_version",
    "name",
    "data",
    "horizon_periods",
    "unit",
    "features",
    "train_periods",
    "n_folds",
    "quantile_levels",
    "online_update",
    "adapters",
)


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or not value.isprintable() or len(value) > 128:
        raise InvalidConfigError(f"{what} must be a non-empty printable name", context={"got": value})
    return value


def _int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InvalidConfigError(f"{name} must be an integer >= {minimum}", context={"got": value})
    return value


def _levels(value: object) -> tuple[float, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, list | tuple):
        raise InvalidConfigError("quantile_levels must be a list of numbers")
    levels: list[float] = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            raise InvalidConfigError("quantile levels must be finite numbers", context={"got": v})
        levels.append(float(v))
    if any(not 0.0 < v < 1.0 for v in levels) or any(b <= a for a, b in itertools.pairwise(levels)):
        raise InvalidConfigError(
            "quantile levels must be strictly increasing inside (0, 1)", context={"levels": levels}
        )
    return tuple(levels)


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    """Fold ``fold``: fit on labels released by ``fit_cutoff``; issue at cutoffs ``fit_cutoff .. stop_cutoff - 1``."""

    fold: int
    fit_cutoff: int
    stop_cutoff: int

    def __post_init__(self) -> None:
        _int(self.fold, "fold", 0)
        _int(self.fit_cutoff, "fit_cutoff", 0)
        if _int(self.stop_cutoff, "stop_cutoff", 0) <= self.fit_cutoff:
            raise InvalidConfigError(
                "a split needs at least one test cutoff",
                context={"fit_cutoff": self.fit_cutoff, "stop_cutoff": self.stop_cutoff},
            )

    def to_mapping(self) -> dict[str, Any]:
        return {"fold": self.fold, "fit_cutoff": self.fit_cutoff, "stop_cutoff": self.stop_cutoff}


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """A registered adapter kind, its JSON parameters (read-only view) and its report name."""

    name: str
    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _name(self.name, "adapter name")
        if not isinstance(self.params, Mapping):
            raise InvalidConfigError("adapter params must be an object", context={"adapter": self.name})
        try:
            text = canonical_json(dict(self.params))
        except TypeError as exc:
            raise InvalidConfigError("adapter params must be JSON values", context={"adapter": self.name}) from exc
        params = MappingProxyType(loads_strict(text))  # a private deep copy
        build_adapter(self.kind, params)  # validates the kind and every parameter eagerly
        object.__setattr__(self, "params", params)

    def build(self) -> ForecastAdapter:
        """A fresh, unfitted adapter instance."""
        return build_adapter(self.kind, self.params)

    def to_mapping(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "params": loads_strict(canonical_json(dict(self.params)))}

    @classmethod
    def from_mapping(cls, doc: object) -> AdapterSpec:
        d = require_keys(doc, ("name", "kind", "params"), where="adapter", error=InvalidConfigError)
        return cls(name=d["name"], kind=d["kind"], params=d["params"])

    def __hash__(self) -> int:
        return hash((self.name, self.kind, canonical_json(dict(self.params))))


@dataclass(frozen=True, slots=True)
class DemandProtocol:
    """Registered demand forecasting protocol (frozen before any evaluation it governs)."""

    name: str
    data: DemandSpec
    adapters: tuple[AdapterSpec, ...]
    horizon_periods: int = 1
    unit: str = "units"
    features: FeatureSpec = field(default_factory=FeatureSpec)
    train_periods: int = 24 * 7 * 6
    """First fit cutoff: the initial fit sees only labels released by then; tests start there."""
    n_folds: int = 1
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    """Requested from quantile-capable adapters only; point-only adapters are scored without pinball loss."""
    online_update: bool = True

    def __post_init__(self) -> None:
        _name(self.name, "protocol name")
        if not isinstance(self.data, DemandSpec) or not isinstance(self.features, FeatureSpec):
            raise InvalidConfigError("data must be a DemandSpec and features a FeatureSpec")
        adapters = tuple(self.adapters)
        if not adapters or not all(isinstance(a, AdapterSpec) for a in adapters):
            raise InvalidConfigError("a protocol needs at least one AdapterSpec")
        names = [a.name for a in adapters]
        if len(set(names)) != len(names):
            raise InvalidConfigError("adapter names must be distinct", context={"names": names})
        horizon = _int(self.horizon_periods, "horizon_periods", 1)
        if not isinstance(self.online_update, bool):
            raise InvalidConfigError("online_update must be true or false")
        for name, value in (
            ("adapters", adapters),
            ("horizon_periods", horizon),
            ("unit", _name(self.unit, "unit")),
            ("train_periods", _int(self.train_periods, "train_periods", 1)),
            ("n_folds", _int(self.n_folds, "n_folds", 1)),
            ("quantile_levels", _levels(self.quantile_levels)),
        ):
            object.__setattr__(self, name, value)
        delay = self.data.label_delay_periods
        first = self.features.first_cutoff(horizon, delay)  # rejects lags that are not yet released
        if self.train_periods - horizon - delay < first:
            raise InvalidConfigError(
                "no complete matured training row exists at the first fit cutoff",
                context={"train_periods": self.train_periods, "first_feature_cutoff": first},
            )
        self.splits()

    def target(self) -> TargetSpec:
        return demand_target_for(
            period_ns=self.data.period_ns,
            horizon_periods=self.horizon_periods,
            delay_periods=self.data.label_delay_periods,
            unit=self.unit,
        )

    def splits(self, n_periods: int | None = None) -> tuple[ChronologicalSplit, ...]:
        """Contiguous test folds after ``train_periods`` for a series of ``n_periods`` (default: the generator's)."""
        n = self.data.n_periods if n_periods is None else _int(n_periods, "n_periods", 1)
        total = (n - self.horizon_periods) + 1 - self.train_periods  # test cutoffs whose target exists
        if total < self.n_folds:
            raise InvalidConfigError(
                "not enough test cutoffs for the requested folds",
                context={"test_cutoffs": max(total, 0), "n_folds": self.n_folds},
            )
        edges = [self.train_periods + (total * k) // self.n_folds for k in range(self.n_folds + 1)]
        return tuple(ChronologicalSplit(k, edges[k], edges[k + 1]) for k in range(self.n_folds))

    @property
    def identity(self) -> str:
        return sha256_json(self.to_mapping())

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": PROTOCOL_SCHEMA,
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "name": self.name,
            "data": self.data.to_mapping(),
            "horizon_periods": self.horizon_periods,
            "unit": self.unit,
            "features": self.features.to_mapping(),
            "train_periods": self.train_periods,
            "n_folds": self.n_folds,
            "quantile_levels": list(self.quantile_levels),
            "online_update": self.online_update,
            "adapters": [a.to_mapping() for a in self.adapters],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_mapping())

    @classmethod
    def from_mapping(cls, doc: object) -> DemandProtocol:
        d = require_keys(
            require_schema(doc, schema=PROTOCOL_SCHEMA, version=PROTOCOL_SCHEMA_VERSION, where="demand protocol"),
            _KEYS,
            where="demand protocol",
            error=InvalidConfigError,
        )
        if not isinstance(d["adapters"], list):
            raise InvalidConfigError("adapters must be a list")
        return cls(
            name=d["name"],
            data=DemandSpec.from_mapping(d["data"]),
            adapters=tuple(AdapterSpec.from_mapping(a) for a in d["adapters"]),
            horizon_periods=d["horizon_periods"],
            unit=d["unit"],
            features=FeatureSpec.from_mapping(d["features"]),
            train_periods=d["train_periods"],
            n_folds=d["n_folds"],
            quantile_levels=_levels(d["quantile_levels"]),
            online_update=d["online_update"],
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> DemandProtocol:
        return cls.from_mapping(loads_strict(text))


def default_protocol(*, weeks: int = 12, streams: int = 2, n_folds: int = 2, seed: int = 7) -> DemandProtocol:
    """The example protocol: hourly synthetic demand, a level shift at 75 %, three baselines, half for training."""
    weeks = _int(weeks, "weeks", 5)  # the first fit must see every hour-of-week phase after the warm-up
    n = 24 * 7 * weeks
    data = DemandSpec(
        streams=tuple(f"sensor-{k}" for k in range(_int(streams, "streams", 1))),
        n_periods=n,
        shift_period=(3 * n) // 4,
        shift_factor=1.5,
        seed=seed,
    )
    adapters = (
        AdapterSpec("seasonal_naive", "seasonal_naive"),
        AdapterSpec("seasonal_ewma", "seasonal_ewma"),
        AdapterSpec("ridge_lags", "ridge_lags"),
    )
    return DemandProtocol(
        name="synthetic-delayed-demand", data=data, adapters=adapters, train_periods=n // 2, n_folds=n_folds
    )
