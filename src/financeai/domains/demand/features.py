"""Causal demand features, the demand target, and train/serve-consistent queries and mature batches.

Row ``c`` describes the forecast issued at cutoff ``c`` (clock ``start + c P``) for target period ``s = c + h - 1``,
the ``h``-th period after the cutoff. Every value a row uses was released by its cutoff: lags ``L >= h + d`` (``d``
the reporting delay, so the most recent released value is lag ``h + d``) and trailing means over the released
prefix. Calendar terms (hour/day sin-cos, weekend flag, hour-of-week phase) describe the target period and are known
in advance. Training rows come from the same matrix as queries, so a model is fitted on exactly the features it
would have had at issue time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from financeai.core.clock import NS_PER_DAY
from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.core.forecast import (
    AvailabilityRule,
    ForecastQuery,
    MatureBatch,
    TargetSpec,
    encode_ns,
    require_keys,
)
from financeai.core.hashing import sha256_json
from financeai.domains.demand.data import DemandSeries

FEATURE_VERSION: Final = "demand.features/1"
TARGET_SCHEMA: Final = "demand.period_total/1"
TARGET_LABELER: Final = "demand.period_total@1"
_MONDAY_NS: Final = 4 * NS_PER_DAY  # 1970-01-05T00:00:00Z
CALENDAR_FEATURES: Final = ("hod_sin", "hod_cos", "dow_sin", "dow_cos", "weekend", "phase")


def _periods(value: object, name: str) -> tuple[int, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, tuple | list):
        raise InvalidConfigError(f"{name} must be a list of positive integers")
    out: list[int] = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise InvalidConfigError(f"{name} must contain positive integers", context={"got": v})
        out.append(v)
    if len(set(out)) != len(out):
        raise InvalidConfigError(f"{name} must be distinct", context={name: out})
    return tuple(sorted(out))


def _horizon(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidConfigError("horizon_periods must be a positive integer", context={"got": value})
    return value


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Seasonal lags and trailing-mean windows (in periods); the lag ``horizon + delay`` is always added."""

    lags: tuple[int, ...] = (24, 48, 168)
    windows: tuple[int, ...] = (24, 168)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lags", _periods(self.lags, "lags"))
        object.__setattr__(self, "windows", _periods(self.windows, "windows"))

    def effective_lags(self, horizon_periods: int, delay_periods: int) -> tuple[int, ...]:
        recent = _horizon(horizon_periods) + delay_periods
        unreleased = [lag for lag in self.lags if lag < recent]
        if unreleased:
            raise InvalidConfigError(
                "lags need values not yet released at the cutoff; every lag must be >= horizon + delay",
                context={"lags": unreleased, "minimum": recent},
            )
        return tuple(sorted({*self.lags, recent}))

    def names(self, horizon_periods: int, delay_periods: int) -> tuple[str, ...]:
        lags = self.effective_lags(horizon_periods, delay_periods)
        return (*(f"lag_{lag}" for lag in lags), *(f"mean_{w}" for w in self.windows), *CALENDAR_FEATURES)

    def first_cutoff(self, horizon_periods: int, delay_periods: int) -> int:
        """First cutoff with a complete row: every lag index and trailing window lies in the released prefix."""
        lags = self.effective_lags(horizon_periods, delay_periods)
        return max([lag - horizon_periods + 1 for lag in lags] + [w + delay_periods for w in self.windows])

    def to_mapping(self) -> dict[str, Any]:
        return {"lags": list(self.lags), "windows": list(self.windows)}

    @classmethod
    def from_mapping(cls, doc: object) -> FeatureSpec:
        d = require_keys(doc, ("lags", "windows"), where="features", error=InvalidConfigError)
        return cls(lags=d["lags"], windows=d["windows"])


@dataclass(frozen=True, slots=True, eq=False)
class DemandFeatures:
    """Feature rows for cutoffs ``first_cutoff .. last_cutoff`` of one series (read-only)."""

    stream: str
    series_hash: str
    horizon_periods: int
    feature_names: tuple[str, ...]
    transform_version: str
    first_cutoff: int
    x: NDArray[np.float64]

    @property
    def last_cutoff(self) -> int:
        return self.first_cutoff + int(self.x.shape[0]) - 1

    def row(self, cutoff: int) -> NDArray[np.float64]:
        if not self.first_cutoff <= cutoff <= self.last_cutoff:
            raise InvalidDataError(
                "no complete feature row at this cutoff",
                context={"cutoff": cutoff, "first": self.first_cutoff, "last": self.last_cutoff},
            )
        row: NDArray[np.float64] = self.x[cutoff - self.first_cutoff]
        return row


def compute_features(series: DemandSeries, *, horizon_periods: int, spec: FeatureSpec) -> DemandFeatures:
    """All complete causal feature rows of ``series``; row ``c`` reads only the prefix released at cutoff ``c``."""
    h = _horizon(horizon_periods)
    d, period, values = series.label_delay_periods, series.period_ns, series.values
    lags = spec.effective_lags(h, d)
    first = spec.first_cutoff(h, d)
    last = series.n - h
    if last < first:
        raise InvalidDataError(
            "series too short for these features",
            context={"stream": series.stream, "periods": series.n, "first_cutoff": first},
        )
    cutoffs = np.arange(first, last + 1, dtype=np.int64)
    target = cutoffs + (h - 1)
    released = cutoffs - d  # values[:released] are reported by the cutoff
    columns: list[NDArray[np.float64]] = [values[target - lag] for lag in lags]
    for window in spec.windows:
        means = sliding_window_view(values, window).mean(axis=1)
        columns.append(means[released - window])  # mean of values[released - window : released]
    starts = series.start_ns + target * period
    day = (starts % NS_PER_DAY).astype(np.float64) / NS_PER_DAY
    dow = (starts // NS_PER_DAY + 3) % 7  # 0 = Monday
    phase = ((starts - _MONDAY_NS) // period) % (7 * NS_PER_DAY // period)
    columns += [
        np.sin(2.0 * np.pi * day),
        np.cos(2.0 * np.pi * day),
        np.sin(2.0 * np.pi * dow / 7.0),
        np.cos(2.0 * np.pi * dow / 7.0),
        (dow >= 5).astype(np.float64),
        phase.astype(np.float64),
    ]
    x = np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)
    frozen = np.frombuffer(x.tobytes(), dtype=np.float64).reshape(x.shape)
    version = sha256_json(
        {
            "lags": list(lags),
            "windows": list(spec.windows),
            "horizon_periods": h,
            "label_delay_periods": d,
            "period_ns": encode_ns(period),
        }
    )[:16]
    return DemandFeatures(
        stream=series.stream,
        series_hash=series.content_hash,
        horizon_periods=h,
        feature_names=spec.names(h, d),
        transform_version=f"{FEATURE_VERSION}:{version}",
        first_cutoff=first,
        x=frozen,
    )


def demand_target_for(*, period_ns: int, horizon_periods: int, delay_periods: int, unit: str = "units") -> TargetSpec:
    """Total demand of the ``horizon_periods``-th period after the cutoff, reported ``delay_periods`` after it ends."""
    h = _horizon(horizon_periods)
    rule = AvailabilityRule("reported_after_period_end", min_delay_ns=delay_periods * period_ns)
    return TargetSpec(
        domain="demand",
        schema=TARGET_SCHEMA,
        unit=unit,
        horizon_ns=h * period_ns,
        labeler=TARGET_LABELER,
        availability=rule,
    )


def demand_target(series: DemandSeries, *, horizon_periods: int, unit: str = "units") -> TargetSpec:
    """The demand target of ``series`` (its period and reporting delay)."""
    return demand_target_for(
        period_ns=series.period_ns, horizon_periods=horizon_periods, delay_periods=series.label_delay_periods, unit=unit
    )


def _check_pairing(series: DemandSeries, features: DemandFeatures, target: TargetSpec) -> None:
    if features.series_hash != series.content_hash:
        raise InvalidDataError("features were computed from another series", context={"stream": series.stream})
    if (
        target.horizon_ns != features.horizon_periods * series.period_ns
        or target.availability.min_delay_ns != series.label_delay_periods * series.period_ns
    ):
        raise InvalidDataError(
            "target horizon/delay differ from the features and series", context={"stream": series.stream}
        )


def make_query(
    series: DemandSeries,
    features: DemandFeatures,
    cutoff: int,
    *,
    target: TargetSpec,
    query_id: str,
    quantile_levels: tuple[float, ...] = (),
) -> ForecastQuery:
    """The query issued at ``cutoff`` (issue time = feature cutoff: computation is treated as instantaneous)."""
    _check_pairing(series, features, target)
    clock = series.cutoff_ns(cutoff)
    return ForecastQuery(
        query_id=query_id,
        stream=series.stream,
        target=target,
        issued_at_ns=clock,
        feature_cutoff_ns=clock,
        features=features.row(cutoff),
        feature_names=features.feature_names,
        transform_version=features.transform_version,
        quantile_levels=quantile_levels,
    )


def mature_batch(
    series: DemandSeries, features: DemandFeatures, *, as_of_cutoff: int, target: TargetSpec
) -> MatureBatch:
    """Every complete row whose label is released by ``as_of_cutoff`` (chronological, issue-time features)."""
    _check_pairing(series, features, target)
    h, d, period = features.horizon_periods, series.label_delay_periods, series.period_ns
    last = min(features.last_cutoff, as_of_cutoff - h - d)  # label of row c is released at cutoff c + h + d
    if last < features.first_cutoff:
        raise InvalidDataError(
            "no matured training rows at this cutoff",
            context={"as_of_cutoff": as_of_cutoff, "first_cutoff": features.first_cutoff},
        )
    cutoffs = np.arange(features.first_cutoff, last + 1, dtype=np.int64)
    rows = slice(0, last - features.first_cutoff + 1)
    return MatureBatch(
        target=target,
        feature_names=features.feature_names,
        transform_version=features.transform_version,
        x=features.x[rows],
        y=series.values[cutoffs + h - 1],
        feature_cutoff_ns=series.start_ns + cutoffs * period,
        available_at_ns=series.start_ns + (cutoffs + h + d) * period,
        as_of_ns=series.cutoff_ns(as_of_cutoff),
    )
