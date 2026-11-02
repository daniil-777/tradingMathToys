"""Demand/sensor series with delayed label reporting and a deterministic synthetic generator (invented values).

A :class:`DemandSeries` is one stream on a regular period grid: value ``i`` covers ``[start + i P, start + (i + 1) P)``
and is reported ``label_delay_periods`` periods after its period ends, i.e. ``available_at(i) = start + (i + 1 + d) P``.
At a cutoff only the released prefix is visible; nothing else in this domain reads past it.

:func:`generate_demand` draws counts with a daily and a weekly cycle, a weekend effect, zero-inflated intermittency
and an optional level shift. Each stream is seeded from ``(seed, stream position)`` through NumPy's
``SeedSequence``, so adding a stream never changes the others. Values are bitwise reproducible for one NumPy
version (its Poisson sampler is not guaranteed stable across releases).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from financeai.core.clock import NS_PER_DAY, NS_PER_S
from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.core.forecast import INT64_MAX, decode_ns, encode_ns, require_keys
from financeai.core.serialization import canonical_json

HOURLY_NS: Final = 3_600 * NS_PER_S
DEFAULT_START_NS: Final = 1_704_067_200 * NS_PER_S  # 2024-01-01T00:00:00Z, a Monday
_STREAM_LEVEL_SPREAD: Final = 0.25  # log-normal spread of per-stream base levels


def _int(value: object, name: str, minimum: int, maximum: int = INT64_MAX) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or not minimum <= int(value) <= maximum:
        raise InvalidConfigError(f"{name} must be an integer in [{minimum}, {maximum}]", context={"got": value})
    return int(value)


def _float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise InvalidConfigError(f"{name} must be a number", context={"got": value})
    out = float(value)
    if not math.isfinite(out):
        raise InvalidConfigError(f"{name} must be finite", context={"got": value})
    return out


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or not value.isprintable() or len(value) > 128:
        raise InvalidConfigError(f"{what} must be a non-empty printable name", context={"got": value})
    return value


def _period(value: object) -> int:
    period = _int(value, "period_ns", 1, NS_PER_DAY)
    if NS_PER_DAY % period:
        raise InvalidConfigError(
            "period_ns must divide one day (calendar phases need whole periods)", context={"period_ns": period}
        )
    return period


def _config_ns(value: object, name: str) -> int:
    try:
        return decode_ns(value)
    except InvalidDataError as exc:
        raise InvalidConfigError(f"{name} must be a canonical decimal nanosecond string") from exc


@dataclass(frozen=True, slots=True, eq=False)
class DemandSeries:
    """One stream of non-negative period totals with a uniform reporting delay (values copied, read-only)."""

    stream: str
    start_ns: int
    period_ns: int
    label_delay_periods: int
    values: NDArray[np.float64]
    _hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _name(self.stream, "stream")
        start = _int(self.start_ns, "start_ns", 0)
        period = _period(self.period_ns)
        delay = _int(self.label_delay_periods, "label_delay_periods", 0)
        raw = np.asarray(self.values)
        if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.shape[0] == 0:
            raise InvalidDataError("values must be a non-empty 1-D real array", context={"stream": self.stream})
        values = raw.astype(np.float64)
        if not bool(np.all(np.isfinite(values))) or bool(np.any(values < 0)):
            raise InvalidDataError("values must be finite and non-negative", context={"stream": self.stream})
        if start + (values.shape[0] + delay + 1) * period > INT64_MAX:
            raise InvalidConfigError("series clocks overflow int64", context={"stream": self.stream})
        frozen = np.frombuffer(values.tobytes(), dtype=np.float64)
        header = canonical_json(
            {
                "stream": self.stream,
                "start_ns": encode_ns(start),
                "period_ns": encode_ns(period),
                "label_delay_periods": delay,
                "n": int(values.shape[0]),
            }
        )
        digest = hashlib.sha256(header.encode("utf-8") + b"\x00" + frozen.astype("<f8").tobytes()).hexdigest()
        for name, value in (
            ("start_ns", start),
            ("period_ns", period),
            ("label_delay_periods", delay),
            ("values", frozen),
            ("_hash", digest),
        ):
            object.__setattr__(self, name, value)

    @property
    def n(self) -> int:
        return int(self.values.shape[0])

    @property
    def content_hash(self) -> str:
        """SHA-256 of the stream identity, clocks, delay and values (little-endian float64)."""
        return self._hash

    def cutoff_ns(self, cutoff: int) -> int:
        """Clock of cutoff ``cutoff`` (the boundary before period ``cutoff``)."""
        return self.start_ns + _int(cutoff, "cutoff", 0) * self.period_ns

    def period_start_ns(self, index: int) -> int:
        return self.start_ns + _int(index, "period index", 0, self.n - 1) * self.period_ns

    def available_at_ns(self, index: int) -> int:
        """When the value of period ``index`` is reported: ``label_delay_periods`` periods after the period ends."""
        index = _int(index, "period index", 0, self.n - 1)
        return self.start_ns + (index + 1 + self.label_delay_periods) * self.period_ns

    def visible_count(self, cutoff: int) -> int:
        """Number of leading values released by cutoff ``cutoff`` (availability is monotone, so a prefix)."""
        return min(self.n, max(0, _int(cutoff, "cutoff", 0) - self.label_delay_periods))

    def visible(self, cutoff: int) -> NDArray[np.float64]:
        """The released prefix at ``cutoff`` (read-only)."""
        return self.values[: self.visible_count(cutoff)]

    def with_values(self, values: NDArray[np.float64]) -> DemandSeries:
        """Same stream and clocks with other values (used for counterfactual and mutation checks)."""
        return dataclasses.replace(self, values=values)


@dataclass(frozen=True, slots=True)
class DemandSpec:
    """Synthetic generator configuration (domain protocol data section; no market or currency fields)."""

    streams: tuple[str, ...] = ("sensor-0",)
    n_periods: int = 24 * 7 * 12
    period_ns: int = HOURLY_NS
    start_ns: int = DEFAULT_START_NS
    base_level: float = 6.0
    daily_amplitude: float = 0.8
    """Log-intensity amplitude of the daily cycle (peak at noon UTC)."""
    weekly_amplitude: float = 0.25
    """Log-intensity amplitude of a smooth weekly cycle."""
    weekend_effect: float = -0.5
    """Log-intensity offset on Saturdays and Sundays (UTC)."""
    zero_probability: float = 0.2
    """Intermittency: probability that a period records no demand regardless of its intensity."""
    shift_period: int | None = None
    shift_factor: float = 1.0
    """Multiplicative level shift applied from ``shift_period`` on (a regime change for online adapters)."""
    label_delay_periods: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        streams: object = self.streams
        if isinstance(streams, str) or not isinstance(streams, Iterable):
            raise InvalidConfigError("streams must be a sequence of names")
        names = tuple(_name(s, "stream") for s in streams)
        if not names or len(set(names)) != len(names):
            raise InvalidConfigError("streams must be a non-empty list of distinct names", context={"streams": names})
        n = _int(self.n_periods, "n_periods", 1)
        period = _period(self.period_ns)
        start = _int(self.start_ns, "start_ns", 0)
        delay = _int(self.label_delay_periods, "label_delay_periods", 0)
        if start + (n + delay + 1) * period > INT64_MAX:
            raise InvalidConfigError("generator clocks overflow int64")
        base = _float(self.base_level, "base_level")
        if base <= 0:
            raise InvalidConfigError("base_level must be positive")
        zero = _float(self.zero_probability, "zero_probability")
        if not 0.0 <= zero < 1.0:
            raise InvalidConfigError("zero_probability must be in [0, 1)")
        shift = None if self.shift_period is None else _int(self.shift_period, "shift_period", 0, n - 1)
        factor = _float(self.shift_factor, "shift_factor")
        if factor <= 0:
            raise InvalidConfigError("shift_factor must be positive")
        for name, value in (
            ("streams", names),
            ("n_periods", n),
            ("period_ns", period),
            ("start_ns", start),
            ("base_level", base),
            ("daily_amplitude", _float(self.daily_amplitude, "daily_amplitude")),
            ("weekly_amplitude", _float(self.weekly_amplitude, "weekly_amplitude")),
            ("weekend_effect", _float(self.weekend_effect, "weekend_effect")),
            ("zero_probability", zero),
            ("shift_period", shift),
            ("shift_factor", factor),
            ("label_delay_periods", delay),
            ("seed", _int(self.seed, "seed", 0, 2**63 - 1)),
        ):
            object.__setattr__(self, name, value)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "streams": list(self.streams),
            "n_periods": self.n_periods,
            "period_ns": encode_ns(self.period_ns),
            "start_ns": encode_ns(self.start_ns),
            "base_level": self.base_level,
            "daily_amplitude": self.daily_amplitude,
            "weekly_amplitude": self.weekly_amplitude,
            "weekend_effect": self.weekend_effect,
            "zero_probability": self.zero_probability,
            "shift_period": self.shift_period,
            "shift_factor": self.shift_factor,
            "label_delay_periods": self.label_delay_periods,
            "seed": self.seed,
        }

    @classmethod
    def from_mapping(cls, doc: object) -> DemandSpec:
        fields = (
            "streams",
            "n_periods",
            "period_ns",
            "start_ns",
            "base_level",
            "daily_amplitude",
            "weekly_amplitude",
            "weekend_effect",
            "zero_probability",
            "shift_period",
            "shift_factor",
            "label_delay_periods",
            "seed",
        )
        d = dict(require_keys(doc, fields, where="demand data", error=InvalidConfigError))
        if not isinstance(d["streams"], list):
            raise InvalidConfigError("streams must be a list")
        d["streams"] = tuple(d["streams"])
        d["period_ns"] = _config_ns(d["period_ns"], "period_ns")
        d["start_ns"] = _config_ns(d["start_ns"], "start_ns")
        return cls(**d)


def generate_demand(spec: DemandSpec) -> tuple[DemandSeries, ...]:
    """Draw one deterministic series per stream of ``spec`` (in stream order)."""
    n, period = spec.n_periods, spec.period_ns
    starts = spec.start_ns + np.arange(n, dtype=np.int64) * period
    day = (starts % NS_PER_DAY).astype(np.float64) / NS_PER_DAY
    dow = (starts // NS_PER_DAY + 3) % 7  # 1970-01-01 was a Thursday; 0 = Monday
    week = (dow + day) / 7.0
    shape = (
        spec.daily_amplitude * np.sin(2.0 * np.pi * (day - 0.25))
        + spec.weekly_amplitude * np.cos(2.0 * np.pi * week)
        + spec.weekend_effect * (dow >= 5)
    )
    shift = np.ones(n)
    if spec.shift_period is not None:
        shift[spec.shift_period :] = spec.shift_factor
    out: list[DemandSeries] = []
    for position, stream in enumerate(spec.streams):
        rng = np.random.default_rng(np.random.SeedSequence(spec.seed, spawn_key=(position,)))
        level = spec.base_level * math.exp(_STREAM_LEVEL_SPREAD * float(rng.standard_normal()))
        intensity = level * np.exp(shape) * shift
        active = rng.random(n) >= spec.zero_probability
        counts = rng.poisson(intensity)
        values = np.where(active, counts, 0).astype(np.float64)
        out.append(DemandSeries(stream, spec.start_ns, period, spec.label_delay_periods, values))
    return tuple(out)
