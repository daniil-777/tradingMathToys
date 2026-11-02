"""Scenario, risk-limit and session declarations for market replays.

A :class:`Scenario` fixes fees and timing assumptions; a :class:`SessionPlan` fixes the
trading clock (warm-up, entry cutoff, forced flattening, execution cutoff ``T`` and
observation drain ``D``). Both are plain validated values shared by every entry point.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Final

from financeai.core.clock import NS_PER_MS, NS_PER_S
from financeai.core.errors import InvalidConfigError
from financeai.domains.market.instruments import INSTRUMENTS, InstrumentSpec

MAX_REPLAY_NS: Final = 2**63 - 2
"""Largest usable replay timestamp; int64 max is reserved for the no-event sentinel."""


class TimingMode(IntEnum):
    """Which historical book an order meets when it arrives (HISTORICAL_REPLAY mode)."""

    EXCHANGE_CLOCK = 0
    """Book in effect at *exchange* time ``submit + latency`` (default, conservative)."""
    RECEIPT_CLOCK = 1
    """Book as *received* by local time ``submit + latency`` (simplified comparator)."""


def _check_ns(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_REPLAY_NS:
        raise InvalidConfigError(
            f"{name} must be a non-negative int64 nanosecond count below the no-event sentinel", context={"got": value}
        )


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    fees: Mapping[str, Decimal]
    """Taker fee as a decimal fraction, keyed by instrument key (``venue/symbol``)."""
    order_latency_ns: int
    """Submission -> exchange arrival."""
    ack_latency_ns: int | None = None
    """Exchange arrival -> fill report at the controller; defaults to ``order_latency_ns``."""
    timing: TimingMode = TimingMode.EXCHANGE_CLOCK
    consumption_ttl_ns: int = NS_PER_S
    """How long our own consumed liquidity stays unavailable at a price level."""
    depth_fraction: float = 1.0
    """Execution-depth haircut for sensitivity analysis, ``0 < f <= 1``: our IOC sweeps may take only
    ``floor(f * displayed_size / lot)`` whole lots of each visible level (before our own earlier
    consumption is netted). Decisions still observe the displayed book. 1.0 is the displayed depth."""
    fees_per_unit: Mapping[str, Decimal] = field(default_factory=dict)
    """Additional settlement-currency charge per native quantity, keyed by instrument."""
    _fees: Mapping[str, Decimal] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise InvalidConfigError("scenario name is required")
        fees: dict[str, Decimal] = {}
        for key, value in dict(self.fees).items():
            if not isinstance(value, Decimal) or not value.is_finite() or not Decimal(0) <= value < 1:
                raise InvalidConfigError("fee must be a Decimal in [0, 1)", context={"key": key})
            fees[key] = value
        object.__setattr__(self, "_fees", MappingProxyType(fees))
        object.__setattr__(self, "fees", self._fees)
        fixed = dict(self.fees_per_unit)
        for key, value in fixed.items():
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0 or not math.isfinite(float(value)):
                raise InvalidConfigError("fee per native quantity must be a non-negative Decimal", context={"key": key})
        object.__setattr__(self, "fees_per_unit", MappingProxyType(fixed))
        _check_ns("order_latency_ns", self.order_latency_ns)
        if self.ack_latency_ns is None:
            object.__setattr__(self, "ack_latency_ns", self.order_latency_ns)
        else:
            _check_ns("ack_latency_ns", self.ack_latency_ns)
        if self.order_latency_ns == 0 and self.ack_latency_ns == 0:
            # A zero round trip acknowledges an order at its own submission time, and the decision that the
            # acknowledgement triggers may resubmit (an exit that found no liquidity retries until flat) at
            # that same instant: the replay clock would never advance. Either latency alone may be 0.
            raise InvalidConfigError(
                "order and acknowledgement latency cannot both be 0: a zero round trip "
                "would re-decide at the same instant forever"
            )
        _check_ns("consumption_ttl_ns", self.consumption_ttl_ns)
        if not isinstance(self.timing, TimingMode):
            raise InvalidConfigError("timing must be a TimingMode")
        fraction: object = self.depth_fraction
        if isinstance(fraction, bool) or not isinstance(fraction, int | float) or not 0.0 < fraction <= 1.0:
            raise InvalidConfigError("depth_fraction must be a number in (0, 1]", context={"got": fraction})
        object.__setattr__(self, "depth_fraction", float(fraction))

    @property
    def ack_ns(self) -> int:
        assert self.ack_latency_ns is not None
        return self.ack_latency_ns

    def fee_for(self, instrument: InstrumentSpec) -> Decimal:
        try:
            return self.fees[instrument.key]
        except KeyError:
            raise InvalidConfigError(
                "scenario has no fee for instrument", context={"scenario": self.name, "instrument": instrument.key}
            ) from None

    def fee_per_unit_for(self, instrument: InstrumentSpec) -> Decimal:
        return self.fees_per_unit.get(instrument.key, Decimal(0))

    @classmethod
    def from_mapping(cls, doc: Mapping[str, Any]) -> Scenario:
        """Load explicit execution assumptions; decimal fee rates must be strings."""
        required = {"name", "fees", "order_latency_ns"}
        optional = {"ack_latency_ns", "timing", "consumption_ttl_ns", "depth_fraction", "fees_per_unit"}
        if not isinstance(doc, Mapping) or not required <= doc.keys() or doc.keys() - required - optional:
            raise InvalidConfigError("scenario keys differ", context={"required": sorted(required)})
        values = dict(doc)
        for name in ("fees", "fees_per_unit"):
            rates = values.get(name, {})
            if not isinstance(rates, Mapping) or any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in rates.items()
            ):
                raise InvalidConfigError("fee mappings require instrument keys and decimal strings")
            try:
                values[name] = {k: Decimal(v) for k, v in rates.items()}
            except InvalidOperation as exc:
                raise InvalidConfigError("malformed decimal fee") from exc
        if "timing" in values:
            try:
                values["timing"] = TimingMode[values["timing"]]
            except (KeyError, TypeError) as exc:
                raise InvalidConfigError("timing must name a TimingMode") from exc
        return cls(**values)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fees": {k: str(v) for k, v in self.fees.items()},
            "fees_per_unit": {k: str(v) for k, v in self.fees_per_unit.items()},
            "order_latency_ns": self.order_latency_ns,
            "ack_latency_ns": self.ack_ns,
            "timing": self.timing.name,
            "consumption_ttl_ns": self.consumption_ttl_ns,
            "depth_fraction": self.depth_fraction,
        }


DEMO_BASE_FEES: Final = {key: Decimal("0.0002") for key in INSTRUMENTS}
DEMO_HIGH_FEES: Final = {key: Decimal("0.0006") for key in INSTRUMENTS}


def demo_scenarios(instruments: Sequence[InstrumentSpec] | None = None) -> dict[str, Scenario]:
    """Illustrative cost/latency settings; supply Scenario directly for market-specific assumptions."""
    keys = [spec.key for spec in instruments] if instruments is not None else list(INSTRUMENTS)
    base = {key: Decimal("0.0002") for key in keys}
    high = {key: Decimal("0.0006") for key in keys}
    fast, slow = 3 * NS_PER_MS, 25 * NS_PER_MS
    return {
        "base": Scenario("base", base, fast),
        "higher_fees": Scenario("higher_fees", high, fast),
        "longer_delay": Scenario("longer_delay", base, slow),
        "combined_stress": Scenario("combined_stress", high, slow),
    }


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_abs_position_units: int = 10_000

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_abs_position_units, bool)
            or not isinstance(self.max_abs_position_units, int)
            or not 0 < self.max_abs_position_units <= (2**63 - 1) // 2
        ):
            raise InvalidConfigError(
                "max_abs_position_units must be an integer in [1, 2**62 - 1] so a full reversal fits int64"
            )


@dataclass(frozen=True, slots=True)
class SessionPlan:
    """Trading clock in local (receipt) epoch nanoseconds.

    ``start_ns <= entry_cutoff_ns <= flatten_start_ns < exec_cutoff_ns <= end_ns``. No
    exposure-increasing order after the entry cutoff; targets are forced to zero from
    ``flatten_start_ns``; no order may *arrive* after ``exec_cutoff_ns`` (T); fill
    reports are drained until ``end_ns`` (D).
    """

    start_ns: int
    entry_cutoff_ns: int
    flatten_start_ns: int
    exec_cutoff_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        values = (self.start_ns, self.entry_cutoff_ns, self.flatten_start_ns, self.exec_cutoff_ns, self.end_ns)
        for name, value in zip(
            ("start_ns", "entry_cutoff_ns", "flatten_start_ns", "exec_cutoff_ns", "end_ns"), values, strict=True
        ):
            _check_ns(name, value)
        if not (self.start_ns <= self.entry_cutoff_ns <= self.flatten_start_ns < self.exec_cutoff_ns <= self.end_ns):
            raise InvalidConfigError("session clocks are out of order", context={"clocks": values})

    @classmethod
    def for_window(
        cls,
        first_ns: int,
        last_ns: int,
        *,
        warmup_ns: int = 5 * 60 * NS_PER_S,
        entry_stop_ns: int = 15 * 60 * NS_PER_S,
        flatten_ns: int = 10 * 60 * NS_PER_S,
        exec_stop_ns: int = 5 * NS_PER_S,
    ) -> SessionPlan:
        """Plan for a data window: flat warm-up, then stop entries/flatten before the end."""
        return cls(
            start_ns=first_ns + warmup_ns,
            entry_cutoff_ns=last_ns - entry_stop_ns,
            flatten_start_ns=last_ns - flatten_ns,
            exec_cutoff_ns=last_ns - exec_stop_ns,
            end_ns=last_ns,
        )
