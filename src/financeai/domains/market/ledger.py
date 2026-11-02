"""Currency-aware cash flows for spot, linear, and inverse contracts.

Realized PnL is available only when the account is flat. Different settlement
currencies remain separate; no implicit exchange rate or currency summation is used.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidDataError
from financeai.core.jit import njit
from financeai.domains.market.instruments import InstrumentSpec


@njit()
def fill_cash(
    side: int,
    qty_units: int,
    price: float,
    fee: float,
    inverse: bool = True,
    factor: float = 1.0,
    fixed_fee: float = 0.0,
) -> float:
    """Cash-flow contribution in settlement currency; fixed_fee is per stored quantity unit."""
    if inverse:
        return (side - fee) * (qty_units * factor) / price - qty_units * fixed_fee
    return (-side - fee) * (qty_units * factor) * price - qty_units * fixed_fee


_FILL_COLUMNS = (
    ("order_id", np.int64),
    ("venue", np.int8),
    ("side", np.int8),
    ("px_ticks", np.int32),
    ("qty_units", np.int64),
    ("submit_ns", np.int64),
    ("arrival_ns", np.int64),
    ("ack_ns", np.int64),
    ("cash", np.float64),
)


@dataclass(frozen=True, slots=True)
class Fills:
    """Columnar fill records in true (exchange-arrival) order; one row per price level."""

    order_id: NDArray[np.int64]
    venue: NDArray[np.int8]
    side: NDArray[np.int8]
    px_ticks: NDArray[np.int32]
    qty_units: NDArray[np.int64]
    submit_ns: NDArray[np.int64]
    arrival_ns: NDArray[np.int64]
    ack_ns: NDArray[np.int64]
    cash: NDArray[np.float64]

    def __post_init__(self) -> None:
        n = self.order_id.shape[0]
        for name, dtype in _FILL_COLUMNS:
            arr = getattr(self, name)
            if arr.dtype != np.dtype(dtype) or arr.ndim != 1 or arr.shape[0] != n:
                raise InvalidDataError(
                    "fill column has wrong dtype/shape", context={"column": name, "dtype": str(arr.dtype)}
                )
            arr.setflags(write=False)
        if n and (np.any(self.qty_units <= 0) or np.any(np.abs(self.side) != 1)):
            raise InvalidDataError("fills need positive quantity and side +/-1")

    def __len__(self) -> int:
        return int(self.order_id.shape[0])

    @classmethod
    def empty(cls) -> Fills:
        return cls(
            order_id=np.zeros(0, np.int64),
            venue=np.zeros(0, np.int8),
            side=np.zeros(0, np.int8),
            px_ticks=np.zeros(0, np.int32),
            qty_units=np.zeros(0, np.int64),
            submit_ns=np.zeros(0, np.int64),
            arrival_ns=np.zeros(0, np.int64),
            ack_ns=np.zeros(0, np.int64),
            cash=np.zeros(0, np.float64),
        )


@dataclass(frozen=True, slots=True)
class Settlement:
    final_position_units: tuple[int, ...]
    max_abs_position_units: tuple[int, ...]
    cash: tuple[float, ...]
    fees: tuple[float, ...]
    traded_units: tuple[int, ...]
    n_fills: tuple[int, ...]
    valid: bool
    reason: str | None
    """``None`` when valid, else ``"cap_exceeded"`` or ``"not_flat"``."""
    currencies: tuple[str, ...] = ()

    @property
    def currency(self) -> str | None:
        distinct = set(self.currencies)
        return next(iter(distinct)) if len(distinct) == 1 else None

    @property
    def pnl_by_currency(self) -> dict[str, float] | None:
        if not self.valid:
            return None
        currencies = self.currencies or ("UNSPECIFIED",) * len(self.cash)
        return {
            currency: math.fsum(value for value, unit in zip(self.cash, currencies, strict=True) if unit == currency)
            for currency in sorted(set(currencies))
        }

    @property
    def net_pnl(self) -> float | None:
        return math.fsum(self.cash) if self.valid and len(set(self.currencies)) <= 1 else None


def _check_venues(fills: Fills, n_venues: int) -> None:
    if len(fills) and (int(fills.venue.min()) < 0 or int(fills.venue.max()) >= n_venues):
        raise InvalidDataError("fill venue index out of range")


def settle(
    fills: Fills,
    instruments: Sequence[InstrumentSpec],
    fees: Sequence[Decimal],
    *,
    cap_units: int | Sequence[int],
    fees_per_unit: Sequence[Decimal] | None = None,
) -> Settlement:
    """Per-venue positions, cap excursions and the terminal score (float, compensated)."""
    n_venues = len(instruments)
    if len(fees) != n_venues:
        raise InvalidDataError("need one fee per instrument")
    fixed = tuple(fees_per_unit) if fees_per_unit is not None else (Decimal(0),) * n_venues
    if len(fixed) != n_venues:
        raise InvalidDataError("need one fixed fee per instrument")
    caps = (cap_units,) * n_venues if isinstance(cap_units, int) and not isinstance(cap_units, bool) else cap_units
    if (
        not isinstance(caps, Sequence)
        or len(caps) != n_venues
        or any(isinstance(cap, bool) or not isinstance(cap, int) or not 0 <= cap <= (2**63 - 1) // 2 for cap in caps)
    ):
        raise InvalidDataError("need one non-negative int64-safe position cap per instrument")
    _check_venues(fills, n_venues)
    final: list[int] = []
    max_abs: list[int] = []
    cash: list[float] = []
    fee_cash: list[float] = []
    traded: list[int] = []
    counts: list[int] = []
    for v in range(n_venues):
        mask = fills.venue == v
        signed = fills.side[mask].astype(np.int64) * fills.qty_units[mask]
        path = np.cumsum(signed)
        previous = np.concatenate((np.zeros(1, np.int64), path[:-1])) if path.size else path
        if np.any((signed > 0) & (path < previous)) or np.any((signed < 0) & (path > previous)):
            raise InvalidDataError("position accumulation overflows int64")
        final.append(int(path[-1]) if path.size else 0)
        max_abs.append(max(int(path.max()), -int(path.min())) if path.size else 0)
        cash.append(math.fsum(fills.cash[mask].tolist()))
        spec = instruments[v]
        price = fills.px_ticks[mask].astype(np.float64) * spec.tick_float
        value = float(fees[v]) * (fills.qty_units[mask] * spec.cash_factor)
        charges = value / price if spec.inverse else value * price
        charges += fills.qty_units[mask] * (float(spec.quantity_step) * float(fixed[v]))
        fee_cash.append(math.fsum(charges.tolist()))
        traded.append(sum(fills.qty_units[mask].tolist()))
        counts.append(int(mask.sum()))
    reason: str | None = None
    if any(m > cap for m, cap in zip(max_abs, caps, strict=True)):
        reason = "cap_exceeded"
    elif any(final):
        reason = "not_flat"
    elif any(
        not spec.shortable
        and np.any(np.cumsum(fills.side[fills.venue == v].astype(np.int64) * fills.qty_units[fills.venue == v]) < 0)
        for v, spec in enumerate(instruments)
    ):
        reason = "short_not_allowed"
    return Settlement(
        tuple(final),
        tuple(max_abs),
        tuple(cash),
        tuple(fee_cash),
        tuple(traded),
        tuple(counts),
        valid=reason is None,
        reason=reason,
        currencies=tuple(str(spec.settlement_currency) for spec in instruments),
    )


def exact_score(
    fills: Fills,
    instruments: Sequence[InstrumentSpec],
    fees: Sequence[Decimal],
    fees_per_unit: Sequence[Decimal] | None = None,
) -> Fraction | None:
    """Independent rational oracle; refuse aggregation across different currencies."""
    _check_venues(fills, len(instruments))
    if len({spec.settlement_currency for spec in instruments}) > 1:
        raise InvalidDataError("exact_score requires a common settlement currency")
    if len(fees) != len(instruments) or (fees_per_unit is not None and len(fees_per_unit) != len(instruments)):
        raise InvalidDataError("need one fee per instrument")
    positions = [0] * len(instruments)
    total = Fraction(0)
    for v, s, px, qty in zip(
        fills.venue.tolist(), fills.side.tolist(), fills.px_ticks.tolist(), fills.qty_units.tolist(), strict=True
    ):
        spec = instruments[v]
        price = Fraction(px) * Fraction(spec.tick_size)
        quantity = qty * Fraction(spec.quantity_step)
        exposure = quantity * Fraction(spec.contract_multiplier)
        if spec.inverse:
            total += (Fraction(s) - Fraction(fees[v])) * exposure / price
        else:
            total += (-Fraction(s) - Fraction(fees[v])) * exposure * price
        if fees_per_unit is not None:
            total -= quantity * Fraction(fees_per_unit[v])
        positions[v] += s * qty
    return total if not any(positions) else None
