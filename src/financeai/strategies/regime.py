"""Causal activity (regime) measures used to gate signals.

Round-1 iteration showed the lead-lag edge concentrated in active periods while quiet
periods bled fees, so signals can be gated on the leader's recent traded volume. Every
measure is one float64 per stream event and uses only events with index <= e.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError
from financeai.core.jit import njit
from financeai.domains.market.engine import PreparedMarket
from financeai.domains.market.stream import BOOK_EVENT, TRADE_EVENT


@njit()
def _decayed_volume(
    recv: NDArray[np.int64],
    venue_ev: NDArray[np.int8],
    kind_ev: NDArray[np.int8],
    row_ev: NDArray[np.int64],
    size: NDArray[np.int32],
    target: int,
    tau: float,
    out: NDArray[np.float64],
) -> None:
    level = 0.0
    last_t = recv[0] if recv.shape[0] else 0
    for i in range(recv.shape[0]):
        t = recv[i]
        level *= np.exp(-(t - last_t) / tau)
        last_t = t
        if venue_ev[i] == target and kind_ev[i] == TRADE_EVENT:
            level += float(size[row_ev[i]])
        out[i] = level


def decayed_volume(prep: PreparedMarket, *, venue: int, tau_ns: int) -> NDArray[np.float64]:
    """Exponentially decayed traded USD (both aggressor sides) of ``venue``."""
    if tau_ns <= 0:
        raise InvalidConfigError("tau must be positive")
    s = prep.stream
    out = np.empty(len(s), np.float64)
    _decayed_volume(s.recv_ns, s.venue, s.kind, s.row, prep.data.venues[venue].trades.sz, venue, float(tau_ns), out)
    return out


def apply_gate(signal: NDArray[np.float64], prep: PreparedMarket, gate: Mapping[str, Any]) -> NDArray[np.float64]:
    """Zero the signal wherever the venue's decayed volume is below ``min_units``."""
    if set(gate) != {"venue", "tau_ms", "min_units"}:
        raise InvalidConfigError("gate needs exactly venue, tau_ms, min_units", context={"got": sorted(gate)})
    keys = [i.key for i in prep.data.instruments]
    if gate["venue"] not in keys:
        raise InvalidConfigError("unknown gate venue", context={"got": gate["venue"]})
    tau_ms, min_units = gate["tau_ms"], gate["min_units"]
    if isinstance(tau_ms, bool) or not isinstance(tau_ms, int) or tau_ms <= 0:
        raise InvalidConfigError("gate tau_ms must be a positive integer")
    if isinstance(min_units, bool) or not isinstance(min_units, int | float) or min_units < 0:
        raise InvalidConfigError("gate min_units must be non-negative")
    vol = decayed_volume(prep, venue=keys.index(gate["venue"]), tau_ns=tau_ms * 1_000_000)
    return np.ascontiguousarray(np.where(vol >= min_units, signal, 0.0))


@njit()
def _fresh_mask(
    venue_ev: NDArray[np.int8],
    kind_ev: NDArray[np.int8],
    row_ev: NDArray[np.int64],
    recv: NDArray[np.int64],
    venue: int,
    book_recv: NDArray[np.int64],
    max_age: int,
    mask: NDArray[np.bool_],
) -> None:
    """Clears ``mask`` where the venue has no observed book row yet or that row (running
    maximum, as in ``observed_rows``) was received more than ``max_age`` ago."""
    current = -1
    for e in range(venue_ev.shape[0]):
        if venue_ev[e] == venue and kind_ev[e] == BOOK_EVENT:
            r = row_ev[e]
            if r > current:
                current = r
        if current < 0 or recv[e] - book_recv[current] > max_age:
            mask[e] = False


def freshness_mask(prep: PreparedMarket, *, venues: tuple[int, ...], max_age_ns: int) -> NDArray[np.bool_]:
    """True where every listed venue's last *received* book update is at most ``max_age_ns`` old."""
    s = prep.stream
    mask = np.ones(len(s), bool)
    bounds = np.iinfo(np.int64)
    # Ages are int64, so ``age <= x`` is ``age <= floor(x)`` for any real ``x`` (a negative fraction
    # admits nothing) and clamping to the int64 range leaves every comparison unchanged.
    max_age = max(min(math.floor(max_age_ns), int(bounds.max)), int(bounds.min))
    for v in venues:
        _fresh_mask(s.venue, s.kind, s.row, s.recv_ns, v, prep.book_recv[v], max_age, mask)
    return mask


@njit()
def _ewma_rms_prev(values: NDArray[np.float64], recv: NDArray[np.int64], tau: float, out: NDArray[np.float64]) -> None:
    level = np.nan
    last_t = 0
    for i in range(values.shape[0]):
        x = values[i]
        out[i] = np.sqrt(level) if level == level else np.nan
        if x == x:
            sq = x * x
            if level == level:
                w = np.exp(-(recv[i] - last_t) / tau)
                level = w * level + (1.0 - w) * sq
            else:
                level = sq
            last_t = recv[i]


def normalise(signal: NDArray[np.float64], prep: PreparedMarket, tau_ns: int) -> NDArray[np.float64]:
    """Signal divided by the causal EWMA root-mean-square of its *previous* values (0 if undefined)."""
    if tau_ns <= 0:
        raise InvalidConfigError("normalisation tau must be positive")
    rms = np.empty(len(signal), np.float64)
    _ewma_rms_prev(np.ascontiguousarray(signal, dtype=np.float64), prep.stream.recv_ns, float(tau_ns), rms)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = signal / rms
    return np.ascontiguousarray(np.where(np.isfinite(z), z, 0.0))
