"""Taker IOC execution against one side of a historical book snapshot.

Historical snapshots do not reflect our own trades, so liquidity we take is recorded in
a small per-(venue, side) *consumption* table and subtracted from later sweeps at the
same price until a time-to-live expires (the assumed replenishment time). This prevents
a strategy from repeatedly harvesting the same stale quote.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from financeai.core.jit import njit

CONSUMPTION_SLOTS: Final = 32


@njit()
def _consume(
    c_px: NDArray[np.int32],
    c_amt: NDArray[np.int64],
    c_exp: NDArray[np.int64],
    price: int,
    amount: int,
    expiry: int,
    now: int,
) -> None:
    free = -1
    soonest = 0
    for s in range(c_px.shape[0]):
        active = c_amt[s] > 0 and c_exp[s] > now
        if active and c_px[s] == price:
            c_amt[s] += amount
            c_exp[s] = expiry
            return
        if not active and free < 0:
            free = s
        if c_exp[s] < c_exp[soonest]:
            soonest = s
    slot = free if free >= 0 else soonest
    c_px[slot] = price
    c_amt[slot] = amount
    c_exp[slot] = expiry


@njit()
def consumed_at(
    c_px: NDArray[np.int32], c_amt: NDArray[np.int64], c_exp: NDArray[np.int64], price: int, now: int
) -> int:
    total = 0
    for s in range(c_px.shape[0]):
        if c_amt[s] > 0 and c_exp[s] > now and c_px[s] == price:
            total += c_amt[s]
    return total


@njit()
def sweep(
    side: int,
    qty: int,
    limit_ticks: int,
    px: NDArray[np.int32],
    sz: NDArray[np.int32],
    n_levels: int,
    lot: int,
    depth_fraction: float,
    c_px: NDArray[np.int32],
    c_amt: NDArray[np.int64],
    c_exp: NDArray[np.int64],
    now: int,
    ttl: int,
    out_px: NDArray[np.int32],
    out_qty: NDArray[np.int64],
) -> int:
    """Fill up to ``qty`` USD from best to worst level within the limit; IOC remainder lapses.

    ``px``/``sz`` are the *opposite* side's levels, best first (asks for a buy, bids for a
    sell). Returns the number of per-level fills written to ``out_px``/``out_qty``.

    ``depth_fraction`` (``0 < f <= 1``) is an execution-depth haircut: of each displayed level
    only ``floor(f * size / lot)`` whole lots may be taken, *before* our own earlier consumption
    at that price is netted. It is evaluated as ``floor(f * size)`` followed by whole lots (the
    same number with a single float rounding, so any ``f < 1`` takes strictly less than a
    displayed level); ``f == 1`` skips the haircut and is the plain sweep bit for bit.
    """
    remaining = qty
    n_out = 0
    for k in range(n_levels):
        if remaining <= 0:
            break
        price = px[k]
        if (side > 0 and price > limit_ticks) or (side < 0 and price < limit_ticks):
            break
        shown = int(sz[k])
        if depth_fraction < 1.0:
            shown = (int(depth_fraction * shown) // lot) * lot
        available = shown - consumed_at(c_px, c_amt, c_exp, price, now)
        available = (available // lot) * lot
        if available <= 0:
            continue
        take = remaining if remaining < available else available
        out_px[n_out] = price
        out_qty[n_out] = take
        n_out += 1
        remaining -= take
        _consume(c_px, c_amt, c_exp, price, take, now + ttl, now)
    return n_out
