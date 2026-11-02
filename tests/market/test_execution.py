"""Taker IOC sweep against one book side, with own-consumed-liquidity tracking."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from financeai.domains.market.execution import CONSUMPTION_SLOTS, sweep

BUY, SELL = 1, -1


def side_levels(levels: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, int]:
    px = np.zeros(10, np.int32)
    sz = np.zeros(10, np.int32)
    for k, (p, s) in enumerate(levels):
        px[k], sz[k] = p, s
    return px, sz, len(levels)


def empty_consumption() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros(CONSUMPTION_SLOTS, np.int32),
        np.zeros(CONSUMPTION_SLOTS, np.int64),
        np.zeros(CONSUMPTION_SLOTS, np.int64),
    )


def run(
    side: int,
    qty: int,
    limit: int,
    levels: list[tuple[int, int]],
    lot: int = 100,
    now: int = 0,
    ttl: int = 1_000,
    cons: tuple[np.ndarray, ...] | None = None,
    fraction: float = 1.0,
) -> list[tuple[int, int]]:
    px, sz, n = side_levels(levels)
    c_px, c_amt, c_exp = cons if cons is not None else empty_consumption()
    out_px = np.zeros(10, np.int32)
    out_qty = np.zeros(10, np.int64)
    k = sweep(side, qty, limit, px, sz, n, lot, fraction, c_px, c_amt, c_exp, now, ttl, out_px, out_qty)
    return [(int(out_px[i]), int(out_qty[i])) for i in range(k)]


ASKS = [(100, 1_000), (101, 2_000), (102, 5_000)]
BIDS = [(99, 1_000), (98, 2_000), (97, 5_000)]


def test_buy_sweeps_asks_up_to_the_limit() -> None:
    assert run(BUY, 2_500, 101, ASKS) == [(100, 1_000), (101, 1_500)]


def test_buy_with_limit_below_best_ask_does_not_fill() -> None:
    assert run(BUY, 1_000, 99, ASKS) == []


def test_buy_larger_than_visible_depth_fills_partially() -> None:
    assert run(BUY, 10_000, 10**6, ASKS) == [(100, 1_000), (101, 2_000), (102, 5_000)]


def test_sell_sweeps_bids_down_to_the_limit() -> None:
    assert run(SELL, 2_500, 98, BIDS) == [(99, 1_000), (98, 1_500)]
    assert run(SELL, 500, 100, BIDS) == []


def test_fills_are_rounded_down_to_the_lot() -> None:
    assert run(BUY, 1_000, 100, [(100, 150)], lot=100) == [(100, 100)]


def test_consumed_liquidity_is_unavailable_until_it_expires() -> None:
    cons = empty_consumption()
    assert run(BUY, 1_000, 101, ASKS, now=0, ttl=1_000, cons=cons) == [(100, 1_000)]
    # Same historical book shortly after: our own fill already took level 100.
    assert run(BUY, 1_000, 101, ASKS, now=500, ttl=1_000, cons=cons) == [(101, 1_000)]
    # After the time-to-live the level is assumed replenished.
    assert run(BUY, 1_000, 101, ASKS, now=1_001, ttl=1_000, cons=cons) == [(100, 1_000)]


def test_consumption_is_tracked_per_price_and_accumulates() -> None:
    cons = empty_consumption()
    run(BUY, 600, 100, ASKS, now=0, cons=cons)
    assert run(BUY, 1_000, 100, ASKS, now=10, cons=cons) == [(100, 400)]
    assert run(BUY, 1_000, 100, ASKS, now=20, cons=cons) == []


def test_depth_haircut_takes_a_fraction_of_every_visible_level() -> None:
    # half of the displayed 1,000 / 2,000 / 5,000 USD on each side
    assert run(BUY, 10_000, 10**6, ASKS, fraction=0.5) == [(100, 500), (101, 1_000), (102, 2_500)]
    assert run(SELL, 10_000, 0, BIDS, fraction=0.5) == [(99, 500), (98, 1_000), (97, 2_500)]
    assert run(BUY, 2_500, 101, ASKS, fraction=0.5) == [(100, 500), (101, 1_000)]  # the limit still binds


def test_depth_haircut_floors_to_whole_lots() -> None:
    assert run(BUY, 1_000, 100, [(100, 1_000)], fraction=0.33) == [(100, 300)]  # 3.3 lots -> 3 lots
    assert run(BUY, 1_000, 100, [(100, 1_000)], fraction=0.05) == []  # half a lot is nothing


def test_depth_haircut_applies_before_own_consumption_is_netted() -> None:
    cons = empty_consumption()
    assert run(BUY, 300, 100, ASKS, now=0, cons=cons, fraction=0.5) == [(100, 300)]
    # 1,000 displayed -> 500 takeable; our earlier 300 leaves 200 (not 1,000 - 300 = 700)
    assert run(BUY, 1_000, 100, ASKS, now=10, cons=cons, fraction=0.5) == [(100, 200)]


def _legacy_sweep(side: int, qty: int, limit: int, levels: list[tuple[int, int]], lot: int) -> list[tuple[int, int]]:
    """The sweep before the depth haircut existed (fresh consumption table)."""
    out: list[tuple[int, int]] = []
    remaining = qty
    for price, size in levels:
        if remaining <= 0 or (side > 0 and price > limit) or (side < 0 and price < limit):
            break
        available = (size // lot) * lot
        if available > 0:
            take = min(remaining, available)
            out.append((price, take))
            remaining -= take
    return out


BOOK_SIDE = st.lists(st.integers(1, 60), min_size=1, max_size=10)


@given(
    sizes=BOOK_SIDE,
    lot=st.sampled_from([10, 100]),
    qty_lots=st.integers(1, 700),
    reach=st.integers(0, 9),
    odd=st.integers(0, 9),
)
@settings(max_examples=200, deadline=None)
def test_full_depth_fraction_is_the_unhaircut_sweep(
    sizes: list[int], lot: int, qty_lots: int, reach: int, odd: int
) -> None:
    levels = [(100 + k, n * lot + (odd if k == 0 else 0)) for k, n in enumerate(sizes)]  # one odd-sized level
    limit = 100 + reach
    assert run(BUY, qty_lots * lot, limit, levels, lot=lot) == _legacy_sweep(BUY, qty_lots * lot, limit, levels, lot)


@given(
    sizes=BOOK_SIDE, lot=st.sampled_from([10, 100]), fraction=st.floats(0.0, 1.0, exclude_min=True, exclude_max=True)
)
@settings(max_examples=200, deadline=None)
def test_partial_depth_fraction_strictly_reduces_every_level(sizes: list[int], lot: int, fraction: float) -> None:
    levels = [(100 + k, n * lot) for k, n in enumerate(sizes)]
    everything = sum(n * lot for n in sizes) + lot
    full = run(BUY, everything, 10**6, levels, lot=lot)
    cut = dict(run(BUY, everything, 10**6, levels, lot=lot, fraction=fraction))
    assert sum(q for _, q in full) == everything - lot  # the whole displayed book
    for price, size in levels:
        taken = cut.get(price, 0)
        assert taken == (int(fraction * size) // lot) * lot  # floor(f * size / lot) whole lots
        assert taken < size  # strictly less than the displayed level
