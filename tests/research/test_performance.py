"""Replay-result statistics: fixed-grid marked increments (hand-worked with exact fractions and
reconciled with the ledger on synthetic replays), paired stationary-bootstrap comparisons, HAC
means and the selection diagnostics with their caveats. Values are invented; no market data is read."""

from __future__ import annotations

import dataclasses
import math
from decimal import Decimal
from fractions import Fraction
from statistics import NormalDist
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.core.serialization import canonical_json
from financeai.domains.market.engine import (
    Orders,
    PreparedMarket,
    ReplayResult,
    ThresholdParams,
    run_replay,
)
from financeai.domains.market.instruments import DEMO_INVERSE_A, DEMO_INVERSE_B
from financeai.domains.market.ledger import Fills, fill_cash, settle
from financeai.domains.market.metrics import marked_value
from financeai.domains.market.scenario import DEMO_BASE_FEES, RiskLimits, Scenario, SessionPlan
from financeai.domains.market.stream import MarketData
from financeai.research.performance import (
    MAX_BOOTSTRAP_ELEMENTS,
    SELECTION_CAVEATS,
    VALUATIONS,
    AccountIncrements,
    account_increments,
    common_bootstrap_indices,
    hac_mean,
    increment_matrix,
    newey_west_lags,
    paired_block_bootstrap,
    selection_diagnostics,
)
from financeai.research.stats import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    newey_west_se,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    return_moments,
)

from ..market.synthetic import Row, ladder, random_market, tapes_from_rows

INSTRUMENTS = (DEMO_INVERSE_A, DEMO_INVERSE_B)
FEES = (Decimal("0.00005"), Decimal("0.0001"))
MS = 1_000_000
PHI = NormalDist().cdf


def two_sided_p(t: float) -> float:
    """``2 Phi(-|t|)`` through the complementary error function (keeps the far tail)."""
    return math.erfc(abs(t) / math.sqrt(2.0))


FillRow = tuple[int, int, int, int, int]
"""(arrival_ms, venue, side, px_ticks, qty_units); ticks are 0.5 USD."""


def hand_result(rows: list[FillRow], *, latency_ms: int = 5, cap: int = 1_000_000) -> ReplayResult:
    """A replay result assembled from hand-written fills (one order per fill)."""
    n = len(rows)
    arrival = np.array([r[0] * MS for r in rows], np.int64)
    ack = arrival + latency_ms * MS
    venue = np.array([r[1] for r in rows], np.int8)
    side = np.array([r[2] for r in rows], np.int8)
    px = np.array([r[3] for r in rows], np.int32)
    qty = np.array([r[4] for r in rows], np.int64)
    cash = np.array(
        [
            fill_cash(int(s), int(q), float(p) * float(INSTRUMENTS[v].tick_size), float(FEES[v]))
            for v, s, p, q in zip(venue, side, px, qty, strict=True)
        ],
        np.float64,
    )
    ids = np.arange(n, dtype=np.int64)
    submit = arrival - latency_ms * MS
    fills = Fills(
        order_id=ids,
        venue=venue,
        side=side,
        px_ticks=px,
        qty_units=qty,
        submit_ns=submit,
        arrival_ns=arrival,
        ack_ns=ack,
        cash=cash,
    )
    orders = Orders(
        order_id=ids.copy(),
        venue=venue.copy(),
        side=side.copy(),
        qty_units=qty.copy(),
        limit_ticks=px.astype(np.int64),
        submit_ns=submit.copy(),
        arrival_ns=arrival.copy(),
        ack_ns=ack.copy(),
        filled_units=qty.copy(),
        reason=np.ones(n, np.int8),
        book_row=np.zeros(n, np.int64),
    )
    settlement = settle(fills, INSTRUMENTS, FEES, cap_units=cap)
    return ReplayResult(fills=fills, orders=orders, settlement=settlement, decisions=n)


def hand_market(*, clear_at_8ms: bool = False) -> PreparedMarket:
    """Books received at 0, 5 and 12 ms (times in us; prices in 0.5 USD ticks).

    venue_a: bid/ask 20,000/20,001 then 20,100/20,101; venue_b: 20,000/20,001 then 19,900/19,901.
    With ``clear_at_8ms`` the venue_a book received at 8 ms has no bids (unusable while held).
    """
    b0, a0 = ladder(40_000, 40_002, 10_000_000)
    b1, a1 = ladder(40_200, 40_202, 10_000_000)
    venue_a: list[Row] = [(0, 0, b0, a0), (5_000, 4_000, b1, a1)]
    if clear_at_8ms:
        venue_a.append((8_000, 7_500, [], a1))
    venue_a.append((12_000, 11_000, b1, a1))
    d0, da0 = ladder(40_000, 40_002, 1_000_000)
    d1, da1 = ladder(39_800, 39_802, 1_000_000)
    venue_b: list[Row] = [(0, 0, d0, da0), (5_000, 4_000, d1, da1), (12_000, 11_000, d1, da1)]
    data = MarketData((tapes_from_rows(DEMO_INVERSE_A, venue_a), tapes_from_rows(DEMO_INVERSE_B, venue_b)))
    return PreparedMarket.build(data)


LONG_FILLS: list[FillRow] = [(2, 0, 1, 40_002, 1_000_000), (10, 0, -1, 40_200, 1_000_000)]
"""venue_a: buy 1M USD at the 20,001 ask (arrives 2 ms), sell at the 20,100 bid (arrives 10 ms)."""
SHORT_FILLS: list[FillRow] = [(1, 1, -1, 40_000, 500_000), (9, 1, 1, 39_802, 500_000)]
"""venue_b: sell 500k USD at the 20,000 bid (arrives 1 ms), buy back at the 19,901 ask (arrives 9 ms)."""


def approx_pnl(value: Fraction, *, flow_pnl: float = 0.0) -> Any:
    """Exact BTC value within 1e-12 relative, plus an absolute floor of a few ulps of the gross flows that
    were differenced to obtain it (a 1M USD position is ~50 BTC of cash against ~50 BTC of inventory)."""
    return pytest.approx(float(value), rel=1e-12, abs=1e-15 + 4e-16 * flow_pnl)


# ---------------------------------------------------------------------------
# Marked fixed-grid path: hand-worked examples
# ---------------------------------------------------------------------------


def test_long_round_trip_marked_at_mid_and_exit_prices() -> None:
    res = hand_result(LONG_FILLS)
    prep = hand_market()
    f = Fraction(FEES[0])
    c1 = (1 - f) * 1_000_000 / Fraction(20_001)
    c2 = c1 - (1 + f) * 1_000_000 / Fraction(20_100)
    assert res.settlement.net_pnl == pytest.approx(float(c2), rel=1e-13)
    expected = {
        # grid 0, 4, 8, 12 ms: flat, long on the 0 ms book, long on the 5 ms book, flat again
        "mid": [Fraction(0), c1 - 1_000_000 / Fraction(40_001, 2), c1 - 1_000_000 / Fraction(40_201, 2), c2],
        "exit": [Fraction(0), c1 - 1_000_000 / Fraction(20_000), c1 - 1_000_000 / Fraction(20_100), c2],
    }
    for valuation in VALUATIONS:
        path = account_increments(res, prep, 4 * MS, valuation)
        assert isinstance(path, AccountIncrements)
        assert path.times_ns.tolist() == [0, 4 * MS, 8 * MS, 12 * MS]
        assert path.step_ns == 4 * MS and path.valuation == valuation
        assert path.position_units.tolist() == [[0, 0], [1_000_000, 0], [1_000_000, 0], [0, 0]]
        want = expected[valuation]
        for got, exact in zip(path.marked_pnl.tolist(), want, strict=True):
            assert got == approx_pnl(exact)
        for got, exact in zip(path.cash.tolist(), [Fraction(0), c1, c1, c2], strict=True):
            assert got == approx_pnl(exact)
        increments = [want[1] - want[0], want[2] - want[1], want[3] - want[2]]
        for got, exact in zip(path.increments_pnl.tolist(), increments, strict=True):
            assert got == approx_pnl(exact, flow_pnl=50.0)
        assert path.n_intervals == 3 and path.n_unmarked == 0
        assert path.net_pnl == res.settlement.net_pnl
        assert path.final_pnl == path.marked_pnl[-1]
        for arr in (path.times_ns, path.position_units, path.cash, path.marked_pnl, path.increments_pnl):
            assert not arr.flags.writeable
    # Right after a buy at the ask the exit valuation is minus the fee paid minus the spread (the position
    # could only be sold back at the bid); the mid valuation is better by half a spread.
    exit_path = account_increments(res, prep, 4 * MS, "exit")
    fee_and_spread = -float(f) * 1_000_000 / 20_001 - 1_000_000 * (1 / 20_000 - 1 / 20_001)
    assert exit_path.marked_pnl[1] == pytest.approx(fee_and_spread, rel=1e-9)
    mid_path = account_increments(res, prep, 4 * MS, "mid")
    assert np.all(exit_path.marked_pnl[1:3] < mid_path.marked_pnl[1:3])


def test_short_round_trip_is_marked_at_the_ask_under_the_exit_valuation() -> None:
    res = hand_result(SHORT_FILLS)
    prep = hand_market()
    f = Fraction(FEES[1])
    c1 = -(1 + f) * 500_000 / Fraction(20_000)
    c2 = c1 + (1 - f) * 500_000 / Fraction(19_901)
    mid = account_increments(res, prep, 4 * MS, "mid")
    exit_ = account_increments(res, prep, 4 * MS, "exit")
    assert mid.position_units.tolist() == [[0, 0], [0, -500_000], [0, -500_000], [0, 0]]
    assert mid.marked_pnl[1] == approx_pnl(c1 + 500_000 / Fraction(40_001, 2))
    assert exit_.marked_pnl[1] == approx_pnl(c1 + 500_000 / Fraction(20_001))
    assert mid.marked_pnl[2] == approx_pnl(c1 + 500_000 / Fraction(39_801, 2))
    assert exit_.marked_pnl[2] == approx_pnl(c1 + 500_000 / Fraction(19_901))
    assert mid.marked_pnl[3] == approx_pnl(c2) and exit_.marked_pnl[3] == approx_pnl(c2)
    assert np.all(exit_.marked_pnl[1:3] < mid.marked_pnl[1:3])
    assert math.fsum(exit_.increments_pnl.tolist()) == approx_pnl(c2)


def test_a_short_held_through_an_unusable_book_is_unmarked_under_both_valuations() -> None:
    # venue_a short: sell 500k USD at the 20,000 bid (arrives 1 ms), buy back at the 20,101 ask (arrives 9 ms).
    # The book received at 8 ms has asks but no bids: a one-sided book is unusable whichever side the
    # valuation would read, so the short is unmarked there exactly like a long.
    res = hand_result([(1, 0, -1, 40_000, 500_000), (9, 0, 1, 40_202, 500_000)])
    prep = hand_market(clear_at_8ms=True)
    f = Fraction(FEES[0])
    c1 = -(1 + f) * 500_000 / Fraction(20_000)
    c2 = c1 + (1 - f) * 500_000 / Fraction(20_101)
    for valuation in VALUATIONS:
        path = account_increments(res, prep, 4 * MS, valuation)
        assert path.position_units.tolist() == [[0, 0], [-500_000, 0], [-500_000, 0], [0, 0]]
        assert np.isnan(path.marked_pnl[2]) and np.isfinite(path.marked_pnl[[0, 1, 3]]).all()
        assert np.isnan(path.increments_pnl[1]) and np.isnan(path.increments_pnl[2])
        assert path.n_unmarked == 1
        priced_at = Fraction(40_001, 2) if valuation == "mid" else Fraction(20_001)  # mid, or the ask for a short
        assert path.marked_pnl[1] == approx_pnl(c1 + 500_000 / priced_at, flow_pnl=25.0)
        assert path.marked_pnl[3] == approx_pnl(c2, flow_pnl=25.0)


def test_fills_arriving_exactly_at_a_grid_time_count_there_including_the_last_grid_time() -> None:
    # The buy arrives exactly at the 4 ms grid time and the sell exactly at the last grid time (12 ms, the last
    # receipt): both belong to the grid time they land on, so the grid covers the account.
    res = hand_result([(4, 0, 1, 40_002, 1_000_000), (12, 0, -1, 40_200, 1_000_000)])
    prep = hand_market()
    f = Fraction(FEES[0])
    c1 = (1 - f) * 1_000_000 / Fraction(20_001)
    for valuation in VALUATIONS:
        path = account_increments(res, prep, 4 * MS, valuation)
        assert path.times_ns.tolist() == [0, 4 * MS, 8 * MS, 12 * MS]
        assert path.position_units[:, 0].tolist() == [0, 1_000_000, 1_000_000, 0]
        assert path.cash[1] == approx_pnl(c1) and path.cash[0] == 0.0
        assert path.marked_pnl[1] < 0.0  # the fee (and spread) of the buy is already in the account at 4 ms
        assert path.marked_pnl[3] == pytest.approx(res.settlement.net_pnl, rel=1e-12)


def test_mid_valuation_is_the_metrics_marked_value_convention() -> None:
    # Both venues held at once, in arrival order: venue_b short at 1 ms, venue_a long 2-10 ms, venue_b flat at 11 ms.
    res = hand_result([SHORT_FILLS[0], *LONG_FILLS, (11, 1, 1, 39_802, 500_000)])
    prep = hand_market()
    path = account_increments(res, prep, 3 * MS, "mid")
    np.testing.assert_array_equal(path.marked_pnl, marked_value(res, prep, path.times_ns))


def test_unmarked_grid_point_is_nan_in_the_mark_and_both_adjacent_increments() -> None:
    res = hand_result(LONG_FILLS)
    prep = hand_market(clear_at_8ms=True)
    for valuation in VALUATIONS:
        path = account_increments(res, prep, 4 * MS, valuation)
        assert np.isnan(path.marked_pnl[2]) and np.isfinite(path.marked_pnl[[0, 1, 3]]).all()
        assert np.isnan(path.increments_pnl[1]) and np.isnan(path.increments_pnl[2])
        assert path.n_unmarked == 1
        assert path.marked_pnl[3] == pytest.approx(res.settlement.net_pnl, rel=1e-12)  # still reconciles


def test_grid_end_defaults_to_the_last_receipt_and_the_last_point_covers_it() -> None:
    res = hand_result(LONG_FILLS)
    prep = hand_market()  # receipts at 0, 5 and 12 ms
    path = account_increments(res, prep, 5 * MS, "mid")
    assert path.times_ns.tolist() == [0, 5 * MS, 10 * MS, 15 * MS]
    explicit = account_increments(res, prep, 5 * MS, "mid", start_ns=-5 * MS, end_ns=12 * MS)
    assert explicit.times_ns.tolist() == [-5 * MS, 0, 5 * MS, 10 * MS, 15 * MS]
    assert explicit.marked_pnl[0] == 0.0 and explicit.marked_pnl[1] == 0.0
    np.testing.assert_array_equal(explicit.marked_pnl[1:], path.marked_pnl)


def test_grid_must_cover_every_fill() -> None:
    res = hand_result(LONG_FILLS)
    prep = hand_market()
    with pytest.raises(InvalidConfigError):  # first fill arrives at 2 ms: the account is not flat at the start
        account_increments(res, prep, 4 * MS, "mid", start_ns=2 * MS)
    with pytest.raises(InvalidConfigError):  # last fill arrives at 10 ms, after the last grid point (8 ms)
        account_increments(res, prep, 4 * MS, "mid", end_ns=6 * MS)


def test_idle_account_is_a_zero_path() -> None:
    path = account_increments(hand_result([]), hand_market(), 4 * MS, "exit")
    assert path.times_ns.tolist() == [0, 4 * MS, 8 * MS, 12 * MS]
    assert not path.position_units.any() and not path.marked_pnl.any() and not path.increments_pnl.any()
    assert path.net_pnl == 0.0 and path.n_unmarked == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid_ns": 0},
        {"grid_ns": True},
        {"grid_ns": 1.5},
        {"valuation": "last"},
        {"start_ns": 12 * MS},
        {"end_ns": 0},
        {"start_ns": 1.0},
    ],
)
def test_invalid_grid_parameters_are_rejected(kwargs: dict[str, Any]) -> None:
    params: dict[str, Any] = {"grid_ns": 4 * MS, "valuation": "mid"} | kwargs
    with pytest.raises(InvalidConfigError):
        account_increments(hand_result(LONG_FILLS), hand_market(), **params)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid_ns": 2**70},
        {"grid_ns": 2**63},
        {"start_ns": -(2**63) - 1},
        {"end_ns": 2**63},
        {"start_ns": 2**63 - 2, "end_ns": 2**63 - 1, "grid_ns": 2**62},
    ],
)
def test_grid_parameters_outside_the_int64_clock_are_refused(kwargs: dict[str, Any]) -> None:
    # The clock is int64 epoch nanoseconds: an anchor or a step beyond it, or a last grid time that would
    # overflow it, is a configuration error, never a numpy OverflowError or a silently wrapped grid.
    params: dict[str, Any] = {"grid_ns": 4 * MS, "valuation": "mid"} | kwargs
    with pytest.raises(InvalidConfigError):
        account_increments(hand_result([]), hand_market(), **params)


def test_a_grid_coarser_than_the_span_is_one_interval() -> None:
    path = account_increments(hand_result(LONG_FILLS), hand_market(), 2**40, "mid")
    assert path.times_ns.tolist() == [0, 2**40] and path.n_intervals == 1
    assert path.increments_pnl[0] == pytest.approx(path.net_pnl, rel=1e-12)


def test_oversized_grids_are_refused_before_allocation() -> None:
    with pytest.raises(InvalidConfigError):
        account_increments(hand_result(LONG_FILLS), hand_market(), 1, "mid")  # 12 million nanosecond points


def test_market_must_match_the_replayed_venues() -> None:
    one_venue = PreparedMarket.build(MarketData((hand_market().data.venues[0],)))
    for valuation in VALUATIONS:
        with pytest.raises(InvalidConfigError):
            account_increments(hand_result(LONG_FILLS), one_venue, 4 * MS, valuation)


def test_a_settlement_that_disagrees_with_the_fills_is_refused() -> None:
    res = hand_result(LONG_FILLS)
    c0, c1 = res.settlement.cash
    tampered = dataclasses.replace(res, settlement=dataclasses.replace(res.settlement, cash=(c0 + 1e-6, c1)))
    with pytest.raises(InvalidDataError):
        account_increments(tampered, hand_market(), 4 * MS, "mid")


# ---------------------------------------------------------------------------
# Reconciliation on synthetic replays
# ---------------------------------------------------------------------------

BASE = Scenario("base", DEMO_BASE_FEES, 5 * MS)


def synthetic_replay(seed: int, *, latency_ms: int = 5) -> tuple[ReplayResult, PreparedMarket]:
    prep = PreparedMarket.build(random_market(seed, n_rows=300))
    stream = prep.stream
    signal = np.random.default_rng(seed + 1).normal(0.0, 1.0, size=len(stream))
    first, last = int(stream.recv_ns[0]), int(stream.recv_ns[-1])
    session = SessionPlan(first + 20 * MS, last - 400 * MS, last - 300 * MS, last - 60 * MS, last)
    params = ThresholdParams(
        direction=(1, -1),
        size_units=(2_000, 1_000),
        enter=0.7,
        exit=0.0,
        max_hold_ns=150 * MS,
        entry_slip_ticks=2,
        exit_slip_ticks=6,
    )
    scenario = Scenario("base", DEMO_BASE_FEES, latency_ms * MS)
    return run_replay(prep, signal, params, scenario, session, RiskLimits(5_000)), prep


def oracle_state(res: ReplayResult, grid: NDArray[np.int64], n_venues: int) -> tuple[list[list[int]], list[float]]:
    """Positions and example cash at each grid time from a plain loop over the fill records."""
    fills = res.fills
    positions: list[list[int]] = []
    cash: list[float] = []
    for t in grid.tolist():
        seen = fills.arrival_ns <= t
        positions.append(
            [
                int(
                    (
                        fills.side[seen & (fills.venue == v)].astype(np.int64)
                        * fills.qty_units[seen & (fills.venue == v)]
                    ).sum()
                )
                for v in range(n_venues)
            ]
        )
        cash.append(math.fsum(fills.cash[seen].tolist()))
    return positions, cash


def oracle_mark(
    prep: PreparedMarket, grid: NDArray[np.int64], positions: list[list[int]], cash: list[float], valuation: str
) -> list[float]:
    """M at each grid time from the last *received* book, priced per the valuation (NaN when unusable)."""
    out: list[float] = []
    for t, q_all, c in zip(grid.tolist(), positions, cash, strict=True):
        value = c
        for v, q in enumerate(q_all):
            if q == 0:
                continue
            tape = prep.data.venues[v]
            rows = [r for r in range(tape.book.recv_ns.shape[0]) if int(tape.book.recv_ns[r]) <= t]
            if not rows or not prep.valid[v][rows[-1]]:
                value = math.nan
                break
            r = rows[-1]
            bid = float(tape.book.bid_px[r, 0]) * float(tape.instrument.tick_size)
            ask = float(tape.book.ask_px[r, 0]) * float(tape.instrument.tick_size)
            price = 0.5 * (bid + ask) if valuation == "mid" else (bid if q > 0 else ask)
            value -= q / price
        out.append(value)
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("valuation", VALUATIONS)
def test_fixed_grid_path_reconciles_with_the_ledger_on_synthetic_replays(seed: int, valuation: str) -> None:
    res, prep = synthetic_replay(seed)
    assert res.settlement.valid and len(res.fills) > 10
    path = account_increments(res, prep, 50 * MS, valuation)
    grid = path.times_ns
    assert grid[0] == prep.stream.recv_ns[0] and grid[-2] < prep.stream.recv_ns[-1] <= grid[-1]
    assert np.all(np.diff(grid) == 50 * MS)
    assert path.position_units.shape == (grid.shape[0], 2) and path.increments_pnl.shape == (grid.shape[0] - 1,)
    positions, cash = oracle_state(res, grid, 2)
    assert path.position_units.tolist() == positions
    np.testing.assert_allclose(path.cash, cash, rtol=1e-12, atol=1e-15)
    expected = oracle_mark(prep, grid, positions, cash, valuation)
    np.testing.assert_allclose(path.marked_pnl, expected, rtol=1e-12, atol=1e-15, equal_nan=True)
    # Exact reconciliation: the account ends flat, so the final mark is the settlement's net settlement cash.
    net = res.settlement.net_pnl
    assert net is not None and path.net_pnl == net
    assert abs(path.final_pnl - net) <= 1e-12 + 1e-12 * abs(net)
    assert math.fsum(path.increments_pnl[np.isfinite(path.increments_pnl)].tolist()) == pytest.approx(net, rel=1e-9)
    # Fixed intervals include idle ones: while flat at both ends M equals C and the increment is exactly 0.
    flat = ~path.position_units.any(axis=1)
    idle = flat[:-1] & flat[1:] & (np.diff(path.cash) == 0.0)
    assert idle.any() and (~idle).any()
    assert not path.increments_pnl[idle].any()
    if valuation == "mid":
        np.testing.assert_array_equal(path.marked_pnl, marked_value(res, prep, grid))


@pytest.mark.parametrize("seed", [4, 5])
def test_exit_valuation_is_never_above_the_mid_valuation(seed: int) -> None:
    res, prep = synthetic_replay(seed)
    mid = account_increments(res, prep, 20 * MS, "mid")
    exit_ = account_increments(res, prep, 20 * MS, "exit")
    held = mid.position_units.any(axis=1)
    both = np.isfinite(mid.marked_pnl) & np.isfinite(exit_.marked_pnl)
    assert (held & both).any()
    assert np.all(exit_.marked_pnl[held & both] < mid.marked_pnl[held & both])
    np.testing.assert_array_equal(exit_.marked_pnl[~held], mid.marked_pnl[~held])
    np.testing.assert_array_equal(exit_.position_units, mid.position_units)
    np.testing.assert_array_equal(exit_.cash, mid.cash)


def test_increment_matrix_stacks_paths_on_one_grid() -> None:
    res_a, prep = synthetic_replay(6)
    res_b, _ = synthetic_replay(6, latency_ms=50)
    a = account_increments(res_a, prep, 40 * MS, "mid")
    b = account_increments(res_b, prep, 40 * MS, "mid")
    matrix = increment_matrix([a, b])
    assert matrix.shape == (a.n_intervals, 2) and matrix.dtype == np.float64
    np.testing.assert_array_equal(matrix[:, 0], a.increments_pnl)
    np.testing.assert_array_equal(matrix[:, 1], b.increments_pnl)
    with pytest.raises(InvalidConfigError, match="settlement currency"):
        increment_matrix([a, dataclasses.replace(b, currency="EUR")])
    with pytest.raises(InvalidConfigError):
        increment_matrix([a, account_increments(res_b, prep, 20 * MS, "mid")])
    with pytest.raises(InvalidConfigError):
        increment_matrix([a, account_increments(res_b, prep, 40 * MS, "exit")])
    with pytest.raises(InvalidConfigError):
        increment_matrix([])


# ---------------------------------------------------------------------------
# Paired stationary block bootstrap
# ---------------------------------------------------------------------------


def candidates(seed: int = 0, n: int = 240) -> NDArray[np.float64]:
    """Incumbent noise, a copy of it, a better candidate (+0.5 per interval) and one that differs from
    the incumbent by demeaned noise only (its true and sample mean differences are both zero)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=n)
    wobble = rng.normal(size=n)
    return np.column_stack([base, base.copy(), base + 0.5 + 0.2 * rng.normal(size=n), base + wobble - wobble.mean()])


def test_identical_columns_differ_by_exactly_zero_in_every_resample() -> None:
    x = candidates()
    out = paired_block_bootstrap(x, 0, block_lengths=(5, 15, 30), n_boot=300, seed=1)
    assert out.replicates.shape == (3, 300, 4)
    assert out.lower.shape == out.upper.shape == out.share_le_zero.shape == (3, 4)
    for k in (0, 1):  # the incumbent itself and its copy
        assert not out.replicates[:, :, k].any()
        assert out.mean_diff[k] == 0.0
        assert np.all(out.lower[:, k] == 0.0) and np.all(out.upper[:, k] == 0.0)
        assert np.all(out.share_le_zero[:, k] == 1.0)
    assert out.incumbent == 0 and out.block_lengths == (5.0, 15.0, 30.0)
    assert (out.n_boot, out.n_obs, out.alpha, out.seed) == (300, 240, 0.1, 1)
    assert not out.replicates.flags.writeable and not out.lower.flags.writeable


def _jumps(idx: NDArray[np.int64]) -> NDArray[np.bool_]:
    """Positions at which a resample row does not continue its block (a restart that moved it)."""
    n = idx.shape[1]
    out = np.ones(idx.shape, np.bool_)
    out[:, 1:] = idx[:, 1:] != (idx[:, :-1] + 1) % n
    return out


def test_common_bootstrap_indices_nest_restarts_and_share_starts_across_block_lengths() -> None:
    n, boots = 300, 400
    short, mid, long_ = common_bootstrap_indices(n, block_lengths=(5, 15, 30), n_boot=boots, seed=4)
    for idx, block in zip((short, mid, long_), (5, 15, 30), strict=True):
        assert idx.shape == (boots, n) and idx.dtype == np.int64 and not idx.flags.writeable
        assert int(idx.min()) >= 0 and int(idx.max()) < n
        # A step continues its block unless it restarts (probability 1 / block); a restart lands on the
        # successor with probability 1 / n (the stationary bootstrap of stats.stationary_bootstrap_indices).
        assert float((~_jumps(idx)[:, 1:]).mean()) == pytest.approx((1 - 1 / block) + (1 / block) / n, abs=0.01)
    # Common random numbers: the same uniform decides a restart for every block length, so a longer block
    # length only removes restarts, and the same start position is drawn wherever both restart.
    np.testing.assert_array_equal(short[:, 0], mid[:, 0])
    np.testing.assert_array_equal(mid[:, 0], long_[:, 0])
    for coarse, fine in ((long_, mid), (mid, short), (long_, short)):
        moved = _jumps(coarse)
        np.testing.assert_array_equal(fine[moved], coarse[moved])
    assert _jumps(short).sum() > _jumps(mid).sum() > _jumps(long_).sum()


def test_each_block_length_draws_the_same_numbers_whatever_else_is_requested() -> None:
    (alone,) = common_bootstrap_indices(120, block_lengths=(15,), n_boot=90, seed=2)
    together = common_bootstrap_indices(120, block_lengths=(30, 15, 5), n_boot=90, seed=2)
    np.testing.assert_array_equal(alone, together[1])
    x = candidates(2, 120)
    one = paired_block_bootstrap(x, 0, block_lengths=(15,), n_boot=90, seed=2)
    three = paired_block_bootstrap(x, 0, block_lengths=(30, 15, 5), n_boot=90, seed=2)
    np.testing.assert_array_equal(one.replicates[0], three.replicates[1])


def _scalar_common_indices(n: int, blocks: list[float], n_boot: int, seed: int) -> list[NDArray[np.int64]]:
    """Plain-loop oracle: one uniform and one start per position, applied to every block length (one chunk)."""
    rng = np.random.default_rng(seed)
    u = rng.random((n_boot, n))
    starts = rng.integers(0, n, size=(n_boot, n), dtype=np.int64)
    out: list[NDArray[np.int64]] = []
    for block in blocks:
        idx = np.empty((n_boot, n), np.int64)
        for r in range(n_boot):
            position = 0
            for t in range(n):
                position = int(starts[r, t]) if t == 0 or u[r, t] < 1.0 / block else (position + 1) % n
                idx[r, t] = position
        out.append(idx)
    return out


@given(
    n=st.integers(1, 40),
    blocks=st.lists(st.floats(1.0, 30.0), min_size=1, max_size=3),
    n_boot=st.integers(1, 12),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=60, deadline=None)
def test_common_bootstrap_indices_match_a_scalar_reference(n: int, blocks: list[float], n_boot: int, seed: int) -> None:
    fast = common_bootstrap_indices(n, block_lengths=blocks, n_boot=n_boot, seed=seed)
    for got, want in zip(fast, _scalar_common_indices(n, blocks, n_boot, seed), strict=True):
        np.testing.assert_array_equal(got, want)


def test_common_bootstrap_indices_reject_invalid_parameters() -> None:
    for kwargs in (
        {"n": 0},
        {"n_boot": 0},
        {"block_lengths": ()},
        {"block_lengths": (0.5,)},
        {"seed": -1},
        {"n": 2**20, "n_boot": MAX_BOOTSTRAP_ELEMENTS // 2**20 + 1},
    ):
        params: dict[str, Any] = {"n": 20, "block_lengths": (3,), "n_boot": 5, "seed": 0} | kwargs
        with pytest.raises(InvalidConfigError):
            common_bootstrap_indices(params.pop("n"), **params)


def test_block_lengths_share_random_numbers_so_their_replicates_are_correlated() -> None:
    # The same uniforms and starts serve every block length, so the resampled means of one candidate move
    # together across block lengths (unrelated seeds are uncorrelated): the 5/15/30 rows differ by the block
    # length far more than by Monte Carlo noise.
    x = candidates(9, 400)
    out = paired_block_bootstrap(x, 0, block_lengths=(5, 15, 30), n_boot=1_000, seed=2)
    other = paired_block_bootstrap(x, 0, block_lengths=(5,), n_boot=1_000, seed=3)
    r = out.replicates[:, :, 2]
    assert np.corrcoef(r[0], r[1])[0, 1] > 0.2 and np.corrcoef(r[1], r[2])[0, 1] > 0.4  # measured 0.33 / 0.51
    assert abs(np.corrcoef(r[0], other.replicates[0, :, 2])[0, 1]) < 0.1


def test_paired_resamples_use_the_published_indices_for_every_column() -> None:
    x = candidates(3, 60)
    out = paired_block_bootstrap(x, 0, block_lengths=(2, 7), n_boot=50, seed=11)
    for b, idx in enumerate(common_bootstrap_indices(60, block_lengths=(2, 7), n_boot=50, seed=11)):
        expected = (x[idx] - x[idx, 0][:, :, None]).mean(axis=1)  # one index row per resample, all columns
        np.testing.assert_allclose(out.replicates[b], expected, rtol=1e-12, atol=1e-15)
        lo, hi = np.quantile(expected, [0.05, 0.95], axis=0)
        np.testing.assert_allclose(out.lower[b], lo, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(out.upper[b], hi, rtol=1e-12, atol=1e-15)
        np.testing.assert_array_equal(out.share_le_zero[b], (expected <= 0.0).mean(axis=0))
    np.testing.assert_allclose(out.mean_diff, x.mean(axis=0) - x[:, 0].mean(), rtol=1e-12, atol=1e-15)


def test_bootstrap_is_reproducible_from_its_seed() -> None:
    x = candidates(5)
    a = paired_block_bootstrap(x, 0, n_boot=200, seed=7)
    b = paired_block_bootstrap(x, 0, n_boot=200, seed=7)
    c = paired_block_bootstrap(x, 0, n_boot=200, seed=8)
    np.testing.assert_array_equal(a.replicates, b.replicates)
    assert not np.array_equal(a.replicates, c.replicates)
    wide = paired_block_bootstrap(x, 0, n_boot=200, seed=2**64 - 1)  # stats.seed_from yields 64-bit seeds
    assert wide.seed == 2**64 - 1 and not np.array_equal(a.replicates, wide.replicates)


def test_bootstrap_separates_a_better_candidate_from_noise() -> None:
    out = paired_block_bootstrap(candidates(9, 400), 0, block_lengths=(5, 15, 30), n_boot=1_000, seed=2)
    assert np.all(out.share_le_zero[:, 2] < 0.01) and np.all(out.lower[:, 2] > 0.0)
    assert out.mean_diff[2] == pytest.approx(0.5, abs=0.1)
    # A zero sample-mean difference sits in the middle of its own bootstrap distribution.
    assert np.all(out.lower[:, 3] < 0.0) and np.all(out.upper[:, 3] > 0.0)
    assert np.all((out.share_le_zero[:, 3] > 0.35) & (out.share_le_zero[:, 3] < 0.65))


def test_incumbent_may_be_any_column() -> None:
    x = candidates(4, 100)
    out = paired_block_bootstrap(x, 2, block_lengths=(4,), n_boot=100, seed=3)
    assert not out.replicates[:, :, 2].any()
    assert np.all(out.share_le_zero[0, [0, 1]] > 0.95)


@pytest.mark.parametrize(
    "override",
    [
        {"incumbent": 4},
        {"incumbent": -1},
        {"incumbent": True},
        {"block_lengths": (0.5,)},
        {"block_lengths": ()},
        {"n_boot": 1},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"seed": -1},
    ],
)
def test_bootstrap_rejects_invalid_parameters(override: dict[str, Any]) -> None:
    params: dict[str, Any] = {"incumbent": 0, "block_lengths": (5,), "n_boot": 10, "alpha": 0.1, "seed": 0} | override
    with pytest.raises(InvalidConfigError):
        paired_block_bootstrap(candidates(0, 40), **params)


def test_bootstrap_rejects_non_finite_or_misshapen_increments() -> None:
    x = candidates(0, 40)
    with pytest.raises(InvalidDataError):
        paired_block_bootstrap(np.where(np.arange(40) == 3, np.nan, x.T).T, 0, n_boot=10, seed=0)
    with pytest.raises(InvalidDataError):
        paired_block_bootstrap(x[:, 0], 0, n_boot=10, seed=0)
    with pytest.raises(InvalidDataError):
        paired_block_bootstrap(x[:1], 0, n_boot=10, seed=0)
    with pytest.raises(InvalidDataError):
        paired_block_bootstrap(x[:, :1], 0, n_boot=10, seed=0)


def test_bootstrap_refuses_an_index_matrix_that_would_not_fit() -> None:
    x = candidates(0, 100)
    with pytest.raises(InvalidConfigError):
        paired_block_bootstrap(x, 0, block_lengths=(5,), n_boot=MAX_BOOTSTRAP_ELEMENTS // 100 + 1, seed=0)


# ---------------------------------------------------------------------------
# HAC mean
# ---------------------------------------------------------------------------


def ar1(rng: np.random.Generator, n: int, phi: float) -> NDArray[np.float64]:
    eps = rng.normal(size=n)
    out = np.empty(n)
    out[0] = eps[0] / math.sqrt(1.0 - phi * phi)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + eps[t]
    return out


def test_hac_mean_reports_the_newey_west_error_of_every_column() -> None:
    rng = np.random.default_rng(1)
    x = np.column_stack([ar1(rng, 500, 0.5) + 0.1, ar1(rng, 500, -0.3)])
    out = hac_mean(x, lags=6)
    assert out.mean.tolist() == [float(np.mean(x[:, 0])), float(np.mean(x[:, 1]))]
    assert out.std_error.tolist() == [newey_west_se(x[:, 0], 6), newey_west_se(x[:, 1], 6)]
    np.testing.assert_array_equal(out.t_stat, out.mean / out.std_error)
    assert out.p_value.tolist() == [two_sided_p(t) for t in out.t_stat.tolist()]
    np.testing.assert_allclose(out.p_value, [2.0 * (1.0 - PHI(abs(t))) for t in out.t_stat], rtol=1e-9)
    assert (out.lags, out.n_obs) == (6, 500)
    assert not out.mean.flags.writeable


def test_hac_p_value_keeps_the_far_tail() -> None:
    # 2 * (1 - Phi(|t|)) and the erf-based NormalDist.cdf(-|t|) are exactly 0 beyond |t| ~ 8.3 in float64;
    # erfc(|t| / sqrt 2) is 2 * Phi(-|t|) exact to the float floor.
    x = 1.0 + np.random.default_rng(5).normal(size=144)  # t around 12
    out = hac_mean(x, lags=0)
    t = float(out.t_stat[0])
    assert t > 10.0 and PHI(-t) == 0.0
    assert 0.0 < out.p_value[0] < 1e-30 and out.p_value[0] == two_sided_p(t)


def test_hac_lag_rule_is_explicit_and_reported() -> None:
    x = np.random.default_rng(2).normal(size=(1_000, 1))
    assert newey_west_lags(1_000) == math.floor(4.0 * (1_000 / 100.0) ** (2.0 / 9.0)) == 6
    assert newey_west_lags(2) == 1  # never beyond n - 1
    out = hac_mean(x)
    assert out.lags == 6
    assert out.std_error[0] == newey_west_se(x[:, 0])  # stats' default is the same rule


def test_hac_mean_accepts_one_series_and_leaves_an_idle_column_without_a_t_statistic() -> None:
    x = np.random.default_rng(3).normal(size=50)
    one = hac_mean(x, lags=2)
    assert one.mean.shape == (1,) and one.mean[0] == x.mean()
    both = hac_mean(np.column_stack([x, np.zeros(50)]), lags=2)
    assert both.std_error[1] == 0.0 and np.isnan(both.t_stat[1]) and np.isnan(both.p_value[1])
    assert np.isfinite(both.t_stat[0]) and both.t_stat[0] == one.t_stat[0]


def test_hac_mean_rejects_non_finite_input_and_lags_outside_the_sample() -> None:
    with pytest.raises(InvalidDataError):
        hac_mean(np.array([1.0, np.nan, 2.0]), lags=1)
    with pytest.raises(InvalidConfigError):
        hac_mean(np.arange(5.0), lags=5)
    with pytest.raises(InvalidConfigError):
        hac_mean(np.arange(5.0), lags=-1)
    with pytest.raises(InvalidDataError):
        hac_mean(np.zeros((2, 2, 2)))


def test_hac_mean_matches_statsmodels_cov_hac_when_it_is_installed() -> None:
    # statsmodels is not a dependency of the library or its tests; the cross-check runs only where it
    # happens to be importable (``pytest.importorskip`` keeps the dependency-hygiene scan clean).
    sm = pytest.importorskip("statsmodels.api")
    sandwich = pytest.importorskip("statsmodels.stats.sandwich_covariance")
    rng = np.random.default_rng(4)
    for lags in (0, 3, 8):
        x = ar1(rng, 300, 0.4) + 0.05
        fit = sm.OLS(x, np.ones((300, 1))).fit()
        cov = sandwich.cov_hac(fit, nlags=lags, use_correction=False)
        ours = hac_mean(x, lags=lags)
        assert ours.std_error[0] == pytest.approx(math.sqrt(float(cov[0, 0])), rel=1e-10)
        assert ours.mean[0] == pytest.approx(float(fit.params[0]), rel=1e-12)


# ---------------------------------------------------------------------------
# Selection diagnostics
# ---------------------------------------------------------------------------


def trials(seed: int = 0, n: int = 160, k: int = 6) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n, k))
    x[:, 2] += 0.3
    return x


def test_selection_diagnostics_wrap_dsr_and_pbo_with_explicit_caveats() -> None:
    x = trials()
    out = selection_diagnostics(x, n_trials=40, n_splits=8)
    sharpes = [return_moments(x[:, j]).sharpe for j in range(6)]
    moments = return_moments(x[:, 2])
    variance = float(np.var(sharpes, ddof=1))
    assert out["selected"] == 2 and out["n_columns"] == 6 and out["n_trials"] == 40 and out["n_obs"] == 160
    assert out["sharpe"] == moments.sharpe and out["skew"] == moments.skew and out["kurt"] == moments.kurt
    assert out["sr_variance"] == variance
    assert out["expected_max_sharpe"] == expected_max_sharpe(40, variance)
    assert out["psr"] == probabilistic_sharpe_ratio(moments.sharpe, 160, moments.skew, moments.kurt)
    assert out["dsr"] == deflated_sharpe_ratio(moments.sharpe, 40, 160, moments.skew, moments.kurt, variance)
    pbo = pbo_cscv(x, 8)
    assert out["pbo"] == pbo.pbo and out["prob_oos_loss"] == pbo.prob_oos_loss
    assert out["n_splits"] == 8 and out["n_combinations"] == pbo.n_combinations and out["metric"] == "sharpe"
    assert out["caveats"] == list(SELECTION_CAVEATS) and len(out["caveats"]) >= 3
    assert all(isinstance(c, str) and c for c in out["caveats"])
    joined = " ".join(out["caveats"]).lower()
    assert "one exposed" in joined and "independent" in joined and "not a fresh" in joined
    canonical_json(out)  # strict JSON: finite numbers only


def test_selection_diagnostics_accept_an_explicit_selection_and_the_mean_metric() -> None:
    x = trials(1)
    out = selection_diagnostics(x, n_trials=6, selected=4, n_splits=4, metric="mean")
    moments = return_moments(x[:, 4])
    assert out["selected"] == 4 and out["sharpe"] == moments.sharpe
    assert out["pbo"] == pbo_cscv(x, 4, metric="mean").pbo and out["metric"] == "mean"


def test_selection_diagnostics_pass_the_cscv_metric_through() -> None:
    x = trials(8)
    sharpe, mean = pbo_cscv(x, 6), pbo_cscv(x, 6, metric="mean")
    assert sharpe.pbo != mean.pbo  # data on which the metric matters, so the pass-through is observable
    out = selection_diagnostics(x, n_trials=6, n_splits=6, metric="mean")
    assert out["pbo"] == mean.pbo and out["prob_oos_loss"] == mean.prob_oos_loss and out["metric"] == "mean"
    assert selection_diagnostics(x, n_trials=6, n_splits=6)["pbo"] == sharpe.pbo


def test_default_selection_is_the_best_trading_column_even_when_every_trade_lost() -> None:
    # An idle trial counts with Sharpe 0 in the variance but is never the selection: when every trading
    # configuration lost on the day, the least bad one is deflated, not the one that did nothing.
    rng = np.random.default_rng(7)
    z = np.column_stack([rng.normal(-0.5, 1.0, size=120), rng.normal(-0.3, 1.0, size=120), np.zeros(120)])
    sharpes = [return_moments(z[:, j]).sharpe for j in range(2)]
    assert max(sharpes) < 0.0
    out = selection_diagnostics(z, n_trials=3, n_splits=4)
    assert out["selected"] == int(np.argmax(sharpes)) and out["idle_columns"] == 1
    assert out["sr_variance"] == float(np.var([*sharpes, 0.0], ddof=1))
    with pytest.raises(InvalidDataError):  # nothing traded: there is no selection to deflate
        selection_diagnostics(np.zeros((120, 2)), n_trials=2, n_splits=4)


def test_a_single_trial_deflates_to_its_probabilistic_sharpe_ratio() -> None:
    x = trials(2)[:, :1]
    out = selection_diagnostics(x, n_trials=1, n_splits=4)
    moments = return_moments(x[:, 0])
    assert out["sr_variance"] == 0.0 and out["expected_max_sharpe"] == 0.0
    assert out["dsr"] == out["psr"] == probabilistic_sharpe_ratio(moments.sharpe, 160, moments.skew, moments.kurt)
    assert out["pbo"] is None and out["prob_oos_loss"] is None and out["n_combinations"] is None
    canonical_json(out)


def test_idle_trials_count_with_a_zero_sharpe_ratio_but_cannot_be_selected() -> None:
    x = trials(3)
    x[:, 5] = 0.0
    out = selection_diagnostics(x, n_trials=12, n_splits=4)
    sharpes = [return_moments(x[:, j]).sharpe for j in range(5)] + [0.0]
    assert out["sr_variance"] == float(np.var(sharpes, ddof=1))
    with pytest.raises(InvalidDataError):
        selection_diagnostics(x, n_trials=12, selected=5, n_splits=4)


def test_selection_diagnostics_reject_inconsistent_inputs() -> None:
    x = trials(4)
    with pytest.raises(InvalidConfigError):  # the ledger cannot hold fewer trials than the matrix
        selection_diagnostics(x, n_trials=5, n_splits=4)
    with pytest.raises(InvalidConfigError):
        selection_diagnostics(x, n_trials=6, selected=6, n_splits=4)
    with pytest.raises(InvalidConfigError):
        selection_diagnostics(x, n_trials=6, n_splits=3)
    with pytest.raises(InvalidDataError):
        selection_diagnostics(np.where(x > 2.5, np.nan, x), n_trials=6, n_splits=4)
    with pytest.raises(InvalidDataError):
        selection_diagnostics(x[:, 0], n_trials=6, n_splits=4)
    constant = x.copy()
    constant[:, 1] = 0.25  # a constant non-zero increment is not an account that exists
    with pytest.raises(InvalidDataError):
        selection_diagnostics(constant, n_trials=6, n_splits=4)
