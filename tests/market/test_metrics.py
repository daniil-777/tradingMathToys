"""Ledger-derived paths and account metrics: step semantics, example cash C, the observed-mid
mark M, round trips and the summary, all reconciled against ``ledger.settle``."""

from __future__ import annotations

import dataclasses
import math
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from financeai.application.grid import episode_nets
from financeai.core.clock import MISSING_NS
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
from financeai.domains.market.ledger import Fills, exact_score, fill_cash, settle
from financeai.domains.market.metrics import (
    cash_path,
    drawdown,
    drawdown_path,
    episodes,
    known_position_path,
    marked_value,
    marked_value_path,
    position_path,
    summary,
)
from financeai.domains.market.scenario import DEMO_BASE_FEES, RiskLimits, Scenario, SessionPlan
from financeai.domains.market.stream import MarketData

from .synthetic import Row, ladder, random_market, tapes_from_rows

INSTRUMENTS = (DEMO_INVERSE_A, DEMO_INVERSE_B)
FEES = (Decimal("0.00005"), Decimal("0.0001"))
MS = 1_000_000

FillRow = tuple[int, int, int, int, int]
"""(arrival_ms, venue, side, px_ticks, qty_units); ticks are 0.5 USD."""


def make_result(
    rows: list[FillRow],
    *,
    latency_ms: int = 5,
    ack_ms: list[int] | None = None,
    fees: tuple[Decimal, ...] = FEES,
    cap: int = 1_000_000,
) -> ReplayResult:
    """A replay result assembled from hand-written fills (one order per fill)."""
    n = len(rows)
    arrival = np.array([r[0] * MS for r in rows], np.int64)
    acks = ack_ms if ack_ms is not None else [latency_ms] * n
    ack = arrival + np.array(acks, np.int64) * MS
    venue = np.array([r[1] for r in rows], np.int8)
    side = np.array([r[2] for r in rows], np.int8)
    px = np.array([r[3] for r in rows], np.int32)
    qty = np.array([r[4] for r in rows], np.int64)
    cash = np.array(
        [
            fill_cash(int(s), int(q), float(p) * float(INSTRUMENTS[v].tick_size), float(fees[v]))
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
    settlement = settle(fills, INSTRUMENTS, fees, cap_units=cap)
    return ReplayResult(fills=fills, orders=orders, settlement=settlement, decisions=n)


ROUND_TRIPS: list[FillRow] = [
    (10, 0, 1, 40_000, 600),  # venue_a buy two levels of one sweep (same arrival)
    (10, 0, 1, 40_001, 400),
    (12, 1, -1, 40_010, 1_000),  # venue_b short
    (20, 0, -1, 40_100, 1_000),  # venue_a flat
    (25, 1, 1, 40_000, 1_000),  # venue_b flat
]


# ---------------------------------------------------------------------------
# Position and cash paths
# ---------------------------------------------------------------------------


def test_position_path_is_a_post_event_step_at_fill_arrival() -> None:
    res = make_result(ROUND_TRIPS)
    times, q = position_path(res, 0)
    assert times.dtype == np.int64 and q.dtype == np.int64
    assert times.tolist() == [10 * MS, 10 * MS, 20 * MS]
    assert q.tolist() == [600, 1_000, 0]
    t1, q1 = position_path(res, 1)
    assert t1.tolist() == [12 * MS, 25 * MS] and q1.tolist() == [-1_000, 0]
    assert not times.flags.writeable and not q.flags.writeable


def test_position_extremes_match_the_ledger_cap_check() -> None:
    rows: list[FillRow] = [(1, 1, 1, 40_000, 600_000), (1, 1, 1, 40_000, 500_000), (3, 1, -1, 40_000, 1_100_000)]
    res = make_result(rows)
    _, q = position_path(res, 1)
    assert int(np.abs(q).max()) == res.settlement.max_abs_position_units[1] == 1_100_000
    assert res.settlement.reason == "cap_exceeded"


def test_known_position_follows_acknowledgements_not_arrivals() -> None:
    # The second fill arrives later but is acknowledged first (per-fill ack delays).
    rows: list[FillRow] = [(10, 0, 1, 40_000, 100), (11, 0, 1, 40_000, 200)]
    res = make_result(rows, ack_ms=[9, 2])
    times, q = known_position_path(res, 0)
    assert times.tolist() == [13 * MS, 19 * MS]
    assert q.tolist() == [200, 300]
    true_t, true_q = position_path(res, 0)
    assert true_t.tolist() == [10 * MS, 11 * MS] and true_q.tolist() == [100, 300]


def test_cash_path_accumulates_the_example_score() -> None:
    res = make_result(ROUND_TRIPS)
    times, c = cash_path(res)
    assert times.tolist() == res.fills.arrival_ns.tolist()
    np.testing.assert_allclose(c, np.cumsum(res.fills.cash), rtol=1e-13, atol=1e-15)
    assert res.settlement.net_pnl is not None
    assert c[-1] == pytest.approx(res.settlement.net_pnl, rel=1e-12, abs=1e-15)
    for v in (0, 1):
        tv, cv = cash_path(res, v)
        assert tv.tolist() == position_path(res, v)[0].tolist()
        assert cv[-1] == pytest.approx(res.settlement.cash[v], rel=1e-12, abs=1e-15)
    assert not c.flags.writeable


def test_opening_cash_is_not_profit() -> None:
    # FX-V02: an opening buy of 1,000,000 stored quantity units at 20,000 credits C with ~+50 BTC.
    res = make_result([(1, 0, 1, 40_000, 1_000_000)])
    _, c = cash_path(res)
    assert c[-1] == pytest.approx(19_999 / 400, rel=1e-14)
    assert res.settlement.net_pnl is None and res.settlement.reason == "not_flat"


def test_empty_results_give_empty_paths() -> None:
    res = make_result([])
    for arrays in (position_path(res, 0), known_position_path(res, 1), cash_path(res)):
        assert arrays[0].shape == (0,) and arrays[1].shape == (0,)


@pytest.mark.parametrize("venue", [-1, 2, True])
def test_unknown_venue_is_rejected(venue: int) -> None:
    with pytest.raises(InvalidConfigError):
        position_path(make_result(ROUND_TRIPS), venue)


def test_fills_out_of_arrival_order_are_rejected() -> None:
    res = make_result(ROUND_TRIPS)
    shuffled = Fills(
        *(
            np.ascontiguousarray(getattr(res.fills, name)[::-1])
            for name in (
                "order_id",
                "venue",
                "side",
                "px_ticks",
                "qty_units",
                "submit_ns",
                "arrival_ns",
                "ack_ns",
                "cash",
            )
        )
    )
    bad = ReplayResult(fills=shuffled, orders=res.orders, settlement=res.settlement, decisions=1)
    with pytest.raises(InvalidDataError):
        cash_path(bad)


# ---------------------------------------------------------------------------
# Observed-mid mark M and drawdown
# ---------------------------------------------------------------------------


def mark_market() -> PreparedMarket:
    """venue_a mids 20,000 -> 20,100 -> (book lost) -> 20,000; venue_b constant (times in us)."""
    b0, a0 = ladder(39_999, 40_001, 10_000_000)
    b1, a1 = ladder(40_199, 40_201, 10_000_000)
    _, a2 = ladder(40_199, 40_201, 10_000_000)
    venue_a: list[Row] = [(0, 0, b0, a0), (5_000, 4_000, b1, a1), (8_000, 7_500, [], a2), (12_000, 11_000, b0, a0)]
    db, da = ladder(40_000, 40_002, 1_000_000)
    venue_b: list[Row] = [(0, 0, db, da), (6_000, 5_000, db, da)]
    data = MarketData((tapes_from_rows(DEMO_INVERSE_A, venue_a), tapes_from_rows(DEMO_INVERSE_B, venue_b)))
    return PreparedMarket.build(data)


MARK_FILLS: list[FillRow] = [(2, 0, 1, 40_000, 1_000_000), (10, 0, -1, 40_100, 1_000_000)]


def test_marked_value_uses_the_observed_mid_and_never_a_stale_price() -> None:
    res = make_result(MARK_FILLS)
    grid = np.array([0, 2, 4, 5, 7, 8, 9, 10, 11, 12], np.int64) * MS
    m = marked_value(res, mark_market(), grid)
    c_open = float(res.fills.cash[0])
    c_flat = float(np.sum(res.fills.cash))
    assert m[0] == 0.0
    # FX-V02: C = 19999/400 BTC after the opening buy, but M = C - q/P = -1/400 BTC.
    assert m[1] == pytest.approx(-1 / 400, rel=1e-9)
    assert m[2] == m[1]
    assert m[3] == pytest.approx(c_open - 1_000_000 / 20_100, rel=1e-12)
    assert m[4] == m[3]
    assert np.isnan(m[5]) and np.isnan(m[6])  # held while the venue_a book is unusable
    assert m[7] == pytest.approx(c_flat, rel=1e-12)  # flat: M = C even without a valid book
    assert m[8] == m[7] == m[9]
    assert not m.flags.writeable


def test_marked_value_accepts_any_grid_order() -> None:
    res = make_result(MARK_FILLS)
    grid = np.array([12, 0, 5, 2, 9], np.int64) * MS
    m = marked_value(res, mark_market(), grid)
    order = np.argsort(grid)
    np.testing.assert_array_equal(m[order], marked_value(res, mark_market(), grid[order]))


@pytest.mark.parametrize("grid", [np.array([0.0, 1.0]), np.zeros((2, 2), np.int64)])
def test_marked_value_rejects_float_or_nested_clocks(grid: np.ndarray) -> None:
    with pytest.raises(InvalidDataError):
        marked_value(make_result(MARK_FILLS), mark_market(), grid)


def test_marked_value_needs_the_replayed_market() -> None:
    one_venue = PreparedMarket.build(MarketData((mark_market().data.venues[0],)))
    with pytest.raises(InvalidConfigError):
        marked_value(make_result(MARK_FILLS), one_venue, np.array([0], np.int64))


def test_marked_value_path_lists_every_change_point_of_m() -> None:
    res = make_result(MARK_FILLS)
    prep = mark_market()
    times, m = marked_value_path(res, prep)
    # Fills at 2 and 10 ms; venue_a books at 5 and 8 ms while held. Books while flat
    # (0 and 12 ms) and venue_b books (never held) cannot change M.
    assert times.tolist() == [2 * MS, 5 * MS, 8 * MS, 10 * MS]
    np.testing.assert_array_equal(m, marked_value(res, prep, times))
    assert not times.flags.writeable and not m.flags.writeable


def test_marked_value_path_matches_the_vectorised_mark_on_replays() -> None:
    data = random_market(5, n_rows=400)
    prep = PreparedMarket.build(data)
    stream = prep.stream
    signal = np.random.default_rng(6).normal(0.0, 1.0, size=len(stream))
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
    res = run_replay(prep, signal, params, Scenario("t", DEMO_BASE_FEES, 5 * MS), session, RiskLimits())
    assert len(res.fills) > 10
    times, m = marked_value_path(res, prep)
    assert np.all(np.diff(times) > 0)
    np.testing.assert_array_equal(m, marked_value(res, prep, times))
    # Whenever the account is flat the mark is exactly the example cash.
    ct, c = cash_path(res)
    k = np.searchsorted(ct, times, side="right") - 1
    flat = np.ones(times.shape[0], bool)
    for v in range(2):
        pt, q = position_path(res, v)
        if q.size == 0:
            continue
        j = np.searchsorted(pt, times, side="right") - 1
        flat &= np.where(j >= 0, q[np.maximum(j, 0)], 0) == 0
    assert flat.any() and (~flat).any()
    np.testing.assert_array_equal(m[flat], np.where(k >= 0, c[np.maximum(k, 0)], 0.0)[flat])


def test_drawdown_runs_from_the_flat_start_and_skips_unmarked_instants() -> None:
    times = np.array([1, 2, 3, 4, 5, 6], np.int64)
    m = np.array([-1.0, 2.0, 1.0, np.nan, -0.5, 3.0])
    dd = drawdown(times, m)
    assert dd.max_pnl == 2.5
    assert (dd.peak_ns, dd.trough_ns, dd.unmarked_points) == (2, 5, 1)
    np.testing.assert_array_equal(drawdown_path(m), [-1.0, 0.0, -1.0, np.nan, -2.5, 0.0])


def test_drawdown_of_a_falling_start_is_measured_from_zero() -> None:
    dd = drawdown(np.array([7, 9], np.int64), np.array([-0.25, -0.5]))
    assert dd.max_pnl == 0.5 and dd.peak_ns == MISSING_NS and dd.trough_ns == 9
    flat = drawdown(np.zeros(0, np.int64), np.zeros(0))
    assert flat.max_pnl == 0.0 and flat.peak_ns == MISSING_NS and flat.trough_ns == MISSING_NS


# ---------------------------------------------------------------------------
# Round trips and the account summary
# ---------------------------------------------------------------------------


def price(ticks: int) -> Fraction:
    return Fraction(ticks, 2)


def test_episodes_are_flat_to_flat_round_trips_per_venue() -> None:
    ep = episodes(make_result(ROUND_TRIPS), mark_market())
    assert len(ep) == 2
    assert ep.venue.tolist() == [0, 1] and ep.direction.tolist() == [1, -1]
    assert ep.entry_ns.tolist() == [10 * MS, 12 * MS]
    assert ep.exit_ns.tolist() == [20 * MS, 25 * MS]
    assert ep.holding_ns.tolist() == [10 * MS, 13 * MS]
    assert ep.n_fills.tolist() == [3, 2] and ep.max_abs_units.tolist() == [1_000, 1_000]
    assert ep.closed.tolist() == [True, True]
    long_turnover = 600 / price(40_000) + 400 / price(40_001) + 1_000 / price(40_100)
    long_gross = 600 / price(40_000) + 400 / price(40_001) - 1_000 / price(40_100)
    short_turnover = 1_000 / price(40_010) + 1_000 / price(40_000)
    short_gross = -1_000 / price(40_010) + 1_000 / price(40_000)
    expected = [
        (long_gross, Fraction(FEES[0]) * long_turnover),
        (short_gross, Fraction(FEES[1]) * short_turnover),
    ]
    for k, (gross, fee) in enumerate(expected):
        assert ep.gross_pnl[k] == pytest.approx(float(gross), rel=1e-12)
        assert ep.fees[k] == pytest.approx(float(fee), rel=1e-9)
        assert ep.net_pnl[k] == pytest.approx(float(gross - fee), rel=1e-12)
    assert not ep.net_pnl.flags.writeable


def test_open_inventory_is_an_unresolved_episode_not_a_zero() -> None:
    rows: list[FillRow] = [(1, 0, 1, 40_000, 100), (2, 0, -1, 40_002, 100), (3, 0, 1, 40_004, 200)]
    ep = episodes(make_result(rows), mark_market())
    assert ep.closed.tolist() == [True, False]
    assert ep.exit_ns[1] == MISSING_NS and ep.holding_ns[1] == MISSING_NS
    assert np.isnan(ep.gross_pnl[1]) and np.isnan(ep.net_pnl[1])
    assert ep.fees[1] == pytest.approx(0.00005 * 200 / 20_002, rel=1e-9)
    assert ep.n_fills.tolist() == [2, 1]


def one_venue_prep() -> tuple[PreparedMarket, SessionPlan]:
    b0, a0 = ladder(40_000, 40_001, 10_000)
    b1, a1 = ladder(40_004, 40_005, 10_000)
    rows: list[Row] = [(1_000, 500, b0, a0), (9_000, 4_000, b1, a1), (30_000, 25_000, b1, a1)]
    session = SessionPlan(
        start_ns=0, entry_cutoff_ns=20 * MS, flatten_start_ns=25 * MS, exec_cutoff_ns=40 * MS, end_ns=60 * MS
    )
    return PreparedMarket.build(MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),))), session


BASE = Scenario("base", {spec.key: fee for spec, fee in zip(INSTRUMENTS, FEES, strict=True)}, 5 * MS)


def synthetic_replay(
    seed: int, params: ThresholdParams | None = None, latency_ms: int = 5
) -> tuple[ReplayResult, PreparedMarket]:
    prep = PreparedMarket.build(random_market(seed, n_rows=300))
    stream = prep.stream
    signal = np.random.default_rng(seed + 1).normal(0.0, 1.0, size=len(stream))
    first, last = int(stream.recv_ns[0]), int(stream.recv_ns[-1])
    session = SessionPlan(first + 20 * MS, last - 400 * MS, last - 300 * MS, last - 60 * MS, last)
    p = params or ThresholdParams(
        direction=(1, -1),
        size_units=(2_000, 1_000),
        enter=0.7,
        exit=0.0,
        max_hold_ns=150 * MS,
        entry_slip_ticks=2,
        exit_slip_ticks=6,
    )
    scen = Scenario("base", BASE.fees, latency_ms * MS)
    return run_replay(prep, signal, p, scen, session, RiskLimits(5_000)), prep


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_summary_reconciles_with_the_ledger_on_synthetic_replays(seed: int) -> None:
    res, prep = synthetic_replay(seed)
    s = summary(res, prep, BASE, limits=RiskLimits(5_000))
    st_ = res.settlement
    assert st_.valid and s["valid"] and s["reason"] is None and s["violations"] == []
    assert s["net_pnl"] == st_.net_pnl
    assert s["final_cash"] == pytest.approx(st_.net_pnl, rel=1e-12, abs=1e-15)
    exact = exact_score(res.fills, prep.data.instruments, [BASE.fee_for(i) for i in prep.data.instruments])
    assert exact is not None and abs(s["final_cash"] - float(exact)) <= 1e-12
    assert s["fills"] == len(res.fills) and s["orders"] == len(res.orders)
    assert s["fills_by_venue"] == list(st_.n_fills)
    assert s["traded_units_by_venue"] == list(st_.traded_units)
    assert s["traded_units"] == sum(st_.traded_units)
    assert s["fees_by_venue"] == list(st_.fees)
    assert s["max_abs_position_units"] == list(st_.max_abs_position_units)
    assert s["final_position_units"] == [0, 0]
    assert s["round_trips"] > 0 and s["open_episodes"] == 0
    assert 0.0 <= s["win_rate"] <= 1.0 and s["mean_holding_s"] > 0.0
    assert s["max_drawdown"] >= 0.0
    assert s["reconciled"] and all(v is not False for v in s["reconciliation"].values())
    assert s["reconciliation"]["final_cash_equals_net_pnl"] is True
    assert s["reconciliation"]["episodes_sum_to_net"] is True
    assert s["scenario"] == "base" and s["timing_mode"] == "EXCHANGE_CLOCK"
    filled = int(res.orders.filled_units.sum())
    assert s["fill_ratio"] == pytest.approx(filled / int(res.orders.qty_units.sum()))
    canonical_json(s)  # strict JSON: finite numbers only


def test_summary_of_an_invalid_ending_has_no_score() -> None:
    prep, session = one_venue_prep()
    # The engine's flatten timer decides at flatten_start_ns (25 ms); with 5 ms latency that order
    # would arrive at 30 ms, so a 29 ms execution cutoff leaves the 1 ms entry open.
    late = dataclasses.replace(session, exec_cutoff_ns=29 * MS)
    params = ThresholdParams(
        direction=(1,), size_units=(1_000,), enter=0.5, exit=-10.0, entry_slip_ticks=10, exit_slip_ticks=20
    )
    res = run_replay(prep, np.array([1.0, 0.0, 0.0]), params, BASE, late, RiskLimits())
    s = summary(res, prep, BASE, limits=RiskLimits())
    assert not s["valid"] and s["reason"] == "not_flat" and s["violations"] == ["not_flat"]
    assert s["net_pnl"] is None
    assert math.isfinite(s["final_cash"])  # C exists but is not profit
    assert s["final_position_units"] == [1_000]
    assert s["round_trips"] == 0 and s["open_episodes"] == 1
    assert s["win_rate"] is None and s["mean_holding_s"] is None
    assert s["reconciled"]
    assert s["reconciliation"]["final_cash_equals_net_pnl"] is None
    canonical_json(s)


def test_summary_reports_a_cap_breach_and_an_open_ending_together() -> None:
    rows: list[FillRow] = [(1, 1, 1, 40_000, 1_000_010), (2, 1, -1, 40_000, 1_000_000)]
    res = make_result(rows)
    s = summary(res, mark_market(), BASE, limits=RiskLimits(1_000_000))
    assert s["reason"] == "cap_exceeded" and s["violations"] == ["cap_exceeded", "not_flat"]
    assert s["net_pnl"] is None and s["cap_units"] == 1_000_000
    assert s["max_abs_position_units"] == [0, 1_000_010]
    assert s["reconciled"]


def test_summary_detects_figures_that_disagree_with_the_ledger() -> None:
    res = make_result(ROUND_TRIPS)
    c0, c1 = res.settlement.cash
    tampered = dataclasses.replace(res, settlement=dataclasses.replace(res.settlement, cash=(c0 + 1e-6, c1)))
    s = summary(tampered, mark_market(), BASE)
    assert s["reconciliation"]["cash_by_venue_match"] is False
    assert s["reconciliation"]["final_cash_equals_net_pnl"] is False
    assert not s["reconciled"]


def test_summary_without_a_scenario_skips_only_the_fee_recomputation() -> None:
    s = summary(make_result(ROUND_TRIPS), mark_market())
    assert s["scenario"] is None and s["timing_mode"] is None
    assert s["reconciliation"]["fees_match"] is None
    assert s["reconciled"]


def test_summary_of_an_idle_account() -> None:
    res = make_result([])
    empty_orders = res.orders
    s = summary(res, mark_market(), BASE)
    assert len(empty_orders) == 0 and s["fill_ratio"] is None
    assert s["valid"] and s["net_pnl"] == 0.0 and s["max_drawdown"] == 0.0
    assert s["round_trips"] == 0 and s["win_rate"] is None and s["reconciled"]


SMALL_PARAMS = st.builds(
    ThresholdParams,
    direction=st.sampled_from([(1, 0), (0, 1), (1, -1), (-1, 1), (1, 1)]),
    size_units=st.tuples(st.integers(1, 30).map(lambda k: 100 * k), st.integers(1, 300).map(lambda k: 10 * k)),
    enter=st.floats(0.2, 1.5),
    exit=st.floats(-1.0, 0.5),
    allow_short=st.booleans(),
    max_hold_ns=st.integers(1, 400).map(lambda k: k * MS),
    entry_slip_ticks=st.integers(0, 3),
    exit_slip_ticks=st.integers(0, 8),
)


@given(seed=st.integers(0, 10_000), params=SMALL_PARAMS, latency_ms=st.sampled_from([1, 5, 50]))
@settings(max_examples=25, deadline=None)
def test_summary_always_reconciles(seed: int, params: ThresholdParams, latency_ms: int) -> None:
    res, prep = synthetic_replay(seed, params, latency_ms)
    s = summary(res, prep, BASE, limits=RiskLimits(5_000))
    assert s["reconciled"]
    assert s["round_trips"] + s["open_episodes"] == len(episodes(res, prep))
    if s["valid"]:
        ep = episodes(res, prep)
        assert math.fsum(ep.net_pnl.tolist()) == pytest.approx(s["net_pnl"], rel=1e-12, abs=1e-14)


POLICY_OPTIONS = st.fixed_dictionaries(
    {
        "reverse": st.booleans(),
        "refresh_hold": st.booleans(),
        "max_clips": st.integers(1, 3),
        "clip_spacing_ns": st.integers(0, 50).map(lambda k: k * MS),
        "size_units_cap": st.integers(1, 3),
        "depth_ticks": st.integers(-1, 4),
    }
)
"""The engine's later policy options: flips, staged accumulation, hold refresh, strength and depth sizing."""


@given(seed=st.integers(0, 10_000), params=SMALL_PARAMS, options=POLICY_OPTIONS, latency_ms=st.sampled_from([1, 5, 50]))
@settings(max_examples=25, deadline=None)
def test_metrics_reconcile_and_split_episodes_like_the_grid_under_every_policy_option(
    seed: int, params: ThresholdParams, options: dict[str, Any], latency_ms: int
) -> None:
    res, prep = synthetic_replay(seed, dataclasses.replace(params, **options), latency_ms)
    s = summary(res, prep, BASE, limits=RiskLimits(5_000))
    assert s["reconciled"]
    ep = episodes(res, prep)
    assert s["round_trips"] + s["open_episodes"] == len(ep)
    # application.grid's concentration statistics split an account into the same per-venue
    # flat -> flat round trips (venue by venue there, chronological here; open ones are NaN here).
    grid = [net for _, net in episode_nets(res.fills, len(prep.data))]
    ours = [
        (float(n), bool(c))
        for v in range(len(prep.data))
        for n, c in zip(ep.net_pnl[ep.venue == v], ep.closed[ep.venue == v], strict=True)
    ]
    assert len(grid) == len(ours)
    for theirs, (mine, closed) in zip(grid, ours, strict=True):
        assert mine == pytest.approx(theirs, rel=1e-12, abs=1e-15) if closed else math.isnan(mine)
