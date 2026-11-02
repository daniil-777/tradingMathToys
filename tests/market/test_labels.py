"""Executable labels (E2): one-shot round trips and liquidations with the replay engine's exact
execution semantics. Hand-worked examples on tiny books, an oracle against ``run_replay`` and the
reference twin, causality, compiled/pure-Python parity and the exact (``math.fsum``) summation."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from financeai.core.errors import InvalidConfigError
from financeai.domains.market import labels as labels_mod
from financeai.domains.market.engine import REASON_ENTRY, PreparedMarket, ThresholdParams, run_replay
from financeai.domains.market.instruments import DEMO_INVERSE_A
from financeai.domains.market.labels import (
    LabelSet,
    LabelStatus,
    executable_labels,
    hold_value_labels,
    label_fills,
)
from financeai.domains.market.ledger import Fills, fill_cash
from financeai.domains.market.reference import ThresholdPolicy, run_reference
from financeai.domains.market.scenario import DEMO_BASE_FEES, RiskLimits, Scenario, SessionPlan, TimingMode
from financeai.domains.market.stream import MarketData
from financeai.domains.market.tape import BOOK_CLEAR

from .synthetic import US, Row, ladder, random_market, tapes_from_rows

MS = 1_000_000
S = 1_000 * MS
TICK = 0.5
LABEL_FEES = {key: Decimal("0.00001") for key in DEMO_BASE_FEES}
FEE = float(LABEL_FEES["venue_a/ALPHA-USD"])
DEEP = 10_000
FILLED, PARTIAL, NO_FILL, CENSORED, NO_ORDER = (
    LabelStatus.FILLED,
    LabelStatus.PARTIAL_ENTRY,
    LabelStatus.NO_FILL,
    LabelStatus.CENSORED,
    LabelStatus.NO_ORDER,
)
B0, A0 = ladder(40_000, 40_001, DEEP)
B1, A1 = ladder(40_010, 40_011, DEEP)
ARRAY_FIELDS = [
    f.name for f in dataclasses.fields(LabelSet) if f.name not in ("side", "venue", "size_units", "horizon_ns")
]
FILL_FIELDS = ("order_id", "venue", "side", "px_ticks", "qty_units", "submit_ns", "arrival_ns", "ack_ns", "cash")


def scen(timing: TimingMode = TimingMode.EXCHANGE_CLOCK, latency_ms: int = 5, **kw: Any) -> Scenario:
    return Scenario("labels", LABEL_FEES, latency_ms * MS, timing=timing, **kw)


def market(*rows: Row) -> PreparedMarket:
    return PreparedMarket.build(MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),)))


def with_invalid_rows(data: MarketData, venue: int, rows: Mapping[int, str]) -> MarketData:
    """Copy of ``data`` whose venue book rows are unusable for decisions and fills: ``"clear"`` (a
    recorder BOOK_CLEAR, no levels), ``"no_bid"`` or ``"no_ask"`` (one-sided books)."""
    tapes = data.venues[venue]
    book = tapes.book
    kind, n_bid, n_ask = book.kind.copy(), book.n_bid.copy(), book.n_ask.copy()
    for r, how in rows.items():
        if how == "clear":
            kind[r], n_bid[r], n_ask[r] = BOOK_CLEAR, 0, 0
        elif how == "no_bid":
            n_bid[r] = 0
        else:
            n_ask[r] = 0
    venues = list(data.venues)
    venues[venue] = dataclasses.replace(tapes, book=dataclasses.replace(book, kind=kind, n_bid=n_bid, n_ask=n_ask))
    return MarketData(tuple(venues))


def moving_market() -> PreparedMarket:
    """Quotes 40,000/40,001 until exchange time 25 ms (received at 30 ms), then 40,010/40,011."""
    return market((1_000, 500, B0, A0), (30_000, 25_000, B1, A1), (60_000, 55_000, B1, A1))


def cash(side: int, qty: int, px: int) -> float:
    return fill_cash(side, qty, px * TICK, FEE)


def gross(side: int, qty: int, px: int) -> float:
    return fill_cash(side, qty, px * TICK, 0.0)


def fee(qty: int, px: int) -> float:
    return FEE * qty / (px * TICK)


def inv_avg(*fills: tuple[int, int]) -> float:
    """Inverse average USD price of ``(qty, px_ticks)`` fills: sum q / sum q/p."""
    return sum(q for q, _ in fills) / math.fsum(q / (p * TICK) for q, p in fills)


def one(prep: PreparedMarket, t: int, side: int = 1, **kw: Any) -> LabelSet:
    args: dict[str, Any] = {"venue": 0, "size_units": 1_000, "horizon_ns": 30 * MS, "scenario": scen()}
    args.update(kw)
    return executable_labels(prep, np.array([t], np.int64), side, **args)


def one_shot_replay(
    prep: PreparedMarket,
    t: int,
    side: int,
    venue: int,
    size: int,
    horizon: int,
    scenario: Scenario,
    cutoff: int,
    entry_slip: int,
    exit_slip: int,
) -> tuple[Any, Any]:
    """The engine's view of a label, by both engines: a threshold policy entering on the single cohort at
    ``t`` and leaving after ``max_hold = horizon`` (``t + horizon`` must be a cohort or an acknowledgement
    for the hold expiry to act then). Entries stop at ``t + 1`` ns so the policy never re-enters; the forced
    flatten sits 1 ns before the execution cutoff, where no order can arrive in time."""
    recv = prep.stream.recv_ns
    nv = len(prep.data)
    params = ThresholdParams(
        direction=tuple(int(v == venue) for v in range(nv)),
        size_units=(size,) * nv,
        enter=0.5,
        exit=-10.0,
        max_hold_ns=horizon,
        entry_slip_ticks=entry_slip,
        exit_slip_ticks=exit_slip,
    )
    signal = np.where(recv == t, float(side), 0.0)
    end = int(recv[-1])
    session = SessionPlan(
        start_ns=t,
        entry_cutoff_ns=t + 1,
        flatten_start_ns=cutoff - 1,
        exec_cutoff_ns=cutoff,
        end_ns=max(cutoff, end) + 100 * (scenario.order_latency_ns + scenario.ack_ns),
    )
    res = run_replay(prep, signal, params, scenario, session, RiskLimits(size))
    ref = run_reference(
        prep.data, signal, ThresholdPolicy(params), scenario, session, RiskLimits(size), stream=prep.stream
    )
    return res, ref


def assert_same_fills(fills: Fills, *others: Any) -> None:
    for other in others:
        for name in FILL_FIELDS:
            np.testing.assert_array_equal(getattr(fills, name), getattr(other.fills, name), err_msg=name)


def label_and_replay(
    prep: PreparedMarket,
    t: int,
    side: int,
    horizon: int,
    scenario: Scenario,
    cutoff: int,
    entry_slip: int = 0,
    exit_slip: int = 20,
    size: int = 1_000,
) -> tuple[LabelSet, Fills, Any]:
    """One hand-built label next to its engine replay, with the fills asserted identical."""
    kw: dict[str, Any] = {"entry_slip_ticks": entry_slip, "exit_slip_ticks": exit_slip, "session_end_ns": cutoff}
    lab = executable_labels(prep, np.array([t], np.int64), side, 0, size, horizon, scenario, **kw)
    fills = label_fills(prep, t, side, 0, size, horizon, scenario, **kw)
    res, ref = one_shot_replay(prep, t, side, 0, size, horizon, scenario, cutoff, entry_slip, exit_slip)
    assert_same_fills(fills, res, ref)
    return lab, fills, res


# ---------------------------------------------------------------------------
# Hand-worked examples (Venue A tick 0.5 USD, lot 100, fee 0.5 bp, L = L_ack = 5 ms)
# ---------------------------------------------------------------------------


def test_full_round_trip_nets_both_fees_with_exact_inverse_cash() -> None:
    lab = one(moving_market(), 1 * MS)
    # entry decided at 1 ms on the observed 40,001 ask, arrives at 6 ms (same book in effect);
    # exit decided at 1 + 30 ms on the 40,010 bid received at 30 ms, arrives at 36 ms, acknowledged at 41 ms
    assert lab.status.tolist() == [FILLED]
    assert lab.decision_ns.tolist() == [1 * MS]
    assert lab.entry_arrival_ns.tolist() == [6 * MS]
    assert lab.entry_qty_units.tolist() == [1_000]
    assert lab.entry_avg_px.tolist() == [inv_avg((1_000, 40_001))]
    assert lab.exit_decision_ns.tolist() == [31 * MS]
    assert lab.exit_qty_units.tolist() == [1_000]
    assert lab.exit_avg_px.tolist() == [inv_avg((1_000, 40_010))]
    assert lab.exit_orders.tolist() == [1]
    assert lab.net_pnl.tolist() == [math.fsum([cash(1, 1_000, 40_001), cash(-1, 1_000, 40_010)])]
    assert lab.gross_pnl.tolist() == [math.fsum([gross(1, 1_000, 40_001), gross(-1, 1_000, 40_010)])]
    assert lab.fees.tolist() == [math.fsum([fee(1_000, 40_001), fee(1_000, 40_010)])]
    assert lab.knowable_ns.tolist() == [41 * MS]
    assert lab.net_pnl[0] > 0.0
    assert (lab.side, lab.venue, lab.size_units, lab.horizon_ns) == (1, 0, 1_000, 30 * MS)


def test_partial_entry_when_the_visible_depth_runs_out() -> None:
    thin = [(40_001, 300), (40_002, 300)]  # only two ask levels displayed
    prep = market((1_000, 500, B0, thin), (30_000, 25_000, B1, A1), (60_000, 55_000, B1, A1))
    lab = one(prep, 1 * MS, entry_slip_ticks=20)
    assert lab.status.tolist() == [PARTIAL]
    assert lab.entry_qty_units.tolist() == [600]
    assert lab.entry_avg_px.tolist() == [inv_avg((300, 40_001), (300, 40_002))]
    assert lab.exit_qty_units.tolist() == [600]  # only the matched quantity is closed and scored
    expected = [cash(1, 300, 40_001), cash(1, 300, 40_002), cash(-1, 600, 40_010)]
    assert lab.net_pnl.tolist() == [math.fsum(expected)]


def test_the_depth_fraction_haircuts_every_level_the_sweep_may_take() -> None:
    thin = [(40_001, 300), (40_002, 300)]
    prep = market((1_000, 500, B0, thin), (30_000, 25_000, B1, A1), (60_000, 55_000, B1, A1))
    half = scen(depth_fraction=0.5)  # floor(0.5 * 300) = 150 -> one whole lot of 100 per level
    lab, fills, _ = label_and_replay(prep, 1 * MS, 1, 29 * MS, half, 60 * MS, entry_slip=20)
    assert lab.status.tolist() == [PARTIAL]
    assert lab.entry_qty_units.tolist() == [200]
    assert lab.entry_avg_px.tolist() == [inv_avg((100, 40_001), (100, 40_002))]
    assert lab.exit_qty_units.tolist() == [200]
    assert fills.qty_units.tolist() == [100, 100, 200]
    assert one(prep, 1 * MS, entry_slip_ticks=20, scenario=scen(depth_fraction=1.0)).entry_qty_units.tolist() == [600]
    thin_bids = [(40_000, 300), (39_999, 300)]  # the bid side is haircut the same way for a sell
    mirror = market((1_000, 500, thin_bids, A0), (30_000, 25_000, B1, A1), (60_000, 55_000, B1, A1))
    short, fills, _ = label_and_replay(mirror, 1 * MS, -1, 29 * MS, half, 60 * MS, entry_slip=20)
    assert short.entry_qty_units.tolist() == [200]
    assert short.entry_avg_px.tolist() == [inv_avg((100, 40_000), (100, 39_999))]
    assert fills.qty_units.tolist() == [100, 100, 200]


def test_partial_entry_when_the_ioc_limit_binds_before_the_depth() -> None:
    asks = [(40_001, 300), (40_002, 300)] + [(40_003 + k, DEEP) for k in range(8)]
    prep = market((1_000, 500, B0, asks), (30_000, 25_000, B1, A1), (60_000, 55_000, B1, A1))
    lab = one(prep, 1 * MS, entry_slip_ticks=1)  # limit 40,002: the deep 40,003 level is out of reach
    assert lab.status.tolist() == [PARTIAL]
    assert lab.entry_qty_units.tolist() == [600]
    assert lab.entry_avg_px.tolist() == [inv_avg((300, 40_001), (300, 40_002))]


def test_no_fill_is_an_observed_zero_outcome_known_at_the_entry_acknowledgement() -> None:
    b1, a1 = ladder(40_004, 40_005, DEEP)
    prep = market((1_000, 500, B0, A0), (9_000, 4_000, b1, a1), (30_000, 25_000, b1, a1))
    lab = one(prep, 1 * MS, horizon_ns=20 * MS)  # limit 40,001 meets the 40,005 ask in effect at 6 ms
    assert lab.status.tolist() == [NO_FILL]
    assert lab.net_pnl.tolist() == [0.0]
    assert lab.gross_pnl.tolist() == [0.0]
    assert lab.fees.tolist() == [0.0]
    assert lab.entry_arrival_ns.tolist() == [6 * MS]
    assert lab.entry_qty_units.tolist() == [0]
    assert np.isnan(lab.entry_avg_px[0]) and np.isnan(lab.exit_avg_px[0])
    assert lab.exit_decision_ns.tolist() == [-1]
    assert lab.exit_orders.tolist() == [0]
    assert lab.knowable_ns.tolist() == [11 * MS]


def test_an_order_arriving_during_a_clear_fills_nothing_but_counts_as_sent() -> None:
    data = market((1_000, 500, B0, A0), (9_000, 4_000, B0, A0), (30_000, 25_000, B0, A0), (60_000, 55_000, B0, A0)).data
    prep = PreparedMarket.build(with_invalid_rows(data, 0, {1: "clear"}))
    # decided at 1 ms on a valid book; at the 6 ms arrival the clear (exchange time 4 ms) is in effect
    lab, fills, _ = label_and_replay(prep, 1 * MS, 1, 29 * MS, scen(), 60 * MS, entry_slip=20)
    assert lab.status.tolist() == [NO_FILL]
    assert lab.entry_arrival_ns.tolist() == [6 * MS]
    assert lab.knowable_ns.tolist() == [11 * MS]
    assert len(fills.order_id) == 0
    recv = one(prep, 1 * MS, horizon_ns=29 * MS, entry_slip_ticks=20, scenario=scen(TimingMode.RECEIPT_CLOCK))
    assert recv.status.tolist() == [FILLED]  # under the receipt clock the clear is only received at 9 ms
    # a one-sided book still shows the asks a buy would sweep; it is unusable for execution all the same
    one_sided = PreparedMarket.build(with_invalid_rows(data, 0, {1: "no_bid"}))
    lab, fills, _ = label_and_replay(one_sided, 1 * MS, 1, 29 * MS, scen(), 60 * MS, entry_slip=20)
    assert lab.status.tolist() == [NO_FILL]
    assert len(fills.order_id) == 0


def test_stale_book_fills_at_the_observed_quote() -> None:
    # no update between the decision (1 ms) and the arrival (6 ms): the sell meets the observed 40,000 bid
    lab = one(moving_market(), 1 * MS, side=-1)
    assert lab.status.tolist() == [FILLED]
    assert lab.entry_avg_px.tolist() == [inv_avg((1_000, 40_000))]
    assert lab.exit_avg_px.tolist() == [inv_avg((1_000, 40_011))]
    assert lab.net_pnl.tolist() == [math.fsum([cash(-1, 1_000, 40_000), cash(1, 1_000, 40_011)])]


def test_the_order_meets_the_book_in_effect_at_arrival() -> None:
    b1, a1 = ladder(40_004, 40_005, DEEP)
    prep = market((1_000, 500, B0, A0), (9_000, 4_000, b1, a1), (30_000, 25_000, b1, a1))
    exch = one(prep, 1 * MS, horizon_ns=20 * MS, entry_slip_ticks=10)
    recv = one(prep, 1 * MS, horizon_ns=20 * MS, entry_slip_ticks=10, scenario=scen(TimingMode.RECEIPT_CLOCK))
    # the book changed at exchange time 4 ms (received at 9 ms): it is in effect at the 6 ms arrival
    assert exch.entry_avg_px.tolist() == [inv_avg((1_000, 40_005))]
    # the receipt-clock comparator only uses books received by the arrival
    assert recv.entry_avg_px.tolist() == [inv_avg((1_000, 40_001))]


def test_a_book_received_exactly_at_the_exit_decision_is_the_observed_book() -> None:
    down_b, down_a = ladder(39_990, 39_991, DEEP)
    prep = market((1_000, 500, B0, A0), (30_000, 29_000, down_b, down_a), (60_000, 59_000, down_b, down_a))
    # the exit decided at 30 ms observes the row received at 30 ms (limit 39,990 at zero slip); on the
    # previous book its limit 40,000 would not fill and the exit would need a retry
    lab, _, _ = label_and_replay(prep, 1 * MS, 1, 29 * MS, scen(), 60 * MS, exit_slip=0)
    assert lab.status.tolist() == [FILLED]
    assert lab.exit_decision_ns.tolist() == [30 * MS]
    assert lab.exit_orders.tolist() == [1]
    assert lab.exit_avg_px.tolist() == [inv_avg((1_000, 39_990))]


def retry_market() -> PreparedMarket:
    thin_bids = [(40_010, 300), (40_009, 300)] + [(40_008 - k, DEEP) for k in range(8)]
    b2, a2 = ladder(40_012, 40_013, DEEP)
    return market(
        (1_000, 500, B0, A0), (30_000, 25_000, thin_bids, A1), (38_000, 37_000, b2, a2), (80_000, 79_000, b2, a2)
    )


def test_a_partial_exit_retries_at_its_acknowledgement_until_flat() -> None:
    lab = one(retry_market(), 1 * MS, exit_slip_ticks=1)
    # exit decided at 31 ms on the 40,010 bid (limit 40,009) sells 600; its acknowledgement at 41 ms is the
    # next decision: the 40,012 bid received at 38 ms takes the remaining 400 (arrival 46 ms, ack 51 ms)
    assert lab.status.tolist() == [FILLED]
    assert lab.exit_decision_ns.tolist() == [31 * MS]
    assert lab.exit_orders.tolist() == [2]
    assert lab.exit_qty_units.tolist() == [1_000]
    assert lab.exit_avg_px.tolist() == [inv_avg((300, 40_010), (300, 40_009), (400, 40_012))]
    assert lab.knowable_ns.tolist() == [51 * MS]
    fills = label_fills(
        retry_market(), 1 * MS, 1, venue=0, size_units=1_000, horizon_ns=30 * MS, scenario=scen(), exit_slip_ticks=1
    )
    assert fills.order_id.tolist() == [0, 1, 1, 2]
    assert fills.side.tolist() == [1, -1, -1, -1]
    assert fills.px_ticks.tolist() == [40_001, 40_010, 40_009, 40_012]
    assert fills.qty_units.tolist() == [1_000, 300, 300, 400]
    assert fills.submit_ns.tolist() == [1 * MS, 31 * MS, 31 * MS, 41 * MS]
    assert fills.arrival_ns.tolist() == [6 * MS, 36 * MS, 36 * MS, 46 * MS]
    assert fills.ack_ns.tolist() == [11 * MS, 41 * MS, 41 * MS, 51 * MS]
    assert lab.net_pnl.tolist() == [math.fsum(fills.cash.tolist())]


def thin_exit_market(side: int) -> PreparedMarket:
    """The exit of a ``side`` entry meets a book with only 600 USD within one tick of the touch, and no
    further book arrives until 200 ms (so every retry meets the same levels)."""
    if side > 0:
        bids = [(40_010, 300), (40_009, 300)] + [(40_008 - k, DEEP) for k in range(8)]
        rows: tuple[Row, ...] = ((1_000, 500, B0, A0), (30_000, 25_000, bids, A1), (200_000, 199_000, bids, A1))
    else:
        asks = [(40_011, 300), (40_012, 300)] + [(40_013 + k, DEEP) for k in range(8)]
        rows = ((1_000, 500, B0, A0), (30_000, 25_000, B1, asks), (200_000, 199_000, B1, asks))
    return market(*rows)


@pytest.mark.parametrize("side", [1, -1])
def test_retries_meet_the_liquidity_they_consumed_only_after_the_ttl(side: int) -> None:
    # exit decided at 30 ms (limit one tick past the touch) takes 600 at 35 ms; the retry at the 40 ms
    # acknowledgement arrives at 45 ms against the same book: our own 600 are still consumed for 1 s ...
    prep = thin_exit_market(side)
    touch, second = (40_010, 40_009) if side > 0 else (40_011, 40_012)
    held, _, res = label_and_replay(prep, 1 * MS, side, 29 * MS, scen(consumption_ttl_ns=S), 200 * MS, exit_slip=1)
    assert held.status.tolist() == [CENSORED]
    assert held.exit_qty_units.tolist() == [600]
    assert held.exit_orders.tolist() == [17]  # decisions at 30, 40, ..., 190 ms; the 200 ms one cannot arrive
    assert not res.settlement.valid
    # ... but a 2 ms TTL (expired at 37 ms) or none at all lets the retry take the same levels again
    for ttl in (2 * MS, 0):
        again, fills, _ = label_and_replay(
            prep, 1 * MS, side, 29 * MS, scen(consumption_ttl_ns=ttl), 200 * MS, exit_slip=1
        )
        assert again.status.tolist() == [FILLED]
        assert again.exit_orders.tolist() == [2]
        assert again.exit_avg_px.tolist() == [inv_avg((300, touch), (300, second), (300, touch), (100, second))]
        assert fills.submit_ns.tolist() == [1 * MS, 30 * MS, 30 * MS, 40 * MS, 40 * MS]


def test_not_flat_by_the_session_end_is_censored_never_zero() -> None:
    lab = one(retry_market(), 1 * MS, exit_slip_ticks=1, session_end_ns=45 * MS)
    # the retry decided at 41 ms would arrive at 46 ms, after the execution cutoff: still long 400 USD
    assert lab.status.tolist() == [CENSORED]
    assert np.isnan(lab.net_pnl[0]) and np.isnan(lab.gross_pnl[0]) and np.isnan(lab.fees[0])
    assert lab.exit_qty_units.tolist() == [600]
    assert lab.exit_orders.tolist() == [1]
    assert lab.knowable_ns.tolist() == [41 * MS]
    capped = one(retry_market(), 1 * MS, exit_slip_ticks=1, max_exit_retries=0)
    assert capped.status.tolist() == [CENSORED]
    assert capped.knowable_ns.tolist() == [41 * MS]


def test_an_arrival_exactly_at_the_cutoff_is_accepted_for_entries_and_exits() -> None:
    prep = moving_market()
    entry_ok, _, _ = label_and_replay(prep, 1 * MS, 1, 29 * MS, scen(), 6 * MS)  # entry arrives at 6 ms
    assert entry_ok.status.tolist() == [CENSORED]  # ... but no exit can
    assert entry_ok.entry_arrival_ns.tolist() == [6 * MS]
    assert entry_ok.entry_qty_units.tolist() == [1_000]
    assert entry_ok.exit_orders.tolist() == [0]
    assert entry_ok.knowable_ns.tolist() == [30 * MS]
    refused, _, _ = label_and_replay(prep, 1 * MS, 1, 29 * MS, scen(), 5 * MS + 999_999)
    assert refused.status.tolist() == [NO_ORDER]
    exit_ok, _, _ = label_and_replay(prep, 1 * MS, 1, 29 * MS, scen(), 35 * MS)  # exit arrives at 35 ms
    assert exit_ok.status.tolist() == [FILLED]
    assert exit_ok.knowable_ns.tolist() == [40 * MS]
    exit_late, _, _ = label_and_replay(prep, 1 * MS, 1, 29 * MS, scen(), 35 * MS - 1)
    assert exit_late.status.tolist() == [CENSORED]
    assert exit_late.exit_orders.tolist() == [0]


def test_the_exit_waits_for_the_entry_acknowledgement() -> None:
    lab = one(moving_market(), 1 * MS, horizon_ns=2 * MS)
    # the position is unknown until the entry's fill report at 11 ms: the exit is decided then
    assert lab.exit_decision_ns.tolist() == [11 * MS]
    assert lab.knowable_ns.tolist() == [21 * MS]


def test_the_acknowledgement_latency_is_separate_from_the_order_latency() -> None:
    slow_ack = scen(ack_latency_ns=20 * MS)  # L = 5 ms, L_ack = 20 ms
    # entry arrives at 6 ms, reported at 26 ms: the exit (horizon 2 ms) is decided at the report on the book
    # observed then (40,000 bid, received at 1 ms) and arrives at 31 ms, where the 40,010 book is in effect
    lab, fills, _ = label_and_replay(moving_market(), 1 * MS, 1, 2 * MS, slow_ack, 60 * MS)
    assert lab.status.tolist() == [FILLED]
    assert lab.entry_arrival_ns.tolist() == [6 * MS]
    assert lab.exit_decision_ns.tolist() == [26 * MS]
    assert lab.exit_avg_px.tolist() == [inv_avg((1_000, 40_010))]
    assert lab.knowable_ns.tolist() == [51 * MS]
    assert fills.arrival_ns.tolist() == [6 * MS, 31 * MS]
    assert fills.ack_ns.tolist() == [26 * MS, 51 * MS]


def test_a_clear_at_the_exit_decision_waits_for_the_venues_next_valid_book() -> None:
    b2, a2 = ladder(40_020, 40_021, DEEP)
    data = market((1_000, 500, B0, A0), (20_000, 19_000, B0, A0), (45_000, 44_000, b2, a2)).data
    prep = PreparedMarket.build(with_invalid_rows(data, 0, {1: "clear"}))
    # the exit due at 20 ms observes the clear: like a blocked engine decision it is taken at the next book
    # (45 ms), arrives at 50 ms and sells at 40,020
    lab, fills, _ = label_and_replay(prep, 1 * MS, 1, 19 * MS, scen(), 100 * MS)
    assert lab.status.tolist() == [FILLED]
    assert lab.exit_decision_ns.tolist() == [45 * MS]
    assert lab.exit_orders.tolist() == [1]
    assert lab.exit_avg_px.tolist() == [inv_avg((1_000, 40_020))]
    assert lab.knowable_ns.tolist() == [55 * MS]
    assert fills.submit_ns.tolist() == [1 * MS, 45 * MS]
    one_sided = PreparedMarket.build(with_invalid_rows(data, 0, {1: "no_bid"}))
    assert one(one_sided, 1 * MS, horizon_ns=19 * MS, session_end_ns=100 * MS).exit_decision_ns.tolist() == [45 * MS]


def test_censoring_by_the_end_of_the_data_is_known_once_no_order_can_arrive_in_time() -> None:
    data = market((1_000, 500, B0, A0), (20_000, 19_000, B0, A0), (25_000, 24_000, B0, A0)).data
    prep = PreparedMarket.build(with_invalid_rows(data, 0, {2: "clear"}))  # the venue's last row is a clear
    lab = one(prep, 1 * MS, horizon_ns=30 * MS, session_end_ns=500 * MS)
    # the exit due at 31 ms waits for a book that never comes; that is certain only at 495 ms, the last
    # instant from which an order could still arrive by the cutoff
    assert lab.status.tolist() == [CENSORED]
    assert lab.exit_orders.tolist() == [0]
    assert lab.exit_decision_ns.tolist() == [-1]
    assert lab.knowable_ns.tolist() == [495 * MS]
    early_cutoff = one(prep, 1 * MS, horizon_ns=30 * MS, session_end_ns=20 * MS)
    assert early_cutoff.status.tolist() == [CENSORED]
    assert early_cutoff.knowable_ns.tolist() == [31 * MS]  # never before the decision that found no book


def test_no_order_without_an_observed_book_or_past_the_execution_cutoff() -> None:
    t = np.array([500 * US, 1 * MS, 58 * MS], np.int64)
    lab = executable_labels(moving_market(), t, 1, 0, 1_000, 2 * MS, scen(), session_end_ns=60 * MS)
    # 0.5 ms: nothing received yet; 58 ms: the entry would arrive at 63 ms, after the cutoff
    assert lab.status.tolist() == [NO_ORDER, FILLED, NO_ORDER]
    assert lab.net_pnl[[0, 2]].tolist() == [0.0, 0.0]
    assert lab.entry_arrival_ns[[0, 2]].tolist() == [-1, -1]
    assert lab.knowable_ns[[0, 2]].tolist() == [500 * US, 58 * MS]
    assert lab.trainable.tolist() == [False, True, False]


def test_long_and_short_labels_are_separate_outcomes_not_negatives() -> None:
    static = market((1_000, 500, B0, A0), (60_000, 55_000, B0, A0))
    assert one(static, 1 * MS, side=1).net_pnl[0] < 0.0  # both directions pay the spread and two fees
    assert one(static, 1 * MS, side=-1).net_pnl[0] < 0.0
    up_long, up_short = one(moving_market(), 1 * MS, side=1), one(moving_market(), 1 * MS, side=-1)
    assert up_long.net_pnl[0] > 0.0 > up_short.net_pnl[0]
    assert up_long.net_pnl[0] + up_short.net_pnl[0] < 0.0


def test_empty_decisions_give_an_empty_label_set() -> None:
    lab = executable_labels(moving_market(), np.zeros(0, np.int64), 1, 0, 1_000, MS, scen())
    assert len(lab) == 0 and lab.status.dtype == np.int8


def test_zero_total_latency_is_rejected_before_any_label_is_built() -> None:
    # With L = L_ack = 0 the retry clock could never advance (an exit that fills nothing more would be
    # re-sent at the same instant). Scenario now refuses that round trip outright, so the kernel's own
    # censor guard is unreachable through the public API; a one-sided zero latency is still allowed.
    with pytest.raises(InvalidConfigError):
        scen(latency_ms=0)
    lab = one(moving_market(), 1 * MS, scenario=Scenario("one-sided", DEMO_BASE_FEES, 0, ack_latency_ns=1 * MS))
    assert lab.status.tolist() == [FILLED]


@pytest.mark.parametrize(
    "override",
    [
        {"side": 0},
        {"side": 2},
        {"venue": 1},
        {"venue": -1},
        {"size_units": 150},
        {"size_units": 0},
        {"size_units": 1_000_100},
        {"horizon_ns": -1},
        {"entry_slip_ticks": -1},
        {"exit_slip_ticks": -1},
        {"max_exit_retries": -1},
        {"decision_t": np.array([1.0e6])},
        {"decision_t": np.ones((1, 1), np.int64)},
        {"scenario": Scenario("huge", DEMO_BASE_FEES, 2**62)},
        {"scenario": Scenario("huge", DEMO_BASE_FEES, 2**60, ack_latency_ns=1)},
        {"scenario": Scenario("huge", DEMO_BASE_FEES, 1, consumption_ttl_ns=2**61 + 1)},
    ],
)
def test_invalid_label_requests_are_rejected(override: dict[str, Any]) -> None:
    args: dict[str, Any] = {
        "decision_t": np.array([MS], np.int64),
        "side": 1,
        "venue": 0,
        "size_units": 1_000,
        "horizon_ns": MS,
        "scenario": scen(),
    }
    args.update(override)
    with pytest.raises(InvalidConfigError):
        executable_labels(moving_market(), **args)


def test_latencies_that_could_wrap_the_int64_clock_are_rejected_everywhere() -> None:
    # 2**62 + 2**61 + 2**60 ns (decision + horizon + both latencies) is the largest clock the kernel forms
    huge = Scenario("huge", DEMO_BASE_FEES, 2**61, ack_latency_ns=2**61)
    with pytest.raises(InvalidConfigError):
        label_fills(moving_market(), MS, 1, 0, 1_000, MS, huge)
    with pytest.raises(InvalidConfigError):
        hold_value_labels(moving_market(), np.array([MS], np.int64), 1_000, 0, MS, huge)
    edge = Scenario("edge", DEMO_BASE_FEES, 2**60 - 1, ack_latency_ns=1, consumption_ttl_ns=2**61)
    lab = executable_labels(moving_market(), np.array([2**62], np.int64), 1, 0, 1_000, 2**61, edge)
    assert lab.status.tolist() == [NO_ORDER]  # the entry would arrive long after the cutoff


# ---------------------------------------------------------------------------
# Hold value of an existing position
# ---------------------------------------------------------------------------


def test_hold_value_is_the_difference_of_two_liquidations() -> None:
    t = np.full(3, 1 * MS, np.int64)
    hv = hold_value_labels(moving_market(), t, np.array([1_000, -1_000, 0]), 0, 30 * MS, scen(), exit_slip_ticks=20)
    now_long, later_long = cash(-1, 1_000, 40_000), cash(-1, 1_000, 40_010)
    now_short, later_short = cash(1, 1_000, 40_001), cash(1, 1_000, 40_011)
    assert hv.now.cash.tolist() == [now_long, now_short, 0.0]
    assert hv.later.cash.tolist() == [later_long, later_short, 0.0]
    assert hv.value.tolist() == [later_long - now_long, later_short - now_short, 0.0]
    assert hv.value[0] > 0.0 > hv.value[1]
    assert hv.status.tolist() == [FILLED, FILLED, NO_ORDER]
    assert hv.trainable.tolist() == [True, True, False]
    assert hv.now.first_order_ns.tolist() == [1 * MS, 1 * MS, -1]
    assert hv.later.first_order_ns.tolist() == [31 * MS, 31 * MS, -1]
    assert hv.later.avg_px.tolist()[:2] == [inv_avg((1_000, 40_010)), inv_avg((1_000, 40_011))]
    assert hv.knowable_ns.tolist() == [41 * MS, 41 * MS, 31 * MS]


def test_hold_value_is_known_only_when_both_liquidations_are() -> None:
    # liquidating now (31 ms) on the thin 40,010 book needs a retry acknowledged at 51 ms; holding 7 ms longer
    # closes in one order on the 40,012 book received at 38 ms (acknowledged at 48 ms)
    hv = hold_value_labels(retry_market(), np.array([31 * MS], np.int64), 1_000, 0, 7 * MS, scen(), exit_slip_ticks=1)
    assert hv.now.orders.tolist() == [2]
    assert hv.now.knowable_ns.tolist() == [51 * MS]
    assert hv.later.orders.tolist() == [1]
    assert hv.later.knowable_ns.tolist() == [48 * MS]
    assert hv.knowable_ns.tolist() == [51 * MS]


def test_holding_for_no_time_is_worth_exactly_nothing() -> None:
    hv = hold_value_labels(moving_market(), np.array([1 * MS, 31 * MS], np.int64), 1_000, 0, 0, scen())
    assert hv.value.tolist() == [0.0, 0.0]


def test_hold_value_is_censored_when_the_later_liquidation_cannot_finish() -> None:
    hv = hold_value_labels(
        moving_market(), np.array([1 * MS], np.int64), 1_000, 0, 30 * MS, scen(), session_end_ns=33 * MS
    )
    # the liquidation decided at 31 ms would arrive at 36 ms, after the 33 ms cutoff
    assert hv.now.status.tolist() == [FILLED]
    assert hv.later.status.tolist() == [CENSORED]
    assert hv.status.tolist() == [CENSORED]
    assert np.isnan(hv.value[0])


def test_hold_value_rejects_positions_off_the_lot_grid_or_over_the_cap() -> None:
    for position in (150, 1_000_100, -1_000_100):
        with pytest.raises(InvalidConfigError):
            hold_value_labels(moving_market(), np.array([MS], np.int64), position, 0, MS, scen())


# ---------------------------------------------------------------------------
# Oracle: a one-shot threshold policy replayed by the engine (and its reference twin)
# ---------------------------------------------------------------------------


def oracle_market(seed: int, venue: int, invalid_frac: float) -> tuple[MarketData, PreparedMarket]:
    """A random market with a fraction of the venue's book rows made unusable (clears, one-sided books)."""
    data = random_market(seed)
    if invalid_frac > 0.0:
        rng = np.random.default_rng(seed + 1)
        n = len(data.venues[venue].book)
        rows = np.flatnonzero(rng.random(n) < invalid_frac)
        hows = rng.choice(["clear", "no_bid", "no_ask"], size=rows.shape[0])
        data = with_invalid_rows(data, venue, dict(zip(rows.tolist(), hows.tolist(), strict=True)))
    return data, PreparedMarket.build(data)


@given(
    seed=st.integers(0, 10_000),
    venue=st.sampled_from([0, 1]),
    side=st.sampled_from([1, -1]),
    lots=st.integers(1, 80),
    latency_ms=st.sampled_from([1, 5, 50]),
    ack_ms=st.sampled_from([0, 1, 3, 20, 80]),
    ttl=st.sampled_from([0, 2 * MS, S]),
    depth=st.sampled_from([1.0, 0.5, 0.3]),
    invalid_frac=st.sampled_from([0.0, 0.0, 0.15, 0.4]),
    timing=st.sampled_from(list(TimingMode)),
    entry_slip=st.integers(0, 3),
    exit_slip=st.integers(0, 6),
    start=st.floats(0.0, 0.8),
    gap=st.integers(0, 250),
    cutoff_ms=st.one_of(st.none(), st.integers(-60, 120)),
)
@settings(max_examples=120, deadline=None)
def test_labels_equal_a_one_shot_replay_exactly(
    seed: int,
    venue: int,
    side: int,
    lots: int,
    latency_ms: int,
    ack_ms: int,
    ttl: int,
    depth: float,
    invalid_frac: float,
    timing: TimingMode,
    entry_slip: int,
    exit_slip: int,
    start: float,
    gap: int,
    cutoff_ms: int | None,
) -> None:
    """Alignment: the policy enters on the single cohort at ``t`` and leaves after ``max_hold = horizon``.
    The engine only decides at receipt cohorts and acknowledgements, so ``t + horizon`` is chosen as a cohort
    time (the label's exit clock is the wall clock). ``t`` is the receipt of one of the venue's valid books:
    on an invalid observed book the engine would enter later while a label reports NO_ORDER (documented).
    Cutoffs are drawn around the exit so that refused entries, refused retries and censoring are covered;
    invalid rows example blocked exit decisions and orders arriving during a clear."""
    data, prep = oracle_market(seed, venue, invalid_frac)
    recv = prep.stream.recv_ns
    n = len(recv)
    valid_recv = data.venues[venue].book.recv_ns[prep.valid[venue]]
    t = int(valid_recv[int(start * (len(valid_recv) - 1))])
    e = int(np.searchsorted(recv, t))
    horizon = int(recv[min(e + gap, n - 1)]) - t
    lat = latency_ms * MS
    end = int(recv[-1])
    cutoff = end if cutoff_ms is None else max(t + 2, t + horizon + cutoff_ms * MS)
    size = lots * data.venues[venue].instrument.lot_size
    scenario = Scenario(
        "oracle",
        DEMO_BASE_FEES,
        lat,
        ack_latency_ns=ack_ms * MS,
        timing=timing,
        consumption_ttl_ns=ttl,
        depth_fraction=depth,
    )
    res, ref = one_shot_replay(prep, t, side, venue, size, horizon, scenario, cutoff, entry_slip, exit_slip)
    kw: dict[str, Any] = {"entry_slip_ticks": entry_slip, "exit_slip_ticks": exit_slip, "session_end_ns": cutoff}
    lab = executable_labels(prep, np.array([t], np.int64), side, venue, size, horizon, scenario, **kw)
    fills = label_fills(prep, t, side, venue, size, horizon, scenario, **kw)
    assert_same_fills(fills, res, ref)
    status = int(lab.status[0])
    if res.settlement.valid:
        assert status != CENSORED
        assert lab.net_pnl[0] == res.settlement.net_pnl
        assert lab.fees[0] == res.settlement.fees[venue]
    else:
        assert res.settlement.reason == "not_flat"
        assert status == CENSORED and np.isnan(lab.net_pnl[0])
    entry = res.orders.reason == REASON_ENTRY
    exits = ~entry
    assert int(lab.exit_orders[0]) == int(exits.sum())
    if not entry.any():
        assert status == NO_ORDER and lab.net_pnl[0] == 0.0
        return
    assert lab.entry_arrival_ns[0] == res.orders.arrival_ns[entry][0]
    assert lab.entry_qty_units[0] == res.orders.filled_units[entry][0]
    assert lab.exit_qty_units[0] == res.orders.filled_units[exits].sum()
    assert lab.exit_decision_ns[0] == (res.orders.submit_ns[exits][0] if exits.any() else -1)
    if status != CENSORED:
        assert lab.knowable_ns[0] == res.orders.ack_ns.max()
        assert status == (
            NO_FILL if lab.entry_qty_units[0] == 0 else FILLED if lab.entry_qty_units[0] == size else PARTIAL
        )
    # the label aggregates are the exact sums of its own fills
    entry_fills, exit_fills = fills.order_id == 0, fills.order_id > 0
    if entry_fills.any():
        assert lab.entry_avg_px[0] == (
            fills.qty_units[entry_fills].sum()
            / math.fsum((fills.qty_units[entry_fills] / (fills.px_ticks[entry_fills] * prep.tick[venue])).tolist())
        )
    if exit_fills.any():
        assert lab.exit_avg_px[0] == (
            fills.qty_units[exit_fills].sum()
            / math.fsum((fills.qty_units[exit_fills] / (fills.px_ticks[exit_fills] * prep.tick[venue])).tolist())
        )
    if status != CENSORED:
        assert lab.net_pnl[0] == math.fsum(fills.cash.tolist())


@given(
    seed=st.integers(0, 10_000),
    venue=st.sampled_from([0, 1]),
    side=st.sampled_from([1, -1]),
    lots=st.integers(1, 40),
    horizon_ms=st.integers(10, 300),
    start=st.floats(0.0, 0.7),
)
@settings(max_examples=25, deadline=None)
def test_liquidating_the_acquired_position_reproduces_the_exit_leg(
    seed: int, venue: int, side: int, lots: int, horizon_ms: int, start: float
) -> None:
    prep = PreparedMarket.build(random_market(seed))
    recv = prep.stream.recv_ns
    lo = int(np.searchsorted(recv, prep.book_recv[venue][0]))
    t = int(recv[lo + int(start * (len(recv) - 1 - lo))])
    size = lots * int(prep.lot[venue])
    horizon = horizon_ms * MS  # >= L + L_ack, so the exit is decided at exactly t + horizon
    lab = executable_labels(prep, np.array([t], np.int64), side, venue, size, horizon, scen(), exit_slip_ticks=3)
    if lab.entry_qty_units[0] == 0:
        return
    fills = label_fills(prep, t, side, venue, size, horizon, scen(), exit_slip_ticks=3)
    hv = hold_value_labels(
        prep, np.array([t], np.int64), side * int(lab.entry_qty_units[0]), venue, horizon, scen(), exit_slip_ticks=3
    )
    exit_fills = fills.order_id > 0
    assert hv.later.qty_units[0] == lab.exit_qty_units[0]
    assert hv.later.orders[0] == lab.exit_orders[0]
    assert hv.later.first_order_ns[0] == lab.exit_decision_ns[0]
    assert hv.later.knowable_ns[0] == lab.knowable_ns[0]
    if lab.status[0] != CENSORED:
        assert hv.later.cash[0] == math.fsum(fills.cash[exit_fills].tolist())


# ---------------------------------------------------------------------------
# Causality, determinism, compiled/pure-Python parity, exact summation
# ---------------------------------------------------------------------------


def _altered_after(data: MarketData, cut: int) -> MarketData:
    """Shift prices and halve sizes of every book row that both happened and was received after ``cut``."""
    venues = []
    for tapes in data.venues:
        book = tapes.book
        late = ((book.recv_ns > cut) & (book.exch_ns > cut))[:, None]
        new_book = dataclasses.replace(
            book,
            recv_ns=book.recv_ns.copy(),
            exch_ns=book.exch_ns.copy(),
            kind=book.kind.copy(),
            n_bid=book.n_bid.copy(),
            n_ask=book.n_ask.copy(),
            bid_px=np.where(late & (book.bid_px > 0), book.bid_px + 7, book.bid_px).astype(np.int32),
            ask_px=np.where(late & (book.ask_px > 0), book.ask_px + 7, book.ask_px).astype(np.int32),
            bid_sz=np.where(late, (book.bid_sz // 200) * 100, book.bid_sz).astype(np.int32),
            ask_sz=np.where(late, (book.ask_sz // 200) * 100, book.ask_sz).astype(np.int32),
        )
        venues.append(dataclasses.replace(tapes, book=new_book))
    return MarketData(tuple(venues))


@pytest.mark.parametrize("timing", list(TimingMode))
def test_labels_never_depend_on_data_after_they_are_known(timing: TimingMode) -> None:
    """A label is a function of the data up to its ``knowable_ns``: rows that happened and were received
    after ``T + horizon + L + L_ack`` cannot change any label decided before ``T`` that is known by then."""
    data = random_market(21)
    prep = PreparedMarket.build(data)
    recv = prep.stream.recv_ns
    big_t = int(recv[len(recv) // 2])
    horizon, lat = 40 * MS, 5 * MS
    cut = big_t + horizon + 2 * lat
    decisions = recv[(recv >= prep.book_recv[0][0]) & (recv < big_t)][::2]
    late = recv[recv > cut + 10 * MS][::5]
    altered = PreparedMarket.build(_altered_after(data, cut))
    for side in (1, -1):
        a = executable_labels(prep, decisions, side, 0, 500, horizon, scen(timing), exit_slip_ticks=20)
        b = executable_labels(altered, decisions, side, 0, 500, horizon, scen(timing), exit_slip_ticks=20)
        resolved = a.knowable_ns <= cut
        assert resolved.mean() > 0.9  # the claim is not vacuous
        for name in ARRAY_FIELDS:
            np.testing.assert_array_equal(getattr(a, name)[resolved], getattr(b, name)[resolved], err_msg=name)
        moved_a = executable_labels(prep, late, side, 0, 500, horizon, scen(timing))
        moved_b = executable_labels(altered, late, side, 0, 500, horizon, scen(timing))
        assert not np.array_equal(moved_a.entry_avg_px, moved_b.entry_avg_px)  # the alteration is effective


def test_labels_are_deterministic_and_order_independent() -> None:
    prep = PreparedMarket.build(random_market(4))
    recv = prep.stream.recv_ns
    t = recv[recv >= prep.book_recv[1][0]][::3]
    a = executable_labels(prep, t, -1, 1, 2_000, 25 * MS, scen(), exit_slip_ticks=2)
    perm = np.random.default_rng(0).permutation(len(t))
    b = executable_labels(prep, t[perm], -1, 1, 2_000, 25 * MS, scen(), exit_slip_ticks=2)
    for name in ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(a, name)[perm], getattr(b, name), err_msg=name)
    assert np.all(a.knowable_ns >= a.decision_ns)
    assert a.trainable.tolist() == (a.status <= NO_FILL).tolist()
    with pytest.raises(ValueError, match="read-only"):
        a.net_pnl[0] = 1.0
    assert t.flags.writeable  # results are frozen copies; the caller's input is untouched


def test_compiled_and_pure_python_kernels_agree_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    _, prep = oracle_market(5, 1, 0.2)
    recv = prep.stream.recv_ns
    t = recv[recv >= prep.book_recv[1][0]][::7]
    scenario = scen(latency_ms=50, ack_latency_ns=3 * MS, consumption_ttl_ns=2 * MS, depth_fraction=0.5)
    kw: dict[str, Any] = {"exit_slip_ticks": 1, "session_end_ns": int(recv[-1]) - 80 * MS}
    fast = executable_labels(prep, t, -1, 1, 3_000, 30 * MS, scenario, **kw)
    hold_fast = hold_value_labels(prep, t, 2_000, 1, 30 * MS, scen(), exit_slip_ticks=1)
    for name in (
        "_label_kernel",
        "_ioc",
        "_add_fills",
        "_record_fills",
        "_fsum_add",
        "_fsum_value",
        "sweep",
        "fill_cash",
    ):
        fn = getattr(labels_mod, name)
        monkeypatch.setattr(labels_mod, name, getattr(fn, "py_func", fn))
    slow = executable_labels(prep, t, -1, 1, 3_000, 30 * MS, scenario, **kw)
    hold_slow = hold_value_labels(prep, t, 2_000, 1, 30 * MS, scen(), exit_slip_ticks=1)
    statuses = set(slow.status.tolist())
    assert int(CENSORED) in statuses and statuses & {int(FILLED), int(PARTIAL)}  # both branches are exampled
    assert int(NO_ORDER) in statuses and int(NO_FILL) in statuses  # and the invalid-book paths
    assert int(slow.exit_orders.max()) > 1  # and the retry path
    for name in ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(fast, name), getattr(slow, name), err_msg=name)
    np.testing.assert_array_equal(hold_fast.value, hold_slow.value)


def _exact_sum(values: list[float]) -> float:
    parts = np.zeros(labels_mod.N_PARTIALS, np.float64)
    n = 0
    for x in values:
        n = labels_mod._fsum_add(parts, n, x)
    return float(labels_mod._fsum_value(parts, n))


FLOATS = st.one_of(
    st.floats(-1e3, 1e3, allow_nan=False),
    st.floats(-1e-6, 1e-6, allow_nan=False),
    st.sampled_from([1e-16, 1.0, 1e16, -1e16, 0.1, 0.2, 0.3, 2.0**-60, -(2.0**53)]),
    st.floats(-1e300, 1e300, allow_nan=False),
)


@given(st.lists(FLOATS, max_size=25))
@settings(max_examples=300, deadline=None)
def test_the_kernel_sum_is_math_fsum(values: list[float]) -> None:
    assert _exact_sum(values) == math.fsum(values)
    py_add = getattr(labels_mod._fsum_add, "py_func", labels_mod._fsum_add)
    py_value = getattr(labels_mod._fsum_value, "py_func", labels_mod._fsum_value)
    parts = np.zeros(labels_mod.N_PARTIALS, np.float64)
    n = 0
    for x in values:
        n = py_add(parts, n, x)
    assert py_value(parts, n) == math.fsum(values)


def test_the_kernel_sum_rounds_half_even_across_partials() -> None:
    assert _exact_sum([1e-16, 1.0, 1e16]) == math.fsum([1e-16, 1.0, 1e16]) == 10000000000000002.0


def test_the_partials_buffer_holds_every_non_overlapping_expansion() -> None:
    """Non-overlapping partials can be one bit wide, so a sum may hold far more than 2098 / 53 of them:
    sign-mixed values across the exponent range keep about 90 live. The buffer must take them all (a
    compiled kernel would otherwise write past its row without any error)."""
    rng = np.random.default_rng(7)
    n_vals = 20_000
    values = np.ldexp(
        rng.uniform(1.0, 2.0, n_vals) * rng.choice([-1.0, 1.0], n_vals), rng.integers(-1000, 1001, n_vals)
    ).tolist()
    py_add = getattr(labels_mod._fsum_add, "py_func", labels_mod._fsum_add)
    py_value = getattr(labels_mod._fsum_value, "py_func", labels_mod._fsum_value)
    parts = np.zeros(labels_mod.N_PARTIALS, np.float64)
    n = peak = 0
    for x in values:
        n = py_add(parts, n, x)
        peak = max(peak, n)
    assert peak > 64  # the former capacity
    assert peak <= 2098 <= labels_mod.N_PARTIALS  # one partial per bit position of the double range at most
    assert py_value(parts, n) == math.fsum(values)
    assert _exact_sum(values) == math.fsum(values)
