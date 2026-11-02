"""Transfer checks for the gated composite and acknowledgement-based confirmation."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from financeai.domains.market.engine import PreparedMarket, ThresholdParams, run_replay
from financeai.domains.market.instruments import DEMO_INVERSE_A, DEMO_INVERSE_B
from financeai.domains.market.reference import ControllerView, PolicyAction, ThresholdPolicy, run_reference
from financeai.domains.market.scenario import DEMO_BASE_FEES, RiskLimits, Scenario, SessionPlan
from financeai.domains.market.stream import MarketData
from financeai.strategies.leadlag import build_signal
from tests.market.synthetic import ladder, tapes_from_rows

MS = 1_000_000
SECOND = 1000 * MS
BMX = "venue_a/ALPHA-USD"
DBT = "venue_b/ALPHA-PERP"

# The selected family's exact entry recipe, kept local so the test has no
# dependency on mutable report files or research-cache availability.
SIGNAL = {
    "kind": "any",
    "components": [
        {
            "scale": 1.0,
            "spec": {
                "kind": "sweep",
                "source": BMX,
                "min_units": 100_000,
                "gate": {"venue": BMX, "tau_ms": 10_000, "min_units": 250_000},
            },
        },
        {
            "scale": 2.5,
            "spec": {
                "kind": "lead_gap",
                "leader": DBT,
                "follower": BMX,
                "window_ms": 100,
                "depth_units": 50_000,
                "gate_ticks": 1,
                "gate": {"venue": BMX, "tau_ms": 10_000, "min_units": 500_000},
            },
        },
    ],
}


def confirmation_market() -> MarketData:
    venues = []
    for spec in (DEMO_INVERSE_A, DEMO_INVERSE_B):
        rows = []
        for ms in range(0, 20_001, 100):
            price = 39_980 if spec == DEMO_INVERSE_A and 1000 <= ms < 6000 else 40_000
            rows.append((ms * 1000, max(0, ms * 1000 - 500), *ladder(price, price + 1, 1_000_000)))
        venues.append(tapes_from_rows(spec, rows))
    # Complete tied cohorts deliberately include both books and multiple trades.
    trade_ms = np.array([500, 500, 7000, 7000, 10_000, 10_000, 12_000, 12_000, 15_000, 15_000], np.int64)
    trades = dataclasses.replace(
        venues[0].trades,
        recv_ns=trade_ms * MS,
        exch_ns=trade_ms * MS - 100_000,
        px=np.array([40_001, 40_002] * 5, np.int32),
        sz=np.full(len(trade_ms), 300_000, np.int32),
        side_label=np.ones(len(trade_ms), np.int8),
    )
    venues[0] = dataclasses.replace(venues[0], trades=trades)
    return MarketData(tuple(venues))


def change_future(data: MarketData, cut: int) -> MarketData:
    venues = []
    for tapes in data.venues:
        book, trades = tapes.book, tapes.trades
        late_book, late_trade = book.recv_ns > cut, trades.recv_ns > cut
        assert np.all(book.exch_ns[late_book] > cut)
        assert np.all(trades.exch_ns[late_trade] > cut)
        shift = np.where(late_book, 200, 0).astype(np.int32)[:, None]
        new_book = dataclasses.replace(
            book,
            bid_px=book.bid_px + shift,
            ask_px=book.ask_px + shift,
            bid_sz=np.where(late_book[:, None], book.bid_sz // 2, book.bid_sz).astype(np.int32),
            ask_sz=np.where(late_book[:, None], book.ask_sz // 2, book.ask_sz).astype(np.int32),
        )
        new_trades = dataclasses.replace(
            trades,
            px=np.where(late_trade, trades.px + 200, trades.px).astype(np.int32),
            sz=np.where(late_trade, trades.sz * 2, trades.sz).astype(np.int32),
            side_label=np.where(late_trade, -trades.side_label, trades.side_label).astype(np.int8),
        )
        venues.append(dataclasses.replace(tapes, book=new_book, trades=new_trades))
    return MarketData(tuple(venues))


def assert_ledger_equal(left, right) -> None:
    for name in ("orders", "fills"):
        a, b = getattr(left, name), getattr(right, name)
        for field in dataclasses.fields(a):
            np.testing.assert_array_equal(getattr(a, field.name), getattr(b, field.name))


@pytest.mark.parametrize("hold_ms,early_ms,refresh", [(12_000, 1000, True), (90_000, 4000, False)])
def test_selected_composite_and_confirmation_ignore_changed_future_books_and_trades(hold_ms, early_ms, refresh):
    data, cut = confirmation_market(), 10 * SECOND
    changed = change_future(data, cut)
    prep, other = PreparedMarket.build(data), PreparedMarket.build(changed)
    for field in dataclasses.fields(prep.stream):
        np.testing.assert_array_equal(getattr(prep.stream, field.name), getattr(other.stream, field.name))
    prefix = prep.stream.recv_ns <= cut  # preserves every event in the boundary cohort
    assert np.count_nonzero(prep.stream.recv_ns == cut) >= 4
    signal, altered = build_signal(prep, SIGNAL), build_signal(other, SIGNAL)
    np.testing.assert_array_equal(signal[prefix], altered[prefix])
    assert np.any(np.abs(signal[prefix]) >= 1)
    assert not np.array_equal(signal[~prefix], altered[~prefix])  # mutation affects actual inputs
    params = ThresholdParams(
        direction=(1, 0),
        size_units=(1_000_000, 0),
        enter=1.0,
        exit=-1e9,
        entry_slip_ticks=0,
        exit_slip_ticks=10,
        depth_ticks=1,
        max_hold_ns=hold_ms * MS,
        refresh_hold=refresh,
        early_check_ns=early_ms * MS,
        early_markout_bps=-2.0,
    )
    # Same absolute session clocks in both runs: never derive cutoffs from a changed prefix.
    session = SessionPlan(0, 16 * SECOND, 18 * SECOND, 19 * SECOND, 20 * SECOND)
    scenario = Scenario("causality", DEMO_BASE_FEES, 5 * MS)
    results = []
    for prepared, values in ((prep, signal), (other, altered)):
        fast = run_replay(prepared, values, params, scenario, session, RiskLimits())
        reference = run_reference(prepared.data, values, ThresholdPolicy(params), scenario, session, RiskLimits())
        assert_ledger_equal(fast, reference)
        results.append(fast)
    before, after = results
    prefix_orders = before.orders.submit_ns <= cut
    assert np.count_nonzero(prefix_orders) >= 3
    assert np.any((before.orders.reason == 2) & prefix_orders)  # early confirmation acted
    for field in dataclasses.fields(before.orders):
        np.testing.assert_array_equal(
            getattr(before.orders, field.name)[prefix_orders],
            getattr(after.orders, field.name)[after.orders.submit_ns <= cut],
        )
    for field in dataclasses.fields(before.fills):
        np.testing.assert_array_equal(
            getattr(before.fills, field.name)[before.fills.ack_ns <= cut],
            getattr(after.fills, field.name)[after.fills.ack_ns <= cut],
        )


def test_confirmation_clock_and_fill_price_wait_for_acknowledgement():
    rows = [
        (
            ms * 1000,
            max(0, ms * 1000 - 500),
            *ladder(40_000 if ms < 10 else 39_980, 40_001 if ms < 10 else 39_981, 1_000_000),
        )
        for ms in range(0, 131, 5)
    ]
    data = MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),))
    prep = PreparedMarket.build(data)
    signal = np.zeros(len(prep.stream))
    signal[0] = 1
    params = ThresholdParams(
        direction=(1,), size_units=(1000,), enter=0.5, exit=-1e9, early_check_ns=10 * MS, early_markout_bps=-2.0
    )
    observed = []

    class Recording(ThresholdPolicy):
        def act(self, view: ControllerView) -> PolicyAction:
            action = super().act(view)
            observed.append(
                (
                    view.t,
                    self.origin,
                    self.entry_qty,
                    self.entry_inv,
                    self.early_done,
                    tuple(f.ack_ns for f in view.acked_fills),
                    action.targets_units,
                )
            )
            return action

    scenario = Scenario("slow-ack", DEMO_BASE_FEES, 5 * MS, ack_latency_ns=50 * MS)
    session = SessionPlan(0, 80 * MS, 90 * MS, 120 * MS, 130 * MS)
    reference = run_reference(data, signal, Recording(params), scenario, session, RiskLimits())
    compiled = run_replay(prep, signal, params, scenario, session, RiskLimits())
    assert_ledger_equal(reference, compiled)
    assert reference.fills.arrival_ns[0] == 5 * MS
    assert reference.fills.ack_ns[0] == 55 * MS
    before_ack = [row for row in observed if 5 * MS <= row[0] < 55 * MS]
    assert len(before_ack) >= 8  # many adverse visible books while the entry fill remains hidden
    assert all(row[1:6] == (None, 0, 0.0, False, ()) and row[6] == (1000,) for row in before_ack)
    acknowledged = next(row for row in observed if row[0] == 55 * MS)
    assert acknowledged[1:3] == (55 * MS, 1000)
    assert acknowledged[4:6] == (False, (55 * MS,))
    assert reference.orders.submit_ns.tolist() == [0, 65 * MS]
    assert reference.orders.reason.tolist() == [1, 2]
