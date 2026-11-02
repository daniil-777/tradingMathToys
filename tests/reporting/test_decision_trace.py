"""Decision trace: received observations, submission, exchange arrival, fills and
acknowledgements on one timeline, with the controller-visible and hidden clocks labelled."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from financeai.core.clock import MISSING_NS
from financeai.core.errors import InvalidConfigError
from financeai.core.serialization import loads_strict
from financeai.domains.market.engine import (
    PreparedMarket,
    ReplayResult,
    ThresholdParams,
    run_replay,
)
from financeai.domains.market.instruments import DEMO_INVERSE_A
from financeai.domains.market.scenario import DEMO_BASE_FEES, RiskLimits, Scenario, SessionPlan
from financeai.domains.market.stream import MarketData
from financeai.domains.market.tape import BOOK_CLEAR
from financeai.reporting import decision_trace_figure

from ..market.synthetic import Row, ladder, random_market, tapes_from_rows

MS = 1_000_000
SCENARIO = Scenario("base", DEMO_BASE_FEES, 5 * MS)


def one_venue_replay() -> tuple[ReplayResult, PreparedMarket]:
    b0, a0 = ladder(40_000, 40_001, 10_000)
    b1, a1 = ladder(40_004, 40_005, 10_000)
    rows: list[Row] = [(1_000, 500, b0, a0), (9_000, 4_000, b1, a1), (30_000, 25_000, b1, a1)]
    prep = PreparedMarket.build(MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),)))
    session = SessionPlan(
        start_ns=0, entry_cutoff_ns=20 * MS, flatten_start_ns=25 * MS, exec_cutoff_ns=40 * MS, end_ns=60 * MS
    )
    params = ThresholdParams(
        direction=(1,), size_units=(1_000,), enter=0.5, exit=-10.0, entry_slip_ticks=10, exit_slip_ticks=20
    )
    res = run_replay(prep, np.array([1.0, 0.0, 0.0]), params, SCENARIO, session, RiskLimits())
    return res, prep


def test_trace_shows_an_order_meeting_a_book_the_controller_had_not_seen(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    res, prep = one_venue_replay()
    art = decision_trace_figure(res, prep, 0, 60 * MS, tmp_path / "trace.png", scenario=SCENARIO)
    m = loads_strict(art.manifest_path.read_text())
    assert m == art.manifest
    assert m["kind"] == "decision_trace" and m["schema_version"] == 1
    assert m["window"]["t0_ns"] == 0 and m["window"]["t1_ns"] == 60 * MS
    assert m["counts"] == {"books": 3, "trades": 0, "orders": 2, "fills": 2}
    assert m["timing_mode"] == "EXCHANGE_CLOCK" and m["scenario"] == "base"
    visibility = {lane["clock"]: lane["visibility"] for lane in m["lanes"]}
    assert visibility == {
        "recv_ns": "visible",
        "exch_ns": "hidden",
        "submit_ns": "visible",
        "arrival_ns": "hidden",
        "ack_ns": "visible",
    }
    first = m["orders"][0]
    assert (first["side"], first["qty_units"], first["filled_units"]) == ("BUY", 1_000, 1_000)
    assert (first["submit_ns"], first["arrival_ns"], first["ack_ns"]) == (1 * MS, 6 * MS, 11 * MS)
    # Decided at 1 ms on the 40,001 ask; at exchange arrival (6 ms) the book stamped 4 ms by the
    # exchange was in force, but the controller only receives it at 9 ms.
    assert first["observed_book_row"] == 0
    assert (first["book_row"], first["book_exch_ns"], first["book_recv_ns"]) == (1, 4 * MS, 9 * MS)
    assert first["book_hidden_at_submit"] is True
    assert first["fills"] == [{"px_ticks": 40_005, "price_units": 20_002.5, "qty_units": 1_000}]
    assert m["output_sha256"] == art.sha256


def test_trace_labels_visible_and_hidden_clocks(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    res, prep = one_venue_replay()
    art = decision_trace_figure(
        res, prep, 0, 60 * MS, tmp_path / "trace.svg", title="One decision", evidence_scope="SYNTHETIC"
    )
    text = art.image_path.read_text()
    for label in ("[visible]", "[hidden]", "order submitted", "exchange arrival", "acknowledged", "SYNTHETIC"):
        assert label in text
    assert art.manifest["timing_mode"] is None  # no scenario given: not guessed


def test_trace_window_on_a_two_venue_replay(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    prep = PreparedMarket.build(random_market(9, n_rows=300))
    stream = prep.stream
    signal = np.random.default_rng(10).normal(0.0, 1.0, size=len(stream))
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
    res = run_replay(prep, signal, params, SCENARIO, session, RiskLimits())
    assert len(res.orders) >= 2
    t0 = int(res.orders.submit_ns[0]) - 2 * MS
    t1 = int(res.orders.ack_ns[1]) + 2 * MS
    art = decision_trace_figure(res, prep, t0, t1, tmp_path / "trace.png", scenario=SCENARIO)
    o = res.orders
    inside = (
        ((o.submit_ns >= t0) & (o.submit_ns <= t1))
        | ((o.ack_ns >= t0) & (o.ack_ns <= t1))
        | ((o.arrival_ns >= t0) & (o.arrival_ns <= t1))
    )
    assert art.manifest["counts"]["orders"] == int(inside.sum())
    assert [x["order_id"] for x in art.manifest["orders"]] == o.order_id[inside].tolist()
    assert art.manifest["counts"]["fills"] == int(np.isin(res.fills.order_id, o.order_id[inside]).sum())


@pytest.mark.parametrize(("t0", "t1"), [(10, 10), (20, 10), (0.0, 10), (True, 10)])
def test_trace_window_is_validated(tmp_path: Path, t0: Any, t1: Any) -> None:
    res, prep = one_venue_replay()
    with pytest.raises(InvalidConfigError):
        decision_trace_figure(res, prep, t0, t1, tmp_path / "trace.png")


def test_trace_refuses_an_unreadably_wide_window(tmp_path: Path) -> None:
    res, prep = one_venue_replay()
    with pytest.raises(InvalidConfigError):
        decision_trace_figure(res, prep, 0, 60 * MS, tmp_path / "trace.png", max_events=4)
    assert not (tmp_path / "trace.png").exists()


def test_a_window_without_orders_shows_observations_only(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    res, prep = one_venue_replay()
    # The stretch before the first decision (1 ms) holds the first book only through its exchange
    # stamp (0.5 ms). The engine's flatten timer submits at flatten_start_ns (25 ms), the third book's
    # exchange stamp, so no later stretch with an observation is free of order events.
    art = decision_trace_figure(res, prep, 0, MS - 1, tmp_path / "quiet.png")
    assert art.manifest["counts"] == {"books": 1, "trades": 0, "orders": 0, "fills": 0}
    assert art.manifest["orders"] == []


def test_an_order_meeting_a_book_clear_without_exchange_clock(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    b0, a0 = ladder(40_000, 40_001, 10_000)
    rows: list[Row] = [(1_000, 500, b0, a0), (3_000, 2_000, b0, a0), (30_000, 25_000, b0, a0)]
    tapes = tapes_from_rows(DEMO_INVERSE_A, rows)
    book = tapes.book
    kind = book.kind.copy()
    kind[1] = BOOK_CLEAR
    exch = book.exch_ns.copy()
    exch[1] = MISSING_NS
    cleared = dataclasses.replace(
        book,
        recv_ns=book.recv_ns.copy(),
        exch_ns=exch,
        kind=kind,
        n_bid=book.n_bid.copy(),
        n_ask=book.n_ask.copy(),
        bid_px=book.bid_px.copy(),
        bid_sz=book.bid_sz.copy(),
        ask_px=book.ask_px.copy(),
        ask_sz=book.ask_sz.copy(),
    )
    prep = PreparedMarket.build(MarketData((dataclasses.replace(tapes, book=cleared),)))
    session = SessionPlan(
        start_ns=0, entry_cutoff_ns=20 * MS, flatten_start_ns=25 * MS, exec_cutoff_ns=40 * MS, end_ns=60 * MS
    )
    params = ThresholdParams(direction=(1,), size_units=(1_000,), enter=0.5, exit=-10.0, entry_slip_ticks=10)
    res = run_replay(prep, np.array([1.0, 0.0, 0.0]), params, SCENARIO, session, RiskLimits())
    art = decision_trace_figure(res, prep, 0, 20 * MS, tmp_path / "clear.png")
    order = art.manifest["orders"][0]
    assert order["book_row"] == 1 and order["book_exch_ns"] is None
    assert order["book_recv_ns"] == 3 * MS and order["filled_units"] == 0 and order["fills"] == []
