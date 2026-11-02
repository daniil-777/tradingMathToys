"""Deterministic teaching markets, evaluated by the real compiled and reference engines.

These small policies explain the flow and lead-gap families. They are deliberately
simplified examples, not evidence of future market performance.
No plotting dependency is needed to build or verify their decision traces.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from typing import Any

import numpy as np

from financeai.domains.market.engine import PreparedMarket, ReplayResult, ThresholdParams, run_replay
from financeai.domains.market.instruments import ContractType, InstrumentSpec
from financeai.domains.market.reference import ThresholdPolicy, run_reference
from financeai.domains.market.scenario import RiskLimits, Scenario, SessionPlan
from financeai.domains.market.stream import MarketData
from financeai.domains.market.tape import DEPTH, BookTape, MarketTapes, ParseReport, TradeTape
from financeai.strategies.leadlag import lead_gap_signal
from financeai.strategies.linear import decayed_trade_flow

NS = 1_000_000_000
VENUE_A = InstrumentSpec("venue_a", "DEMO-USD", Decimal("0.01"), contract_type=ContractType.LINEAR)
VENUE_B = InstrumentSpec("venue_b", "DEMO-USD", Decimal("0.01"), contract_type=ContractType.LINEAR)
MOVIE_SCENARIO = Scenario("teaching", {i.key: Decimal("0.0001") for i in (VENUE_A, VENUE_B)}, 5_000_000)


@dataclasses.dataclass(frozen=True)
class Demo:
    key: str
    market: PreparedMarket
    signal: np.ndarray
    params: ThresholdParams
    session: SessionPlan
    result: ReplayResult


def _tape(
    spec: InstrumentSpec,
    seconds: np.ndarray,
    bids: np.ndarray,
    trades: tuple[tuple[float, int, int], ...] = (),
    *,
    exchange: np.ndarray | None = None,
) -> MarketTapes:
    """One-tick spreads, ten levels, 1,000 units shown at each level; integer contract quantities."""
    recv = np.rint(seconds * NS).astype(np.int64)
    bid_ticks = np.rint(bids / spec.tick_float).astype(np.int32)
    n = len(recv)
    levels = np.arange(DEPTH, dtype=np.int32)[None, :]
    book = BookTape(
        recv,
        recv.copy() if exchange is None else np.rint(exchange * NS).astype(np.int64),
        np.zeros(n, np.int8),
        np.full(n, DEPTH, np.int8),
        np.full(n, DEPTH, np.int8),
        bid_ticks[:, None] - levels,
        np.full((n, DEPTH), 1_000, np.int32),
        bid_ticks[:, None] + 1 + levels,
        np.full((n, DEPTH), 1_000, np.int32),
    )
    tr = TradeTape(
        np.array([round(t * NS) for t, _, _ in trades], np.int64),
        np.array([round(t * NS) for t, _, _ in trades], np.int64),
        np.array([bid_ticks[np.searchsorted(seconds, t, side="right") - 1] for t, _, _ in trades], np.int32),
        np.array([qty for _, qty, _ in trades], np.int32),
        np.array([side for _, _, side in trades], np.int8),
    )
    return MarketTapes(spec, book, tr, ParseReport(n, 0, len(trades), len(trades), 0, 0))


def _evaluate(
    key: str, market: PreparedMarket, signal: np.ndarray, params: ThresholdParams, session: SessionPlan
) -> Demo:
    scenario, limits = MOVIE_SCENARIO, RiskLimits(100)
    result = run_replay(market, signal, params, scenario, session, limits)
    reference = run_reference(market.data, signal, ThresholdPolicy(params), scenario, session, limits)
    # Compare every order and fill field, including submission / arrival / acknowledgement.
    for name in ("orders", "fills"):
        fast, slow = getattr(result, name), getattr(reference, name)
        for column in dataclasses.fields(fast):
            np.testing.assert_array_equal(getattr(fast, column.name), getattr(slow, column.name))
    assert result.settlement == reference.settlement
    assert result.settlement.valid and all(p == 0 for p in result.settlement.final_position_units)
    return Demo(key, market, signal, params, session, result)


def flow_demo() -> Demo:
    t = np.arange(401) / 20
    price = np.interp(t, [0, 3, 5, 8, 11, 14, 18, 20], [100.0, 100.0, 100.03, 100.08, 100.06, 100.0, 99.98, 99.98])
    tape = _tape(VENUE_A, t, price, ((3.0, 1_800, 1), (11.0, 2_200, -1)))
    market = PreparedMarket.build(MarketData((tape,)))
    signal = decayed_trade_flow(market, venue=0, tau_ns=NS)
    params = ThresholdParams(
        direction=(1,),
        size_units=(100,),
        enter=1_500,
        exit=-1e9,
        max_hold_ns=900 * NS,
        entry_slip_ticks=0,
        exit_slip_ticks=10,
        reverse=True,
        refresh_hold=True,
    )
    return _evaluate("flow", market, signal, params, SessionPlan(0, 17 * NS, 18 * NS, 19 * NS, 20 * NS))


def lead_gap_demo() -> Demo:
    t = np.arange(71) / 20
    leader = np.where(t >= 1.5, 100.02, 100.0)
    follower = np.where(t >= 2.3, 100.02, 100.0)
    market = PreparedMarket.build(MarketData((_tape(VENUE_A, t, follower), _tape(VENUE_B, t, leader))))
    signal = lead_gap_signal(market, leader=1, follower=0, window_ns=NS, depth_units=100, gate_ticks=1)
    params = ThresholdParams(
        direction=(1, 0),
        size_units=(100, 0),
        enter=0.75,
        exit=0.05,
        max_hold_ns=NS,
        min_hold_ns=100_000_000,
        entry_slip_ticks=0,
        exit_slip_ticks=10,
    )
    return _evaluate(
        "lead_gap", market, signal, params, SessionPlan(NS, 2_400_000_000, 3 * NS, 3_400_000_000, 3_500_000_000)
    )


def missed_fill_demo() -> Demo:
    # The new quote exists at exchange time 2.003 s but arrives locally only at 2.010 s.
    t = np.array([0.0, 1.0, 2.0, 2.010, 2.5, 3.0])
    exchange = np.array([0.0, 1.0, 2.0, 2.003, 2.5, 3.0])
    tape = _tape(
        VENUE_A, t, np.array([100.0, 100.0, 100.0, 100.02, 100.02, 100.02]), ((2.0, 1_800, 1),), exchange=exchange
    )
    market = PreparedMarket.build(MarketData((tape,)))
    signal = decayed_trade_flow(market, venue=0, tau_ns=NS)
    params = ThresholdParams(
        direction=(1,),
        size_units=(100,),
        enter=1_500,
        exit=-1e9,
        max_hold_ns=900 * NS,
        entry_slip_ticks=0,
        exit_slip_ticks=10,
    )
    return _evaluate(
        "missed_fill", market, signal, params, SessionPlan(0, 2_100_000_000, 2_500_000_000, 2_900_000_000, 3 * NS)
    )


def position_at(demo: Demo, time_ns: int, *, acknowledged: bool = True) -> int:
    fills = demo.result.fills
    clock = fills.ack_ns if acknowledged else fills.arrival_ns
    return int(np.sum(fills.side[clock <= time_ns].astype(np.int64) * fills.qty_units[clock <= time_ns]))


def signal_at(demo: Demo, time_ns: int) -> float:
    i = int(np.searchsorted(demo.market.stream.recv_ns, time_ns, side="right")) - 1
    return float(demo.signal[i]) if i >= 0 else 0.0


def build_demos() -> dict[str, Demo]:
    demos = {d.key: d for d in (flow_demo(), lead_gap_demo(), missed_fill_demo())}
    flow, lead, miss = (demos[k] for k in ("flow", "lead_gap", "missed_fill"))
    np.testing.assert_array_equal(flow.result.orders.side, [1, -1, -1, 1])
    np.testing.assert_array_equal(flow.result.orders.reason, [1, 2, 1, 3])
    assert position_at(flow, 3_005_000_000) == 0
    assert position_at(flow, 3_005_000_000, acknowledged=False) == 100
    assert position_at(flow, 3_010_000_000) == 100
    assert position_at(flow, 11_010_000_000) == 0  # close acknowledged before opening short
    assert position_at(flow, 11_020_000_000) == -100
    assert signal_at(flow, 4 * NS) < flow.params.enter and position_at(flow, 4 * NS) == 100
    np.testing.assert_array_equal(lead.result.orders.side, [1, -1])
    np.testing.assert_array_equal(lead.result.orders.submit_ns, [1_500_000_000, 2_300_000_000])
    assert lead.result.fills.px_ticks[-1] > lead.result.fills.px_ticks[0]
    assert lead.result.settlement.net_pnl is not None and lead.result.settlement.net_pnl < 0
    assert len(miss.result.orders) == 1 and miss.result.orders.filled_units[0] == 0
    assert len(miss.result.fills) == 0
    assert miss.result.orders.limit_ticks[0] < miss.market.apx[0][miss.result.orders.book_row[0], 0]
    return demos


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def evidence(demos: dict[str, Demo]) -> dict[str, Any]:
    """Small, reviewable provenance record: inputs, exact order/fill ledgers, and settlement."""
    return {
        "schema": "financeai.teaching_movie/1",
        "evidence_scope": "SYNTHETIC",
        "policies": "Educational examples on invented markets.",
        "scenario": MOVIE_SCENARIO.to_mapping(),
        "position_cap_units": 100,
        "verification": "Every order/fill column and settlement matches the independent Python reference.",
        "demos": {
            key: {
                "params": _plain(d.params),
                "session": _plain(d.session),
                "signal": _plain(d.signal),
                "receipt_ns": _plain(d.market.stream.recv_ns),
                "tapes": _plain(d.market.data.venues),
                "orders": _plain(d.result.orders),
                "fills": _plain(d.result.fills),
                "settlement": _plain(d.result.settlement),
            }
            for key, d in demos.items()
        },
    }
