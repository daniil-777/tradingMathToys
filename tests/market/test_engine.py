"""Replay engine contract: latency/clock semantics, acknowledgements, caps, session
cut-offs, causality, determinism and exact fast/reference parity."""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from financeai.core.errors import InvalidConfigError
from financeai.domains.market.engine import ReplayResult, ThresholdParams, run_replay
from financeai.domains.market.instruments import DEMO_INVERSE_A, DEMO_INVERSE_B
from financeai.domains.market.reference import ControllerView, FillReport, PolicyAction, ThresholdPolicy, run_reference
from financeai.domains.market.scenario import (
    DEMO_BASE_FEES,
    RiskLimits,
    Scenario,
    SessionPlan,
    TimingMode,
)
from financeai.domains.market.stream import EventStream, MarketData
from financeai.domains.market.tape import DEPTH

from .synthetic import US, ladder, random_market, tapes_from_rows

MS = 1_000_000


def one_venue_market() -> MarketData:
    b0, a0 = ladder(40_000, 40_001, 10_000)
    b1, a1 = ladder(40_004, 40_005, 10_000)
    rows = [
        (1_000, 500, b0, a0),  # seen at 1 ms
        (9_000, 4_000, b1, a1),  # happened at 4 ms (exchange), seen at 9 ms
        (30_000, 25_000, b1, a1),
    ]
    return MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),))


SESSION = SessionPlan(
    start_ns=0,
    entry_cutoff_ns=20_000 * US,
    flatten_start_ns=25_000 * US,
    exec_cutoff_ns=40_000 * US,
    end_ns=60_000 * US,
)
BUY_ONCE = ThresholdParams(
    direction=(1,), size_units=(1_000,), enter=0.5, exit=-10.0, entry_slip_ticks=10, exit_slip_ticks=20
)


def scenario(
    timing: TimingMode = TimingMode.EXCHANGE_CLOCK, latency_ms: int = 5, ack_ms: int | None = None
) -> Scenario:
    return Scenario(
        "t", DEMO_BASE_FEES, latency_ms * MS, ack_latency_ns=None if ack_ms is None else ack_ms * MS, timing=timing
    )


LATENCIES = st.sampled_from([(1, None), (5, None), (50, None), (0, 1), (0, 5), (5, 0), (1, 3), (50, 5)])
"""(order latency, acknowledgement latency) in ms; ``None`` acknowledges after the order latency again. A
zero order latency is valid on its own; only the zero round trip (both 0) is rejected."""


def test_a_zero_round_trip_is_rejected() -> None:
    """Order and acknowledgement latency both 0: an order sent at ``t`` would be acknowledged at ``t`` and an
    exit that cannot complete resubmitted at ``t`` again, so neither engine's clock could ever advance."""
    with pytest.raises(InvalidConfigError):
        Scenario("z", DEMO_BASE_FEES, 0)  # the acknowledgement latency defaults to the order latency
    with pytest.raises(InvalidConfigError):
        Scenario("z", DEMO_BASE_FEES, 0, ack_latency_ns=0)
    assert Scenario("z", DEMO_BASE_FEES, 0, ack_latency_ns=MS).ack_ns == MS
    assert Scenario("z", DEMO_BASE_FEES, MS, ack_latency_ns=0).ack_ns == 0


def test_order_meets_the_book_in_effect_at_exchange_arrival() -> None:
    data = one_venue_market()
    signal = np.array([1.0, 0.0, 0.0])
    res = run_replay(data, signal, BUY_ONCE, scenario(), SESSION, RiskLimits())
    # Decided at 1 ms on the 40,001 ask, but the order arrives at exchange time 6 ms when the
    # book (changed at exchange time 4 ms) already shows 40,005.
    assert res.fills.px_ticks.tolist() == [40_005, 40_004]
    assert res.fills.side.tolist() == [1, -1]
    # The forced flatten is a session-timer decision at exactly flatten_start (25 ms).
    assert res.fills.submit_ns.tolist() == [1_000 * US, 25_000 * US]
    assert res.fills.arrival_ns.tolist() == [6_000 * US, 30_000 * US]
    assert res.fills.ack_ns.tolist() == [11_000 * US, 35_000 * US]
    assert res.settlement.valid and res.settlement.final_position_units == (0,)


def test_receipt_clock_comparator_uses_books_received_by_arrival() -> None:
    res = run_replay(
        one_venue_market(),
        np.array([1.0, 0.0, 0.0]),
        BUY_ONCE,
        scenario(TimingMode.RECEIPT_CLOCK),
        SESSION,
        RiskLimits(),
    )
    assert res.fills.px_ticks.tolist() == [40_001, 40_004]


def test_tight_ioc_limit_misses_a_moved_market() -> None:
    params = dataclasses.replace(BUY_ONCE, entry_slip_ticks=0)
    res = run_replay(one_venue_market(), np.array([1.0, 0.0, 0.0]), params, scenario(), SESSION, RiskLimits())
    assert len(res.fills) == 0
    assert res.orders.filled_units.tolist() == [0]  # one-shot entry: no chasing
    assert res.settlement.valid and res.settlement.net_pnl == 0.0


def test_entry_is_one_shot_while_the_signal_persists() -> None:
    res = run_replay(one_venue_market(), np.array([1.0, 1.0, 1.0]), BUY_ONCE, scenario(), SESSION, RiskLimits())
    assert res.orders.side.tolist() == [1, -1]


def test_target_is_clipped_to_the_position_cap() -> None:
    b, a = ladder(40_000, 40_001, 3_000_000)
    rows = [(1_000, 500, b, a), (30_000, 25_000, b, a)]
    data = MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),))
    params = dataclasses.replace(BUY_ONCE, size_units=(2_000_000,))
    res = run_replay(data, np.array([1.0, 0.0]), params, scenario(), SESSION, RiskLimits(1_000_000))
    assert res.orders.qty_units.tolist() == [1_000_000, 1_000_000]
    assert res.settlement.max_abs_position_units == (1_000_000,)


def test_orders_that_would_arrive_after_the_execution_cutoff_are_not_sent() -> None:
    late = dataclasses.replace(SESSION, exec_cutoff_ns=29_000 * US)
    res = run_replay(one_venue_market(), np.array([1.0, 0.0, 0.0]), BUY_ONCE, scenario(), late, RiskLimits())
    assert res.orders.side.tolist() == [1]  # every flatten order (>= 25 ms + 5 ms > 29 ms) is refused
    assert not res.settlement.valid and res.settlement.reason == "not_flat"
    assert res.settlement.net_pnl is None


def test_no_entries_before_start_or_after_entry_cutoff() -> None:
    late_start = dataclasses.replace(SESSION, start_ns=2_000 * US)
    res = run_replay(one_venue_market(), np.array([1.0, 0.0, 0.0]), BUY_ONCE, scenario(), late_start, RiskLimits())
    assert len(res.orders) == 0


def test_event_stream_is_ordered_by_receipt_with_deterministic_ties() -> None:
    data = random_market(3)
    stream = EventStream.build(data)
    assert np.all(np.diff(stream.recv_ns) >= 0)
    keys = np.stack([stream.recv_ns, stream.venue, stream.kind, stream.row])
    order = np.lexsort(keys[::-1])
    assert np.array_equal(order, np.arange(len(stream)))


RANDOM_PARAMS = st.builds(
    ThresholdParams,
    direction=st.sampled_from([(1, 0), (0, 1), (1, -1), (-1, 1), (1, 1)]),
    size_units=st.tuples(st.integers(1, 30).map(lambda k: 100 * k), st.integers(1, 300).map(lambda k: 10 * k)),
    enter=st.floats(0.2, 1.5),
    exit=st.floats(-1.0, 0.5),
    allow_short=st.booleans(),
    min_hold_ns=st.integers(0, 20).map(lambda k: k * MS),
    max_hold_ns=st.integers(1, 400).map(lambda k: k * MS),
    cooldown_ns=st.integers(0, 50).map(lambda k: k * MS),
    stop_bps=st.sampled_from([0.0, 1.0, 3.0]),
    entry_slip_ticks=st.integers(0, 3),
    exit_slip_ticks=st.integers(0, 8),
    size_units_cap=st.integers(1, 3),
    depth_ticks=st.sampled_from([-1, 0, 2]),
    max_clips=st.integers(1, 3),
    clip_spacing_ns=st.sampled_from([0, 5 * MS]),
    refresh_hold=st.booleans(),
    reverse=st.booleans(),
    abs_cap_ns=st.sampled_from([0, 0, 3 * MS, 15 * MS, 60 * MS]),
    early_check_ns=st.sampled_from([0, 0, 1 * MS, 8 * MS, 30 * MS]),
    early_markout_bps=st.one_of(st.just(0.0), st.floats(-4.0, 4.0)),
)
CLOCKED_PARAMS = st.builds(
    dataclasses.replace,
    RANDOM_PARAMS,
    abs_cap_ns=st.integers(1, 80).map(lambda k: k * MS),
    early_check_ns=st.integers(1, 40).map(lambda k: k * MS),
    early_markout_bps=st.floats(-6.0, 6.0),
)


def assert_same_replay(fast: ReplayResult, ref: ReplayResult) -> None:
    """Exact agreement of every fill and order column, the settlement and the decision count."""
    for table in ("fills", "orders"):
        a, b = getattr(fast, table), getattr(ref, table)
        for f in dataclasses.fields(a):
            np.testing.assert_array_equal(getattr(a, f.name), getattr(b, f.name), err_msg=f"{table}.{f.name}")
    assert fast.settlement == ref.settlement
    assert fast.decisions == ref.decisions


def _random_replays(
    seed: int,
    params: ThresholdParams,
    latency: tuple[int, int | None],
    timing: TimingMode,
    separate_exit: bool,
    depth_fraction: float,
) -> tuple[ReplayResult, ReplayResult]:
    data = random_market(seed)
    stream = EventStream.build(data)
    rng = np.random.default_rng(seed + 1)
    signal = rng.normal(0.0, 1.0, size=len(stream))
    exit_signal = rng.normal(0.0, 1.0, size=len(stream)) if separate_exit else None
    first, last = int(stream.recv_ns[0]), int(stream.recv_ns[-1])
    session = SessionPlan(first + 20 * MS, last - 400 * MS, last - 300 * MS, last - 60 * MS, last)
    scen = dataclasses.replace(scenario(timing, *latency), depth_fraction=depth_fraction)
    fast = run_replay(data, signal, params, scen, session, RiskLimits(5_000), stream=stream, exit_signal=exit_signal)
    ref = run_reference(
        data, signal, ThresholdPolicy(params), scen, session, RiskLimits(5_000), stream=stream, exit_signal=exit_signal
    )
    return fast, ref


@given(
    seed=st.integers(0, 10_000),
    params=RANDOM_PARAMS,
    latency=LATENCIES,
    timing=st.sampled_from(list(TimingMode)),
    separate_exit=st.booleans(),
    depth_fraction=st.sampled_from([1.0, 1.0, 0.8, 0.5, 0.2]),
)
@settings(max_examples=60, deadline=None)
def test_fast_engine_matches_reference_engine_exactly(
    seed: int,
    params: ThresholdParams,
    latency: tuple[int, int | None],
    timing: TimingMode,
    separate_exit: bool,
    depth_fraction: float,
) -> None:
    assert_same_replay(*_random_replays(seed, params, latency, timing, separate_exit, depth_fraction))


@given(
    seed=st.integers(0, 10_000),
    params=CLOCKED_PARAMS,
    latency=LATENCIES,
    timing=st.sampled_from(list(TimingMode)),
    separate_exit=st.booleans(),
    depth_fraction=st.floats(0.05, 1.0),
)
@settings(max_examples=40, deadline=None)
def test_engines_agree_exactly_with_the_duration_cap_and_early_check_on(
    seed: int,
    params: ThresholdParams,
    latency: tuple[int, int | None],
    timing: TimingMode,
    separate_exit: bool,
    depth_fraction: float,
) -> None:
    assert_same_replay(*_random_replays(seed, params, latency, timing, separate_exit, depth_fraction))


def test_decisions_never_depend_on_later_market_data() -> None:
    data = random_market(11)
    stream = EventStream.build(data)
    signal = np.random.default_rng(5).normal(0.0, 1.0, size=len(stream))
    first, last = int(stream.recv_ns[0]), int(stream.recv_ns[-1])
    session = SessionPlan(first + 20 * MS, last - 400 * MS, last - 300 * MS, last - 60 * MS, last)
    params = ThresholdParams(
        direction=(1, -1),
        size_units=(1_000, 1_000),
        enter=0.8,
        exit=0.0,
        max_hold_ns=200 * MS,
        entry_slip_ticks=2,
        exit_slip_ticks=5,
    )
    cut = int(stream.recv_ns[len(stream) // 2])
    mutated = []
    for tapes in data.venues:
        book = tapes.book
        late = (book.recv_ns > cut) & (book.exch_ns > cut)
        shift = np.where(late, 7, 0).astype(np.int32)[:, None]
        bid_px = np.where(book.bid_px > 0, book.bid_px + shift, 0).astype(np.int32)
        ask_px = np.where(book.ask_px > 0, book.ask_px + shift, 0).astype(np.int32)
        new_book = dataclasses.replace(
            book,
            recv_ns=book.recv_ns.copy(),
            exch_ns=book.exch_ns.copy(),
            kind=book.kind.copy(),
            n_bid=book.n_bid.copy(),
            n_ask=book.n_ask.copy(),
            bid_px=bid_px,
            bid_sz=book.bid_sz.copy(),
            ask_px=ask_px,
            ask_sz=book.ask_sz.copy(),
        )
        mutated.append(dataclasses.replace(tapes, book=new_book))
    altered = MarketData(tuple(mutated))
    scen = scenario()
    a = run_replay(data, signal, params, scen, session, RiskLimits(), stream=stream)
    b = run_replay(altered, signal, params, scen, session, RiskLimits(), stream=EventStream.build(altered))
    early_a = a.orders.submit_ns < cut
    early_b = b.orders.submit_ns < cut
    for name in ("venue", "side", "qty_units", "limit_ticks", "submit_ns"):
        np.testing.assert_array_equal(getattr(a.orders, name)[early_a], getattr(b.orders, name)[early_b])


def test_replay_is_deterministic() -> None:
    data = random_market(7)
    stream = EventStream.build(data)
    signal = np.random.default_rng(8).normal(size=len(stream))
    first, last = int(stream.recv_ns[0]), int(stream.recv_ns[-1])
    session = SessionPlan(first, last - 400 * MS, last - 300 * MS, last - 60 * MS, last)
    params = ThresholdParams(direction=(1, 1), size_units=(500, 500), enter=0.5, exit=0.0)
    a = run_replay(data, signal, params, scenario(), session, RiskLimits(), stream=stream)
    b = run_replay(data, signal, params, scenario(), session, RiskLimits(), stream=stream)
    np.testing.assert_array_equal(a.fills.cash, b.fills.cash)
    assert a.settlement == b.settlement


def test_invalid_parameters_are_rejected() -> None:
    from financeai.core.errors import InvalidConfigError

    with pytest.raises(InvalidConfigError):
        ThresholdParams(direction=(1,), size_units=(1_000, 1_000), enter=1.0, exit=0.0)
    with pytest.raises(InvalidConfigError):
        ThresholdParams(direction=(2,), size_units=(1_000,), enter=1.0, exit=0.0)
    with pytest.raises(InvalidConfigError):
        ThresholdParams(direction=(1,), size_units=(1_000,), enter=0.0, exit=0.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"abs_cap_ns": -1},
        {"abs_cap_ns": 1.5},
        {"abs_cap_ns": True},
        {"early_check_ns": -1},
        {"early_check_ns": False},
        {"early_markout_bps": float("nan")},
        {"early_markout_bps": float("inf")},
        {"early_markout_bps": True},
        {"early_markout_bps": "1"},
    ],
)
def test_invalid_clock_options_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(InvalidConfigError):
        dataclasses.replace(BUY_ONCE, **bad)  # type: ignore[arg-type]


def test_clock_options_default_off_and_accept_negative_markout_thresholds() -> None:
    assert (BUY_ONCE.abs_cap_ns, BUY_ONCE.early_check_ns, BUY_ONCE.early_markout_bps) == (0, 0, 0.0)
    params = dataclasses.replace(BUY_ONCE, abs_cap_ns=MS, early_check_ns=MS, early_markout_bps=-1.5)
    assert params.early_markout_bps == -1.5


def test_early_check_requires_a_traded_reference_venue() -> None:
    """The markout is priced from the episode's own entry fills on the reference venue (the first with a
    non-zero direction): with no size there it could never be observed and the option would be inert."""
    with pytest.raises(InvalidConfigError):
        ThresholdParams(direction=(1, 1), size_units=(0, 1_000), enter=0.5, exit=0.0, early_check_ns=10 * MS)
    ThresholdParams(direction=(0, 1), size_units=(0, 1_000), enter=0.5, exit=0.0, early_check_ns=10 * MS)
    ThresholdParams(direction=(1, 1), size_units=(0, 1_000), enter=0.5, exit=0.0, abs_cap_ns=10 * MS)  # no price needed


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.0000001, float("nan"), float("inf"), True, "0.5"])
def test_invalid_depth_fractions_are_rejected(fraction: object) -> None:
    with pytest.raises(InvalidConfigError):
        dataclasses.replace(scenario(), depth_fraction=fraction)  # type: ignore[arg-type]


def test_depth_fraction_defaults_to_the_full_displayed_depth() -> None:
    assert scenario().depth_fraction == 1.0
    assert dataclasses.replace(scenario(), depth_fraction=1).depth_fraction == 1.0
    assert dataclasses.replace(scenario(), depth_fraction=1e-9).depth_fraction == 1e-9


def test_depth_aware_entry_takes_only_the_observed_best_level() -> None:
    b, a = ladder(40_000, 40_001, 7_000)
    rows = [(1_000, 500, b, a), (30_000, 25_000, b, a)]
    data = MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),))
    params = dataclasses.replace(BUY_ONCE, size_units=(100_000,), depth_ticks=0)
    res = run_replay(data, np.array([1.0, 0.0]), params, scenario(), SESSION, RiskLimits())
    assert res.orders.qty_units.tolist()[0] == 7_000  # capped at the 7,000 USD best ask


def test_strength_sizing_scales_units_with_the_signal() -> None:
    b, a = ladder(40_000, 40_001, 100_000)
    rows = [(1_000, 500, b, a), (30_000, 25_000, b, a)]
    data = MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),))
    params = dataclasses.replace(BUY_ONCE, size_units=(1_000,), size_units_cap=3, entry_slip_ticks=10)
    res = run_replay(data, np.array([1.6, 0.0]), params, scenario(), SESSION, RiskLimits())
    assert res.orders.qty_units.tolist()[0] == 3_000  # floor(1.6 / 0.5) = 3 units, capped at 3


def test_separate_exit_signal_closes_on_gap_closure() -> None:
    data = one_venue_market()
    params = dataclasses.replace(BUY_ONCE, exit=0.0)
    entry = np.array([1.0, 1.0, 1.0])
    exit_sig = np.array([1.0, 1.0, -1.0])  # closure only at the third event
    late_flatten = dataclasses.replace(SESSION, flatten_start_ns=45_000 * US, exec_cutoff_ns=50_000 * US)
    res = run_replay(data, entry, params, scenario(), late_flatten, RiskLimits(), exit_signal=exit_sig)
    assert res.orders.side.tolist() == [1, -1]
    assert res.orders.reason.tolist()[1] == 2  # a signal exit, not the forced flatten
    assert res.orders.submit_ns.tolist()[1] == 30_000 * US
    # the same run driven by the entry signal alone never exits on signal (exit at flatten)
    alone = run_replay(data, entry, params, scenario(), late_flatten, RiskLimits())
    assert alone.orders.reason.tolist()[-1] == 3


def test_staged_accumulation_adds_clips_while_the_trigger_persists() -> None:
    b, a = ladder(40_000, 40_001, 100_000)
    rows = [(1_000, 500, b, a), (12_000, 11_500, b, a), (24_000, 23_500, b, a), (40_000, 39_000, b, a)]
    data = MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),))
    session = SessionPlan(
        start_ns=0,
        entry_cutoff_ns=30_000 * US,
        flatten_start_ns=35_000 * US,
        exec_cutoff_ns=50_000 * US,
        end_ns=70_000 * US,
    )
    params = dataclasses.replace(BUY_ONCE, size_units=(1_000,), max_clips=3, clip_spacing_ns=10 * MS)
    res = run_replay(data, np.array([1.0, 1.0, 1.0, 0.0]), params, scenario(), session, RiskLimits())
    # clip 1 at 1 ms, clip 2 at 12 ms (spacing 10 ms passed), clip 3 at 24 ms; flatten sells 3,000
    assert res.orders.side.tolist() == [1, 1, 1, -1]
    assert res.orders.qty_units.tolist() == [1_000, 1_000, 1_000, 3_000]


def _three_book_market() -> MarketData:
    b, a = ladder(40_000, 40_001, 100_000)
    rows = [(1_000, 500, b, a), (12_000, 11_500, b, a), (24_000, 23_500, b, a), (40_000, 39_000, b, a)]
    return MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),))


REVERSE_SESSION = SessionPlan(
    start_ns=0,
    entry_cutoff_ns=30_000 * US,
    flatten_start_ns=35_000 * US,
    exec_cutoff_ns=50_000 * US,
    end_ns=70_000 * US,
)


def test_reverse_flips_the_position_on_an_opposite_trigger() -> None:
    params = dataclasses.replace(BUY_ONCE, size_units=(1_000,), reverse=True)
    signal = np.array([1.0, -1.0, 0.0, 0.0])
    res = run_replay(_three_book_market(), signal, params, scenario(), REVERSE_SESSION, RiskLimits())
    # long at 1 ms; the opposite trigger at 12 ms first closes the long, its acknowledgement
    # opens the short, and the forced flatten at 35 ms buys it back
    assert res.orders.side.tolist() == [1, -1, -1, 1]
    assert res.orders.qty_units.tolist() == [1_000, 1_000, 1_000, 1_000]
    assert res.orders.reason.tolist() == [1, 2, 1, 3]
    assert res.orders.submit_ns.tolist()[1] == 12_000 * US
    held = run_replay(
        _three_book_market(),
        signal,
        dataclasses.replace(params, reverse=False),
        scenario(),
        REVERSE_SESSION,
        RiskLimits(),
    )
    assert held.orders.side.tolist() == [1, -1]  # without reverse the long is held to the flatten


def test_reverse_never_opens_a_short_when_shorts_are_disallowed() -> None:
    params = dataclasses.replace(BUY_ONCE, size_units=(1_000,), reverse=True, allow_short=False)
    signal = np.array([1.0, -1.0, 0.0, 0.0])
    res = run_replay(_three_book_market(), signal, params, scenario(), REVERSE_SESSION, RiskLimits())
    assert res.orders.side.tolist() == [1, -1]
    assert res.orders.reason.tolist()[-1] == 3


# ---------------------------------------------------------------------------
# Frozen ledgers: configurations that only use options predating the duration cap, the early
# markout check and the depth haircut must replay bit-for-bit as they did before those options
# existed. The digests were captured from both engines at commit dcdee90; a change here means
# existing research results (registered rounds, candidates) would no longer reproduce.
# ---------------------------------------------------------------------------


def ledger_digest(res: ReplayResult) -> str:
    """Digest of every fill and order column (dtype, length, raw bytes), the settlement (exact
    float reprs) and the decision count."""
    h = hashlib.sha256()
    for table in (res.fills, res.orders):
        for f in dataclasses.fields(table):
            arr = np.ascontiguousarray(getattr(table, f.name))
            h.update(f"{f.name}:{arr.dtype.str}:{arr.shape[0]};".encode())
            h.update(arr.tobytes())
    h.update(repr(dataclasses.astuple(res.settlement)).encode())
    h.update(f"decisions={res.decisions}".encode())
    return h.hexdigest()[:32]


GOLDEN_CASES: dict[str, tuple[int, ThresholdParams, int, TimingMode, bool, int]] = {
    # name: (market seed, params, latency ms, timing, separate exit signal, position cap USD)
    "single_venue": (
        101,
        ThresholdParams(
            direction=(1, 0),
            size_units=(1_000, 0),
            enter=0.8,
            exit=0.0,
            max_hold_ns=50 * MS,
            entry_slip_ticks=1,
            exit_slip_ticks=5,
        ),
        5,
        TimingMode.EXCHANGE_CLOCK,
        False,
        1_000_000,
    ),
    "hedged_stop_cooldown": (
        102,
        ThresholdParams(
            direction=(1, -1),
            size_units=(2_000, 2_000),
            enter=0.6,
            exit=-0.2,
            min_hold_ns=2 * MS,
            max_hold_ns=120 * MS,
            cooldown_ns=10 * MS,
            stop_bps=2.0,
            entry_slip_ticks=2,
            exit_slip_ticks=6,
        ),
        50,
        TimingMode.RECEIPT_CLOCK,
        True,
        1_000_000,
    ),
    "strength_and_depth_sizing": (
        103,
        ThresholdParams(
            direction=(0, 1),
            size_units=(0, 3_000),
            enter=0.5,
            exit=0.1,
            size_units_cap=3,
            depth_ticks=1,
            entry_slip_ticks=1,
            exit_slip_ticks=4,
        ),
        1,
        TimingMode.EXCHANGE_CLOCK,
        False,
        1_000_000,
    ),
    "accumulation_refresh": (
        104,
        ThresholdParams(
            direction=(1, 1),
            size_units=(500, 700),
            enter=0.4,
            exit=0.0,
            max_hold_ns=30 * MS,
            max_clips=3,
            clip_spacing_ns=5 * MS,
            refresh_hold=True,
            entry_slip_ticks=1,
            exit_slip_ticks=8,
        ),
        5,
        TimingMode.EXCHANGE_CLOCK,
        True,
        1_000_000,
    ),
    "reverse": (
        105,
        ThresholdParams(
            direction=(-1, 1),
            size_units=(1_500, 1_500),
            enter=0.7,
            exit=-0.5,
            min_hold_ns=1 * MS,
            max_hold_ns=200 * MS,
            reverse=True,
            entry_slip_ticks=2,
            exit_slip_ticks=5,
        ),
        5,
        TimingMode.RECEIPT_CLOCK,
        False,
        1_000_000,
    ),
    "long_only_capped": (
        106,
        ThresholdParams(
            direction=(1, 0),
            size_units=(8_000, 0),
            enter=0.5,
            exit=-0.3,
            allow_short=False,
            max_hold_ns=80 * MS,
            cooldown_ns=20 * MS,
            stop_bps=1.0,
            entry_slip_ticks=3,
            exit_slip_ticks=3,
        ),
        50,
        TimingMode.EXCHANGE_CLOCK,
        True,
        5_000,
    ),
}
GOLDEN_DIGESTS: dict[str, str] = {
    "accumulation_refresh": "40c7e411fe52c0c9f365420aac23f31a",
    "hedged_stop_cooldown": "c214cb41bb4b38d800b1afbbc0ea3924",
    "long_only_capped": "b8521bf26d104ab9148421790cccdb63",
    "reverse": "494eaeed9d275c986c320e2e87979805",
    "single_venue": "0fec6f56ad06966658a4ba2d6ccb1446",
    "strength_and_depth_sizing": "272a213216254bb75cda99caa893f774",
}


def _golden_replays(name: str) -> tuple[ReplayResult, ReplayResult]:
    seed, params, latency_ms, timing, separate_exit, cap = GOLDEN_CASES[name]
    data = random_market(seed)
    stream = EventStream.build(data)
    rng = np.random.default_rng(seed + 1)
    signal = rng.normal(0.0, 1.0, size=len(stream))
    exit_signal = rng.normal(0.0, 1.0, size=len(stream)) if separate_exit else None
    first, last = int(stream.recv_ns[0]), int(stream.recv_ns[-1])
    session = SessionPlan(first + 20 * MS, last - 400 * MS, last - 300 * MS, last - 60 * MS, last)
    scen = scenario(timing, latency_ms)
    fast = run_replay(data, signal, params, scen, session, RiskLimits(cap), stream=stream, exit_signal=exit_signal)
    ref = run_reference(
        data, signal, ThresholdPolicy(params), scen, session, RiskLimits(cap), stream=stream, exit_signal=exit_signal
    )
    return fast, ref


@pytest.mark.parametrize("name", sorted(GOLDEN_CASES))
def test_synthetic_ledger_snapshots_match_both_engines(name: str) -> None:
    fast, ref = _golden_replays(name)
    assert len(fast.orders) > 0
    assert ledger_digest(fast) == GOLDEN_DIGESTS[name]
    assert ledger_digest(ref) == GOLDEN_DIGESTS[name]


# ---------------------------------------------------------------------------
# Absolute duration cap, one-shot early markout check and the execution-depth haircut.
# Every hand-worked scenario runs both engines and requires exact agreement.
# ---------------------------------------------------------------------------

LATE_SESSION = SessionPlan(
    start_ns=0, entry_cutoff_ns=25 * MS, flatten_start_ns=45 * MS, exec_cutoff_ns=55 * MS, end_ns=70 * MS
)
UP, DOWN = (40_010, 40_011), (39_990, 39_991)


def both_engines(
    data: MarketData,
    signal: np.ndarray,
    params: ThresholdParams,
    session: SessionPlan,
    scen: Scenario | None = None,
    exit_signal: np.ndarray | None = None,
) -> ReplayResult:
    """Replay with the compiled kernel, require the reference twin to agree exactly, return it."""
    scen = scen if scen is not None else scenario()
    fast = run_replay(data, signal, params, scen, session, RiskLimits(), exit_signal=exit_signal)
    ref = run_reference(data, signal, ThresholdPolicy(params), scen, session, RiskLimits(), exit_signal=exit_signal)
    assert_same_replay(fast, ref)
    return fast


def steady_market(times_us: range) -> MarketData:
    """One venue, the same deep 40,000/40,001 book received at each time (exchange time 0.5 ms earlier)."""
    b, a = ladder(40_000, 40_001, 100_000)
    return MarketData((tapes_from_rows(DEMO_INVERSE_A, [(t, t - 500, b, a) for t in times_us]),))


def moving_market(after: tuple[int, int], later: tuple[int, int] | None = None) -> MarketData:
    """One venue: 40,000/40,001 at 1 ms, ``after`` from the book received at 20 ms (also at 30 ms) and
    ``later`` (default ``after``) at 40 ms."""
    b0, a0 = ladder(40_000, 40_001, 100_000)
    b1, a1 = ladder(*after, 100_000)
    b2, a2 = ladder(*(later or after), 100_000)
    rows = [(1_000, 500, b0, a0), (20_000, 19_000, b1, a1), (30_000, 29_000, b1, a1), (40_000, 39_000, b2, a2)]
    return MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),))


def test_absolute_cap_fires_on_schedule_despite_hold_refreshes_and_clips() -> None:
    data = steady_market(range(1_000, 60_000, 2_000))  # a book every 2 ms: 1, 3, ..., 59 ms
    signal = np.ones(30)  # the trigger persists: each event refreshes the hold and may add a clip
    session = SessionPlan(
        start_ns=0, entry_cutoff_ns=100 * MS, flatten_start_ns=120 * MS, exec_cutoff_ns=150 * MS, end_ns=200 * MS
    )
    params = dataclasses.replace(
        BUY_ONCE, max_hold_ns=15 * MS, refresh_hold=True, max_clips=3, clip_spacing_ns=10 * MS, cooldown_ns=1_000 * MS
    )
    held = both_engines(data, signal, params, session)
    assert held.orders.submit_ns.tolist() == [1 * MS, 11 * MS, 21 * MS, 120 * MS]  # refreshed hold: kept to flatten
    capped = both_engines(data, signal, dataclasses.replace(params, abs_cap_ns=25 * MS), session)
    # origin = acknowledgement of the first entry fill (11 ms), unmoved by the refreshes and the clips
    # acknowledged at 21 and 31 ms: deadline 36 ms, enforced at the 37 ms book
    assert capped.orders.side.tolist() == [1, 1, 1, -1]
    assert capped.orders.qty_units.tolist() == [1_000, 1_000, 1_000, 3_000]
    assert capped.orders.submit_ns.tolist() == [1 * MS, 11 * MS, 21 * MS, 37 * MS]
    assert capped.orders.reason.tolist() == [1, 1, 1, 2]


def test_absolute_cap_counts_from_the_acknowledgement_and_is_enforced_at_the_next_decision() -> None:
    late = dataclasses.replace(SESSION, flatten_start_ns=45 * MS, exec_cutoff_ns=50 * MS)
    res = both_engines(
        one_venue_market(), np.array([1.0, 0.0, 0.0]), dataclasses.replace(BUY_ONCE, abs_cap_ns=5 * MS), late
    )
    # decided at 1 ms, filled at 6 ms, acknowledged at 11 ms (the origin). The 16 ms deadline has no
    # decision of its own (no timer events), so the exit leaves at the next one, the 30 ms book, and
    # the liquidation completes later still (acknowledged at 40 ms)
    assert res.orders.submit_ns.tolist() == [1 * MS, 30 * MS]
    assert res.orders.reason.tolist() == [1, 2]
    assert res.fills.ack_ns.tolist()[-1] == 40 * MS


def test_absolute_cap_never_fires_without_an_acknowledged_fill() -> None:
    b0, a0 = ladder(40_000, 40_001, 10_000)
    b1, a1 = ladder(40_004, 40_005, 10_000)
    rows = [(1_000, 500, b0, a0), (9_000, 4_000, b1, a1), (20_000, 19_000, b1, a1), (30_000, 29_000, b1, a1)]
    data = MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),))
    session = SessionPlan(
        start_ns=0, entry_cutoff_ns=40 * MS, flatten_start_ns=45 * MS, exec_cutoff_ns=55 * MS, end_ns=70 * MS
    )
    # the tight IOC (limit 40,001) meets the 40,005 ask at its exchange arrival and fills nothing
    missed = dataclasses.replace(BUY_ONCE, entry_slip_ticks=0, abs_cap_ns=2 * MS)
    res = both_engines(data, np.ones(4), missed, session)
    assert res.orders.filled_units.tolist() == [0]  # no origin: the empty episode is never capped (no re-entry)
    # control: max_hold does end the empty episode, and the persistent trigger re-enters at the 11 ms ack
    expired = both_engines(data, np.ones(4), dataclasses.replace(missed, max_hold_ns=5 * MS), session)
    assert expired.orders.submit_ns.tolist()[:2] == [1 * MS, 11 * MS]


@pytest.mark.parametrize("option", [{"abs_cap_ns": 20 * MS}, {"early_check_ns": 20 * MS, "early_markout_bps": 50.0}])
def test_settled_fast_path_never_skips_a_due_clock_decision(option: dict[str, float]) -> None:
    data = steady_market(range(1_000, 60_000, 1_000))  # a quiet book every 1 ms
    signal = np.zeros(59)
    signal[0] = 1.0
    session = SessionPlan(
        start_ns=0, entry_cutoff_ns=10 * MS, flatten_start_ns=80 * MS, exec_cutoff_ns=90 * MS, end_ns=100 * MS
    )
    res = both_engines(data, signal, dataclasses.replace(BUY_ONCE, **option), session)  # type: ignore[arg-type]
    # settled long from the 11 ms acknowledgement; the 31 ms book is the first decision at the deadline,
    # which the settled fast path would otherwise skip (holding to the 80 ms flatten)
    assert res.orders.submit_ns.tolist() == [1 * MS, 31 * MS]
    assert res.orders.reason.tolist() == [1, 2]


@pytest.mark.parametrize(
    ("direction", "trigger", "move", "exits"),
    [
        (1, 1.0, DOWN, True),
        (1, 1.0, UP, False),  # long: a falling mid is adverse
        (1, -1.0, UP, True),
        (1, -1.0, DOWN, False),  # short: a rising mid is adverse
        (-1, 1.0, UP, True),
        (-1, 1.0, DOWN, False),  # short through a negative venue direction
        (-1, -1.0, DOWN, True),
        (-1, -1.0, UP, False),  # long through a negative venue direction
    ],
)
def test_early_check_exits_on_adverse_markout_only(
    direction: int, trigger: float, move: tuple[int, int], exits: bool
) -> None:
    params = dataclasses.replace(BUY_ONCE, direction=(direction,), early_check_ns=15 * MS, early_markout_bps=0.0)
    res = both_engines(moving_market(move), np.array([trigger, 0.0, 0.0, 0.0]), params, LATE_SESSION)
    # entry decided at 1 ms, acknowledged at 11 ms (origin): the check is due at 26 ms and runs at the
    # 30 ms book; markout -2.6 bp exits, +2.4 bp holds to the flatten
    assert res.orders.side.tolist()[0] == (1 if direction * trigger > 0 else -1)
    assert res.orders.submit_ns.tolist() == [1 * MS, 30 * MS if exits else 45 * MS]
    assert res.orders.reason.tolist() == [1, 2 if exits else 3]


def test_early_check_runs_once_per_episode() -> None:
    market = moving_market(UP, later=(39_980, 39_981))  # confirms by 30 ms, turns adverse at 40 ms
    signal = np.array([1.0, 0.0, 0.0, 0.0])
    params = dataclasses.replace(BUY_ONCE, early_check_ns=15 * MS)
    once = both_engines(market, signal, params, LATE_SESSION)
    assert once.orders.reason.tolist() == [1, 3]  # passed at 30 ms and never re-checked: held to the flatten
    later = both_engines(market, signal, dataclasses.replace(params, early_check_ns=25 * MS), LATE_SESSION)
    assert later.orders.submit_ns.tolist() == [1 * MS, 40 * MS]  # a check due at 36 ms meets the 40 ms book
    assert later.orders.reason.tolist() == [1, 2]


def test_early_markout_uses_the_inverse_contract_average_of_acknowledged_entry_fills() -> None:
    bids = [(40_000 - k, 100_000) for k in range(DEPTH)]
    asks = [(40_001, 600)] + [(40_101 + k, 100_000) for k in range(DEPTH - 1)]
    b1, a1 = ladder(40_040, 40_041, 100_000)
    rows = [(1_000, 500, bids, asks), (20_000, 19_000, b1, a1), (30_000, 29_000, b1, a1), (40_000, 39_000, b1, a1)]
    data = MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),))
    signal = np.array([1.0, 0.0, 0.0, 0.0])
    # entry sweeps 600 USD at 40,001 and 400 USD at 40,101: pbar = 1,000 / (600/40,001 + 400/40,101),
    # visibly different from the arithmetic 40,041; the mid at the check (30 ms) is 40,040.5
    pbar = 1_000 / (600 / 40_001 + 400 / 40_101)
    markout = (40_040.5 - pbar) / pbar * 1e4
    assert abs(markout - (40_040.5 - 40_041) / 40_041 * 1e4) > 1e-3
    swept = dataclasses.replace(BUY_ONCE, entry_slip_ticks=100, early_check_ns=15 * MS)
    for threshold, exits in ((markout + 1e-6, True), (markout - 1e-6, False)):
        res = both_engines(data, signal, dataclasses.replace(swept, early_markout_bps=threshold), LATE_SESSION)
        assert res.fills.qty_units.tolist()[:2] == [600, 400]
        assert res.orders.reason.tolist() == [1, 2 if exits else 3]
    # a partially filled IOC (600 of 1,000 USD at the best level): only the acknowledged 600 USD count
    partial = dataclasses.replace(swept, entry_slip_ticks=0)
    markout = (40_040.5 - 40_001) / 40_001 * 1e4
    for threshold, exits in ((markout + 1e-6, True), (markout - 1e-6, False)):
        res = both_engines(data, signal, dataclasses.replace(partial, early_markout_bps=threshold), LATE_SESSION)
        assert res.orders.filled_units.tolist() == [600, 600]
        assert res.orders.reason.tolist() == [1, 2 if exits else 3]


def test_early_markout_is_measured_on_the_reference_venue() -> None:
    flat, up, down = ladder(40_000, 40_001, 100_000), ladder(40_020, 40_021, 100_000), ladder(*DOWN, 100_000)
    venue_a = [(1_000, 500, *flat), (20_000, 19_000, *up), (30_000, 29_000, *up), (40_000, 39_000, *up)]
    venue_b = [(2_000, 1_500, *flat), (21_000, 20_000, *down), (31_000, 30_000, *down), (41_000, 40_000, *down)]
    data = MarketData((tapes_from_rows(DEMO_INVERSE_A, venue_a), tapes_from_rows(DEMO_INVERSE_B, venue_b)))
    signal = np.zeros(8)
    signal[1] = 1.0  # the trigger arrives with Venue B's first book (2 ms)
    params = ThresholdParams(
        direction=(0, 1),
        size_units=(0, 1_000),
        enter=0.5,
        exit=-10.0,
        entry_slip_ticks=10,
        exit_slip_ticks=20,
        early_check_ns=15 * MS,
    )
    res = both_engines(data, signal, params, LATE_SESSION)
    # Venue B, the reference venue, moved against the long while Venue A moved for it: exit at the first
    # decision after the 27 ms due time (the 30 ms Venue A book)
    assert res.orders.venue.tolist() == [1, 1]
    assert res.orders.submit_ns.tolist() == [2 * MS, 30 * MS]
    assert res.orders.reason.tolist() == [1, 2]


def test_early_check_waits_for_an_observable_markout() -> None:
    b0, a0 = ladder(40_000, 40_001, 100_000)
    b1, a1 = ladder(*DOWN, 100_000)
    rows = [(1_000, 500, b0, a0), (25_000, 24_000, [], a1), (30_000, 29_000, b1, a1), (40_000, 39_000, b1, a1)]
    data = MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),))
    res = both_engines(
        data, np.array([1.0, 0.0, 0.0, 0.0]), dataclasses.replace(BUY_ONCE, early_check_ns=10 * MS), LATE_SESSION
    )
    # due at 21 ms; the 25 ms book has no bids (no mid), so the one-shot check runs at the 30 ms book
    assert res.orders.submit_ns.tolist() == [1 * MS, 30 * MS]
    assert res.orders.reason.tolist() == [1, 2]


def test_a_reverse_flip_starts_a_new_clock_after_the_close() -> None:
    data = steady_market(range(1_000, 60_000, 2_000))
    signal = np.zeros(30)
    signal[0], signal[6] = 1.0, -1.0  # long at 1 ms, opposite trigger at 13 ms
    session = SessionPlan(
        start_ns=0, entry_cutoff_ns=30 * MS, flatten_start_ns=80 * MS, exec_cutoff_ns=90 * MS, end_ns=100 * MS
    )
    params = dataclasses.replace(BUY_ONCE, reverse=True, abs_cap_ns=12 * MS)
    res = both_engines(data, signal, params, session)
    # the flip at 13 ms closes the long (acknowledged at 23 ms, sell sent then), the short is
    # acknowledged at 33 ms: the short's clock starts there (not at 13 ms), so its cap exit is at 45 ms
    assert res.orders.side.tolist() == [1, -1, -1, 1]
    assert res.orders.submit_ns.tolist() == [1 * MS, 13 * MS, 23 * MS, 45 * MS]
    assert res.orders.reason.tolist() == [1, 2, 1, 2]


def test_depth_fraction_haircuts_execution_but_not_the_observed_book() -> None:
    b, a = ladder(40_000, 40_001, 4_000)
    data = MarketData((tapes_from_rows(DEMO_INVERSE_A, [(1_000, 500, b, a), (30_000, 25_000, b, a)]),))
    signal = np.array([1.0, 0.0])
    # depth-aware sizing caps the entry at the observed (displayed) 4,000 USD best ask
    params = dataclasses.replace(BUY_ONCE, size_units=(5_000,), depth_ticks=0, entry_slip_ticks=0)
    full = both_engines(data, signal, params, SESSION)
    assert full.orders.qty_units.tolist()[0] == 4_000 and full.orders.filled_units.tolist() == [4_000, 4_000]
    half = both_engines(data, signal, params, SESSION, dataclasses.replace(scenario(), depth_fraction=0.5))
    assert half.orders.qty_units.tolist() == [4_000, 2_000]  # sized on the displayed book, ...
    assert half.orders.filled_units.tolist() == [2_000, 2_000]  # ... executed against half of each level


def _clock_signal(stream: EventStream) -> np.ndarray:
    """A signal fixed by each event's receipt time and venue alone, so two markets that agree up to
    some time also agree on the signal up to it."""
    x = stream.recv_ns.astype(np.float64) * 1e-6 * 0.731 + 1.3 * stream.venue
    return 1.7 * np.sin(x) * np.cos(0.37 * x)


def _future_altered(data: MarketData, cut: int, *, drop: bool) -> MarketData:
    """``data`` with every book that is later than ``cut`` on both clocks moved 7 ticks up and thinned
    to a third (or, with ``drop``, removed)."""
    venues = []
    for tapes in data.venues:
        book = tapes.book
        late = (book.recv_ns > cut) & (book.exch_ns > cut)
        cols = {f.name: np.array(getattr(book, f.name)) for f in dataclasses.fields(book)}
        if drop:
            cols = {k: np.ascontiguousarray(v[~late]) for k, v in cols.items()}
        else:
            shift = np.where(late, 7, 0).astype(np.int32)[:, None]
            for side in ("bid", "ask"):
                px, sz = cols[f"{side}_px"], cols[f"{side}_sz"]
                cols[f"{side}_px"] = np.where(px > 0, px + shift, 0).astype(np.int32)
                cols[f"{side}_sz"] = np.where(late[:, None], sz // 3, sz).astype(np.int32)
        venues.append(dataclasses.replace(tapes, book=type(book)(**cols)))
    return MarketData(tuple(venues))


def test_clock_options_and_haircut_never_use_later_market_data() -> None:
    data = random_market(13)
    stream = EventStream.build(data)
    first, last = int(stream.recv_ns[0]), int(stream.recv_ns[-1])
    session = SessionPlan(first + 20 * MS, last - 400 * MS, last - 300 * MS, last - 60 * MS, last)
    cut = int(stream.recv_ns[len(stream) // 2])
    # no signal exit: the clock options end the episodes (a refreshed max_hold would hold them on)
    params = ThresholdParams(
        direction=(1, -1),
        size_units=(1_000, 1_000),
        enter=0.8,
        exit=-1e9,
        max_hold_ns=200 * MS,
        refresh_hold=True,
        max_clips=2,
        clip_spacing_ns=5 * MS,
        entry_slip_ticks=2,
        exit_slip_ticks=5,
        abs_cap_ns=30 * MS,
        early_check_ns=6 * MS,
        early_markout_bps=0.5,
    )
    scen = dataclasses.replace(scenario(), depth_fraction=0.6)

    def early_orders(market: MarketData, p: ThresholdParams) -> list[tuple[int, ...]]:
        ev = EventStream.build(market)
        res = run_replay(market, _clock_signal(ev), p, scen, session, RiskLimits(), stream=ev)
        o = res.orders
        rows = zip(
            o.venue.tolist(),
            o.side.tolist(),
            o.qty_units.tolist(),
            o.limit_ticks.tolist(),
            o.submit_ns.tolist(),
            o.reason.tolist(),
            strict=True,
        )
        return [r for r in rows if r[4] < cut]

    reference = early_orders(data, params)
    assert len(reference) >= 10
    assert early_orders(_future_altered(data, cut, drop=False), params) == reference
    assert early_orders(_future_altered(data, cut, drop=True), params) == reference
    plain = dataclasses.replace(params, abs_cap_ns=0, early_check_ns=0, early_markout_bps=0.0)
    assert early_orders(data, plain) != reference  # the options did act before the cut


def test_reference_policies_see_each_fill_once_at_its_acknowledgement() -> None:
    data = random_market(31)
    stream = EventStream.build(data)
    signal = np.random.default_rng(32).normal(size=len(stream))
    first, last = int(stream.recv_ns[0]), int(stream.recv_ns[-1])
    session = SessionPlan(first + 20 * MS, last - 400 * MS, last - 300 * MS, last - 60 * MS, last)
    params = ThresholdParams(direction=(1, -1), size_units=(2_000, 2_000), enter=0.7, exit=0.0, max_hold_ns=60 * MS)
    seen: list[tuple[int, FillReport]] = []

    class Recording(ThresholdPolicy):
        def act(self, view: ControllerView) -> PolicyAction:
            seen.extend((view.t, f) for f in view.acked_fills)
            return super().act(view)

    res = run_reference(data, signal, Recording(params), scenario(), session, RiskLimits(), stream=stream)
    assert len(seen) == len(res.fills) > 0
    assert all(t == f.ack_ns for t, f in seen)  # delivered at the acknowledgement, never earlier
    f = res.fills
    expected = sorted(
        zip(
            f.venue.tolist(),
            f.side.tolist(),
            f.px_ticks.tolist(),
            f.qty_units.tolist(),
            f.submit_ns.tolist(),
            f.ack_ns.tolist(),
            strict=True,
        )
    )
    assert sorted((r.venue, r.side, r.px_ticks, r.qty_units, r.submit_ns, r.ack_ns) for _, r in seen) == expected


def test_early_check_prices_only_the_episodes_own_entry_orders() -> None:
    flat, down = ladder(40_000, 40_001, 100_000), ladder(*DOWN, 100_000)
    times = [1, 3, 5, 7, 9, *range(13, 40, 2)]
    rows = [(ms * 1_000, ms * 1_000 - 500, *(flat if ms < 10 else down)) for ms in times]
    data = MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),))
    signal = np.zeros(len(times))
    signal[[0, 2]] = 1.0  # entry triggers at 1 ms and 5 ms
    exit_signal = np.zeros(len(times))
    params = dataclasses.replace(BUY_ONCE, early_check_ns=10 * MS)
    session = SessionPlan(
        start_ns=0, entry_cutoff_ns=30 * MS, flatten_start_ns=45 * MS, exec_cutoff_ns=55 * MS, end_ns=70 * MS
    )

    def replay(xsig: np.ndarray) -> ReplayResult:
        fast = run_replay(data, signal, params, scenario(), session, RiskLimits(), exit_signal=xsig)
        ref = run_reference(data, signal, ThresholdPolicy(params), scenario(), session, RiskLimits(), exit_signal=xsig)
        assert_same_replay(fast, ref)
        return fast

    # one episode: its entry (1 ms, acknowledged 11 ms) is priced, the 21 ms check sees -2.6 bp and exits
    own = replay(exit_signal)
    assert own.orders.submit_ns.tolist() == [1 * MS, 21 * MS] and own.orders.reason.tolist() == [1, 2]
    # the first episode ends at 3 ms (exit signal) before its entry is acknowledged; a second one starts
    # at 5 ms and adopts that position at the 11 ms acknowledgement (its clock starts there) but has no
    # entry order of its own to price, so the check never becomes observable: held to the flatten
    exit_signal[1] = -20.0
    adopted = replay(exit_signal)
    assert adopted.orders.submit_ns.tolist() == [1 * MS, 45 * MS] and adopted.orders.reason.tolist() == [1, 3]


# ---------------------------------------------------------------------------
# Overflow bounds on policy sizes and clocks
# ---------------------------------------------------------------------------

S = 1_000_000_000


@pytest.mark.parametrize(
    "bad",
    [
        {
            "size_units": (2**61,),
            "size_units_cap": 8,
        },  # 2**64 wraps in int64: compiled mode could flip the target's sign
        {"size_units": (2**62 + 1,)},  # the size alone exceeds the bound
        {"size_units": (2**31,), "size_units_cap": 2**16, "max_clips": 2**16},  # 2**63 only once all three multiply
        {"direction": (1, 0), "size_units": (0, 3), "size_units_cap": 2**61},  # per venue, a direction-0 venue included
        {"min_hold_ns": 2**62 + 1},
        {"max_hold_ns": 2**62 + 1},
        {"cooldown_ns": 2**62 + 1},
        {"clip_spacing_ns": 2**62 + 1},
        {"abs_cap_ns": 2**62 + 1},
        {"early_check_ns": 2**62 + 1},
    ],
)
def test_overflowing_sizes_and_clocks_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(InvalidConfigError) as info:
        dataclasses.replace(BUY_ONCE, **bad)  # type: ignore[arg-type]
    assert info.value.context  # names the offending field (and venue)


def test_size_and_clock_bounds_are_inclusive_and_admit_the_registered_incumbents() -> None:
    # R7-F (10 s hold, 10 exit slip ticks) and R7-L50 (12 s hold, 20 ticks): USD 1M, one unit, one clip
    for hold_s, slip in ((10, 10), (12, 20)):
        params = ThresholdParams(
            direction=(1, 0),
            size_units=(1_000_000, 0),
            enter=1.0,
            exit=-1e9,
            max_hold_ns=hold_s * S,
            exit_slip_ticks=slip,
            refresh_hold=True,
        )
        assert (params.max_hold_ns, params.size_units) == (hold_s * S, (1_000_000, 0))
    at_bound = dataclasses.replace(
        BUY_ONCE,
        size_units=(2**60,),
        size_units_cap=2,
        max_clips=2,
        min_hold_ns=2**62,
        max_hold_ns=2**62,
        cooldown_ns=2**62,
        clip_spacing_ns=2**62,
        abs_cap_ns=2**62,
        early_check_ns=2**62,
    )
    assert at_bound.size_units[0] * at_bound.size_units_cap * at_bound.max_clips == 2**62
    # a venue without size never binds: the bound is on each venue's own product (2**20 * 2**21 * 2**21 == 2**62)
    ThresholdParams(direction=(1, 0), size_units=(2**20, 0), enter=1.0, exit=0.0, size_units_cap=2**21, max_clips=2**21)
    with pytest.raises(InvalidConfigError) as info:  # the direction-0 venue reaches 2**63 and is rejected
        ThresholdParams(
            direction=(1, 0), size_units=(2**20, 2**21), enter=1.0, exit=0.0, size_units_cap=2**21, max_clips=2**21
        )
    assert info.value.context["venue"] == 1


def test_threshold_params_identity_fields_are_unchanged() -> None:
    """``GridExperiment.config_id`` hashes ``dataclasses.asdict(params)``: the field set is part of every registered
    experiment hash, so validation must never add or rename one."""
    assert [f.name for f in dataclasses.fields(ThresholdParams)] == [
        "direction",
        "size_units",
        "enter",
        "exit",
        "allow_short",
        "min_hold_ns",
        "max_hold_ns",
        "cooldown_ns",
        "stop_bps",
        "entry_slip_ticks",
        "exit_slip_ticks",
        "size_units_cap",
        "depth_ticks",
        "max_clips",
        "clip_spacing_ns",
        "refresh_hold",
        "reverse",
        "abs_cap_ns",
        "early_check_ns",
        "early_markout_bps",
    ]


@pytest.mark.parametrize(
    "bad", [{"entry_slip_ticks": 2**62 + 1}, {"exit_slip_ticks": 2**62 + 1}, {"depth_ticks": 2**62 + 1}]
)
def test_overflowing_tick_offsets_are_rejected(bad: dict[str, object]) -> None:
    """The kernels add these to int64 tick prices (``best +/- slip``, ``best + depth``): a Python int past the
    bound would wrap in compiled mode while the reference twin computes it exactly."""
    with pytest.raises(InvalidConfigError) as info:
        dataclasses.replace(BUY_ONCE, **bad)  # type: ignore[arg-type]
    assert info.value.context["field"] == next(iter(bad))


def test_tick_offset_bound_is_inclusive() -> None:
    at_bound = dataclasses.replace(BUY_ONCE, entry_slip_ticks=2**62, exit_slip_ticks=2**62, depth_ticks=2**62)
    assert (at_bound.entry_slip_ticks, at_bound.exit_slip_ticks, at_bound.depth_ticks) == (2**62,) * 3


def test_a_re_entry_that_adopts_a_pending_position_starts_its_clock_at_the_re_trigger() -> None:
    data = steady_market(range(1_000, 100_000, 1_000))  # a book every 1 ms
    signal = np.ones(99)  # the trigger never lets up (cooldown 0: a new episode starts right after each exit)
    session = SessionPlan(
        start_ns=0, entry_cutoff_ns=70 * MS, flatten_start_ns=120 * MS, exec_cutoff_ns=150 * MS, end_ns=200 * MS
    )
    res = both_engines(data, signal, dataclasses.replace(BUY_ONCE, abs_cap_ns=10 * MS), session)
    o = res.orders
    rows = list(zip(o.submit_ns.tolist(), o.side.tolist(), o.reason.tolist(), o.ack_ns.tolist(), strict=True))
    # Entry at 1 ms, acknowledged at 11 ms (origin): capped at 21 ms (exit acknowledged at 31 ms). The episode
    # re-triggered at 22 ms adopts the long still awaiting liquidation, so its clock starts at 22 ms, before
    # it has sent anything; the venue is free again at 31 ms (its entry, acknowledged 41 ms) and its cap ends
    # it at 32 ms. A third episode (33 ms) adopts that 41 ms fill (origin 41 ms) and is capped at 51 ms.
    assert rows[:6] == [
        (1 * MS, 1, 1, 11 * MS),
        (21 * MS, -1, 2, 31 * MS),
        (31 * MS, 1, 1, 41 * MS),
        (51 * MS, -1, 2, 61 * MS),
        (61 * MS, 1, 1, 71 * MS),
        (81 * MS, -1, 2, 91 * MS),
    ]


@pytest.mark.parametrize("option", [{"abs_cap_ns": 5 * MS}, {"early_check_ns": 5 * MS, "early_markout_bps": 0.0}])
def test_clock_exits_are_not_deferred_by_min_hold(option: dict[str, float]) -> None:
    flat, down = ladder(40_000, 40_001, 100_000), ladder(*DOWN, 100_000)
    rows = [(1_000, 500, *flat)] + [(ms * 1_000, ms * 1_000 - 500, *down) for ms in range(12, 41)]
    data = MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),))
    signal = np.zeros(len(rows))
    signal[0] = 1.0
    exit_signal = np.full(len(rows), -20.0)  # the signal exit is wanted from the first event on
    held = both_engines(
        data, signal, dataclasses.replace(BUY_ONCE, min_hold_ns=60 * MS), LATE_SESSION, exit_signal=exit_signal
    )
    assert held.orders.reason.tolist() == [1, 3]  # the signal exit waits for min_hold: held to the 45 ms flatten
    params = dataclasses.replace(BUY_ONCE, min_hold_ns=60 * MS, **option)  # type: ignore[arg-type]
    res = both_engines(data, signal, params, LATE_SESSION, exit_signal=exit_signal)
    # origin 11 ms, deadline 16 ms: like the stop, the cap and the early check (-2.6 bp) ignore min_hold
    assert res.orders.submit_ns.tolist() == [1 * MS, 16 * MS]
    assert res.orders.reason.tolist() == [1, 2]


def test_early_threshold_boundary_holds_on_equality() -> None:
    pbar = 1_000 / (1_000 / 40_001)  # one 1,000 USD fill at 40,001, accumulated as the engines do
    markout = (39_990.5 - pbar) / pbar * 1e4  # the 30 ms mid of moving_market(DOWN): about -2.62 bp
    params = dataclasses.replace(BUY_ONCE, early_check_ns=15 * MS)
    signal = np.array([1.0, 0.0, 0.0, 0.0])
    equal = both_engines(
        moving_market(DOWN), signal, dataclasses.replace(params, early_markout_bps=markout), LATE_SESSION
    )
    assert equal.orders.reason.tolist() == [1, 3]  # a markout equal to the threshold confirms: held to the flatten
    above = both_engines(
        moving_market(DOWN),
        signal,
        dataclasses.replace(params, early_markout_bps=float(np.nextafter(markout, np.inf))),
        LATE_SESSION,
    )
    assert above.orders.reason.tolist() == [1, 2] and above.orders.submit_ns.tolist()[1] == 30 * MS


def test_early_markout_ignores_the_close_fills_of_a_reverse_flip() -> None:
    flat, up = ladder(40_000, 40_001, 100_000), ladder(40_020, 40_021, 100_000)
    rows = [(t, t - 500, *(flat if t < 20_000 else up)) for t in range(1_000, 60_000, 2_000)]
    data = MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),))
    signal = np.zeros(len(rows))
    signal[0], signal[6] = 1.0, -1.0  # long at 1 ms, opposite trigger at 13 ms
    session = SessionPlan(
        start_ns=0, entry_cutoff_ns=30 * MS, flatten_start_ns=80 * MS, exec_cutoff_ns=90 * MS, end_ns=100 * MS
    )
    params = dataclasses.replace(BUY_ONCE, reverse=True, early_check_ns=10 * MS)
    # The flip's close (sent at 13 ms, the short episode's start, 1,000 USD at 40,000) is an exit order; the
    # short's own entry follows its acknowledgement (sent 23 ms, 1,000 USD at 40,020, acknowledged 33 ms =
    # origin). At the check (43 ms book, mid 40,020.5) the markout is -0.125 bp; had the close fill been
    # priced too (pbar about 40,010) it would read -2.6 bp.
    for threshold, exits in ((-1.0, False), (-0.1, True)):
        res = both_engines(data, signal, dataclasses.replace(params, early_markout_bps=threshold), session)
        assert res.fills.px_ticks.tolist()[:3] == [40_001, 40_000, 40_020]
        assert res.orders.submit_ns.tolist()[:3] == [1 * MS, 13 * MS, 23 * MS]
        assert res.orders.reason.tolist() == [1, 2, 1, 2 if exits else 3]
        assert res.orders.submit_ns.tolist()[3] == (43 * MS if exits else 80 * MS)
