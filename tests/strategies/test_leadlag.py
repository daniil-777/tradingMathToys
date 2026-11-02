"""Robust lead-lag signal recipes: hand values, gating and causality."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from financeai.domains.market.engine import PreparedMarket
from financeai.domains.market.instruments import DEMO_INVERSE_A, DEMO_INVERSE_B
from financeai.domains.market.stream import MarketData
from financeai.strategies.leadlag import (
    build_signal,
    depth_weighted_mid,
    lead_gap_signal,
    sweep_signal,
    window_change_bps,
)

from ..market.synthetic import ladder, random_market, tapes_from_rows

MS = 1_000_000


def test_depth_weighted_mid_averages_side_vwaps_over_the_requested_depth() -> None:
    rows = [(1_000, 500, [(40_000, 30_000), (39_998, 40_000)], [(40_002, 10_000), (40_004, 90_000)])]
    tapes = tapes_from_rows(DEMO_INVERSE_A, rows)
    # bid VWAP of first 50k: (20000*30k + 19999*20k)/50k = 19999.6 ; ask: (20001*10k + 20002*40k)/50k = 20001.8
    value = depth_weighted_mid(tapes, 50_000)
    assert value[0] == pytest.approx((19_999.6 + 20_001.8) / 2)
    assert np.isnan(depth_weighted_mid(tapes, 1_000_000)[0])  # not enough displayed depth


def test_window_change_uses_the_value_as_of_t_minus_window() -> None:
    recv = np.array([0, 10, 20, 30], np.int64) * MS
    values = np.array([100.0, 101.0, 102.0, 103.0])
    out = window_change_bps(values, recv, 15 * MS)
    assert np.isnan(out[0])
    # at t=20ms the value as of 5ms is 100 -> +200 bps ; at 30ms as of 15ms is 101
    assert out[2] == pytest.approx((102 / 100 - 1) * 1e4)
    assert out[3] == pytest.approx((103 / 101 - 1) * 1e4)


def two_venue_market() -> MarketData:
    b, a = ladder(40_000, 40_001, 60_000)
    bw, aw = ladder(40_000, 40_004, 60_000)  # wide follower spread
    up_b, up_a = ladder(40_020, 40_021, 60_000)
    venue_b = tapes_from_rows(DEMO_INVERSE_B, [(1_000, 900, b, a), (100_000, 99_000, up_b, up_a)])
    venue_a = tapes_from_rows(DEMO_INVERSE_A, [(2_000, 1_500, b, a), (50_000, 49_000, bw, aw)])
    return MarketData((venue_a, venue_b))


def test_lead_gap_is_gated_on_a_one_tick_follower_spread() -> None:
    prep = PreparedMarket.build(two_venue_market())
    sig = lead_gap_signal(prep, leader=1, follower=0, window_ns=90 * MS, depth_units=50_000, gate_ticks=1)
    # last event: venue_b jumped +10 USD but venue_a shows a 4-tick spread -> gated to zero
    assert sig[-1] == 0.0
    ungated = lead_gap_signal(prep, leader=1, follower=0, window_ns=90 * MS, depth_units=50_000, gate_ticks=0)
    assert ungated[-1] > 0


def test_sweep_signal_marks_large_aggressor_batches() -> None:
    data = two_venue_market()
    trades = dataclasses.replace(
        data.venues[0].trades,
        recv_ns=np.array([3_000, 3_000, 60_000], np.int64) * 1_000,
        exch_ns=np.array([2_900, 2_900, 59_000], np.int64) * 1_000,
        px=np.array([40_001, 40_002, 40_000], np.int32),
        sz=np.array([80_000, 40_000, 200_000], np.int32),
        side_label=np.array([1, 1, -1], np.int8),
    )
    venues = (dataclasses.replace(data.venues[0], trades=trades), data.venues[1])
    prep = PreparedMarket.build(MarketData(venues))
    sig = sweep_signal(prep, source=0, min_units=100_000)
    recv = prep.stream.recv_ns
    assert set(sig[recv == 3_000_000].tolist()) == {1.0}  # 120k buy batch
    assert set(sig[recv == 60_000_000].tolist()) == {-1.0}  # 200k sell batch
    assert np.all(sig[(recv != 3_000_000) & (recv != 60_000_000)] == 0.0)


def test_leader_delay_placebo_uses_the_leader_as_of_t_minus_delay() -> None:
    prep = PreparedMarket.build(random_market(17))
    recv = prep.stream.recv_ns
    delay = 30 * MS
    base = lead_gap_signal(prep, leader=1, follower=0, window_ns=20 * MS, depth_units=1_000, gate_ticks=0)
    lagged = lead_gap_signal(
        prep, leader=1, follower=0, window_ns=20 * MS, depth_units=1_000, gate_ticks=0, leader_delay_ns=delay
    )
    assert not np.array_equal(base, lagged)
    # zero delay is the identity
    same = lead_gap_signal(
        prep, leader=1, follower=0, window_ns=20 * MS, depth_units=1_000, gate_ticks=0, leader_delay_ns=0
    )
    np.testing.assert_array_equal(base, same)
    # before any leader history exists `delay` ago, the placebo has no signal
    assert np.all(lagged[recv < recv[0] + delay] == 0.0)


@pytest.mark.parametrize(
    "spec",
    [
        {
            "kind": "lead_gap",
            "leader": "venue_b/ALPHA-PERP",
            "follower": "venue_a/ALPHA-USD",
            "window_ms": 50,
            "depth_units": 20_000,
            "gate_ticks": 2,
        },
        {
            "kind": "ewma_dev",
            "leader": "venue_b/ALPHA-PERP",
            "follower": "venue_a/ALPHA-USD",
            "tau_ms": 500,
            "depth_units": 20_000,
            "gate_ticks": 2,
        },
        {"kind": "sweep", "source": "venue_a/ALPHA-USD", "min_units": 1_000},
    ],
)
def test_signals_never_use_later_events(spec: dict[str, object]) -> None:
    data = random_market(4)
    prep = PreparedMarket.build(data)
    base = build_signal(prep, spec)
    k = len(prep.stream) // 2
    cut = int(prep.stream.recv_ns[k])
    mutated = []
    for tapes in data.venues:
        book = tapes.book
        late = book.recv_ns > cut
        shift = np.where(late, 9, 0).astype(np.int32)[:, None]
        new_book = dataclasses.replace(
            book,
            recv_ns=book.recv_ns.copy(),
            exch_ns=book.exch_ns.copy(),
            kind=book.kind.copy(),
            n_bid=book.n_bid.copy(),
            n_ask=book.n_ask.copy(),
            bid_px=(book.bid_px + shift).astype(np.int32),
            bid_sz=book.bid_sz.copy(),
            ask_px=(book.ask_px + shift).astype(np.int32),
            ask_sz=book.ask_sz.copy(),
        )
        mutated.append(dataclasses.replace(tapes, book=new_book))
    other = build_signal(PreparedMarket.build(MarketData(tuple(mutated))), spec)
    early = prep.stream.recv_ns <= cut
    np.testing.assert_array_equal(base[early], other[early])


def test_any_signal_takes_the_strongest_normalised_component() -> None:
    prep = PreparedMarket.build(random_market(33))
    a = {
        "kind": "lead_gap",
        "leader": "venue_b/ALPHA-PERP",
        "follower": "venue_a/ALPHA-USD",
        "window_ms": 20,
        "depth_units": 1_000,
        "gate_ticks": 0,
    }
    b = {
        "kind": "lead_gap",
        "leader": "venue_b/ALPHA-PERP",
        "follower": "venue_a/ALPHA-USD",
        "window_ms": 50,
        "depth_units": 1_000,
        "gate_ticks": 0,
    }
    sa, sb = build_signal(prep, a) / 2.0, build_signal(prep, b) / 4.0
    combo = build_signal(prep, {"kind": "any", "components": [{"spec": a, "scale": 2.0}, {"spec": b, "scale": 4.0}]})
    expected = np.where(np.abs(sa) >= np.abs(sb), sa, sb)
    np.testing.assert_allclose(combo, expected)


def test_all_signal_requires_every_component_in_the_same_direction() -> None:
    prep = PreparedMarket.build(random_market(34))
    a = {
        "kind": "lead_gap",
        "leader": "venue_b/ALPHA-PERP",
        "follower": "venue_a/ALPHA-USD",
        "window_ms": 20,
        "depth_units": 1_000,
        "gate_ticks": 0,
    }
    b = {
        "kind": "lead_gap",
        "leader": "venue_b/ALPHA-PERP",
        "follower": "venue_a/ALPHA-USD",
        "window_ms": 50,
        "depth_units": 1_000,
        "gate_ticks": 0,
    }
    sa, sb = build_signal(prep, a) / 2.0, build_signal(prep, b) / 3.0
    combo = build_signal(prep, {"kind": "all", "components": [{"spec": a, "scale": 2.0}, {"spec": b, "scale": 3.0}]})
    same = np.sign(sa) == np.sign(sb)
    expected = np.where(same, np.sign(sa) * np.minimum(np.abs(sa), np.abs(sb)), 0.0)
    np.testing.assert_allclose(combo, expected)


def test_normalised_signal_divides_by_a_causal_ewma_sd() -> None:
    prep = PreparedMarket.build(random_market(35))
    a = {
        "kind": "lead_gap",
        "leader": "venue_b/ALPHA-PERP",
        "follower": "venue_a/ALPHA-USD",
        "window_ms": 20,
        "depth_units": 1_000,
        "gate_ticks": 0,
    }
    raw = build_signal(prep, a)
    z = build_signal(prep, {**a, "normalize_tau_ms": 50})
    ok = (z != 0) & (raw != 0)
    assert ok.any()
    assert np.all(np.sign(z[ok]) == np.sign(raw[ok]))
    assert np.all(np.isfinite(z))


def test_freshness_gate_blocks_stale_books() -> None:
    prep = PreparedMarket.build(random_market(36))
    base = {"kind": "sweep", "source": "venue_a/ALPHA-USD", "min_units": 1}
    sig = np.ones(len(prep.stream))
    from financeai.strategies.regime import freshness_mask

    mask = freshness_mask(prep, venues=(0, 1), max_age_ns=1)
    assert mask.dtype == bool and not mask.all()  # 1 ns is always stale except at the update itself
    loose = freshness_mask(prep, venues=(0, 1), max_age_ns=10**12)
    first_both = max(int(np.flatnonzero(prep.stream.venue == v)[0]) for v in (0, 1))
    assert loose[first_both:].all()
    gated = build_signal(prep, {**base, "fresh": {"max_age_ms": 1_000_000}})
    np.testing.assert_array_equal(gated[first_both:], build_signal(prep, base)[first_both:])
    _ = sig


def test_move_signal_is_the_venue_trailing_wmid_return() -> None:
    from financeai.strategies.leadlag import depth_weighted_mid, event_values

    prep = PreparedMarket.build(random_market(37))
    spec = {"kind": "move", "venue": "venue_b/ALPHA-PERP", "window_ms": 30, "depth_units": 1_000, "gate_ticks": 0}
    got = build_signal(prep, spec)
    x = event_values(prep.stream, 1, depth_weighted_mid(prep.data.venues[1], 1_000))
    expected = window_change_bps(x, prep.stream.recv_ns, 30 * MS)
    np.testing.assert_allclose(got, np.where(np.isfinite(expected), expected, 0.0))


# ---------------------------------------------------------------------------------------------
# Increment 2 signal kinds: two-sided quote confirmation, multi-level OFI, composite branch index
# ---------------------------------------------------------------------------------------------
def _book(bid: int, ask: int, size: int = 60_000) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    return ladder(bid, ask, size)


def quote_market() -> MarketData:
    """Venue B quotes: flat, both sides up one tick, only the ask up, both sides down two ticks."""
    rows = [
        (1_000, 900, *_book(40_000, 40_001)),
        (60_000, 59_000, *_book(40_001, 40_002)),
        (120_000, 119_000, *_book(40_001, 40_003)),
        (180_000, 179_000, *_book(39_999, 40_001)),
    ]
    venue_b = tapes_from_rows(DEMO_INVERSE_B, rows)
    venue_a = tapes_from_rows(DEMO_INVERSE_A, [(500, 400, *_book(40_000, 40_001))])
    return MarketData((venue_a, venue_b))


def test_quote_confirm_requires_both_sides_to_move_in_the_same_direction() -> None:
    prep = PreparedMarket.build(quote_market())
    spec = {"kind": "quote_confirm", "venue": "venue_b/ALPHA-PERP", "window_ms": 50, "ticks": 1}
    out = build_signal(prep, spec)
    # events: venue_a@0.5ms (no venue_b book yet), venue_b@1ms (no book 50 ms earlier) -> NaN;
    # venue_b@60ms (as-of 10ms = row 0: bid +1, ask +1 -> +1); venue_b@120ms (as-of 70ms = row 1: bid 0, ask +1 -> 0);
    # venue_b@180ms (as-of 130ms = row 2: bid -2, ask -2 -> -1)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == 1.0 and out[3] == 0.0 and out[4] == -1.0
    two = build_signal(prep, {**spec, "ticks": 2})
    assert two[2] == 0.0 and two[4] == -1.0  # a one-tick rise is not a two-tick confirmation


def test_quote_confirm_rejects_bad_specs() -> None:
    prep = PreparedMarket.build(quote_market())
    from financeai.core.errors import InvalidConfigError

    with pytest.raises(InvalidConfigError):
        build_signal(prep, {"kind": "quote_confirm", "venue": "venue_b/ALPHA-PERP", "window_ms": 100, "ticks": 0})
    with pytest.raises(InvalidConfigError):
        build_signal(prep, {"kind": "quote_confirm", "venue": "venue_b/ALPHA-PERP", "window_ms": 100})


def ofi_market() -> MarketData:
    """Venue A books 50 ms apart: (0) base; (1) bid queue grows by 10k at the same price;
    (2) best ask price falls one tick (new ask queue 5k); (3) best bid price falls one tick."""
    base_b, base_a = _book(40_000, 40_002, 20_000)
    grown_b = [(40_000, 30_000), *base_b[1:]]  # same best price, bigger queue
    shifted_a = [(40_001, 5_000), *base_a[:-1]]  # a new best ask one tick lower; ladder shifts down
    lower_b = [(39_999, 7_000), *[(p - 1, q) for p, q in base_b[1:]]]  # best bid gone; ladder shifts down
    rows = [
        (1_000, 900, base_b, base_a),
        (51_000, 50_900, grown_b, base_a),
        (101_000, 100_900, grown_b, shifted_a),
        (151_000, 150_900, lower_b, shifted_a),
    ]
    venue_a = tapes_from_rows(DEMO_INVERSE_A, rows)
    venue_b = tapes_from_rows(DEMO_INVERSE_B, [(500, 400, *_book(40_000, 40_001))])
    return MarketData((venue_a, venue_b))


def test_best_level_ofi_follows_the_cont_kukanov_stoikov_definition() -> None:
    prep = PreparedMarket.build(ofi_market())
    spec = {"kind": "ofi", "venue": "venue_a/ALPHA-USD", "levels": 1, "window_ms": 60, "normalize": "none"}
    out = build_signal(prep, spec)
    # events: venue_b@0.5ms (no venue_a book yet -> NaN), venue_a@1ms (no previous snapshot -> 0 flow in window),
    # venue_a@51ms: e = +10k (bid price unchanged, queue 20k -> 30k); window 60 ms holds only this update
    # venue_a@101ms: ask price fell: e = -(qa_n) = -5k; the 51 ms update is 50 ms old -> still in window: +10k-5k
    # venue_a@151ms: bid price fell: e = -(qb_{n-1}) = -30k; window holds the 101 ms update too: -5k-30k
    assert np.isnan(out[0])
    assert out[1] == 0.0
    assert out[2] == 10_000.0
    assert out[3] == 5_000.0
    assert out[4] == -35_000.0
    sign = build_signal(prep, {**spec, "normalize": "sign"})
    assert sign[2] == 1.0 and sign[3] == 1.0 and sign[4] == -1.0
    depth = build_signal(prep, {**spec, "normalize": "depth"})
    # divided by the mean of (bid + ask queue)/2 over the snapshots received in the window: at 51 ms the window
    # (-9 ms, 51 ms] holds the 1 ms snapshot (20k) and the 51 ms snapshot (25k) -> 22.5k
    assert depth[2] == pytest.approx(10_000.0 / 22_500.0)


def test_multi_level_ofi_sums_the_per_level_imbalances() -> None:
    prep = PreparedMarket.build(ofi_market())
    one = build_signal(
        prep, {"kind": "ofi", "venue": "venue_a/ALPHA-USD", "levels": 1, "window_ms": 60, "normalize": "none"}
    )
    three = build_signal(
        prep, {"kind": "ofi", "venue": "venue_a/ALPHA-USD", "levels": 3, "window_ms": 60, "normalize": "none"}
    )
    # the 51 ms update only changed level 1, so levels 2-3 add nothing there
    assert three[2] == one[2]
    # at 101 ms the ask ladder shifted down a tick: level 1 (-5k as before); level 2 now (40_002, 20k) against the
    # old (40_003, 20k): price fell -> -qa_n = -20k; level 3 likewise -20k
    assert three[3] == pytest.approx(10_000.0 - 5_000.0 - 20_000.0 - 20_000.0)
    from financeai.core.errors import InvalidConfigError

    with pytest.raises(InvalidConfigError):
        build_signal(
            prep, {"kind": "ofi", "venue": "venue_a/ALPHA-USD", "levels": 11, "window_ms": 60, "normalize": "none"}
        )
    with pytest.raises(InvalidConfigError):
        build_signal(
            prep, {"kind": "ofi", "venue": "venue_a/ALPHA-USD", "levels": 1, "window_ms": 60, "normalize": "max"}
        )


@pytest.mark.parametrize(
    "spec",
    [
        {"kind": "quote_confirm", "venue": "venue_b/ALPHA-PERP", "window_ms": 50, "ticks": 1},
        {"kind": "ofi", "venue": "venue_a/ALPHA-USD", "levels": 3, "window_ms": 200, "normalize": "depth"},
    ],
)
def test_new_signal_kinds_never_use_later_events(spec: dict[str, object]) -> None:
    data = random_market(31)
    full = PreparedMarket.build(data)
    out = build_signal(full, spec)
    cut = full.stream.recv_ns[len(full.stream) // 2]
    book_cols = ("recv_ns", "exch_ns", "kind", "n_bid", "n_ask", "bid_px", "bid_sz", "ask_px", "ask_sz")
    trade_cols = ("recv_ns", "exch_ns", "px", "sz", "side_label")
    truncated = MarketData(
        tuple(
            dataclasses.replace(
                t,
                book=dataclasses.replace(t.book, **{k: getattr(t.book, k)[t.book.recv_ns <= cut] for k in book_cols}),
                trades=dataclasses.replace(
                    t.trades, **{k: getattr(t.trades, k)[t.trades.recv_ns <= cut] for k in trade_cols}
                ),
            )
            for t in data.venues
        )
    )
    short = build_signal(PreparedMarket.build(truncated), spec)
    n = len(short)
    assert np.array_equal(out[:n], short, equal_nan=True)


def test_composite_branch_reports_which_component_was_selected() -> None:
    from financeai.core.errors import InvalidConfigError
    from financeai.strategies.leadlag import composite_branch

    prep = PreparedMarket.build(two_venue_market())
    gap = {
        "kind": "lead_gap",
        "leader": "venue_b/ALPHA-PERP",
        "follower": "venue_a/ALPHA-USD",
        "window_ms": 90,
        "depth_units": 50_000,
        "gate_ticks": 0,
    }
    move = {"kind": "move", "venue": "venue_b/ALPHA-PERP", "window_ms": 90, "depth_units": 50_000, "gate_ticks": 0}
    spec = {"kind": "any", "components": [{"spec": gap, "scale": 1.0}, {"spec": move, "scale": 1000.0}]}
    branch = composite_branch(prep, spec)
    value = build_signal(prep, spec)
    assert branch.dtype == np.int8 and branch.shape == value.shape
    # the last event passes through the gap component (|gap| >= 1); the move component is scaled far below 1
    assert branch[-1] == 0
    assert all(branch[i] == -1 for i in range(len(branch)) if not abs(value[i]) >= 1.0)
    with pytest.raises(InvalidConfigError):
        composite_branch(prep, gap)  # not a composite
