"""Causal per-event signals: hand-computed examples, independent slow oracles on random
markets, future-mutation causality for every kind, registry validation and parity of the
pure-Python fallback with the compiled kernels."""

from __future__ import annotations

import bisect
import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.core.hashing import sha256_json
from financeai.core.jit import HAVE_NUMBA
from financeai.domains.market import signals
from financeai.domains.market.engine import PreparedMarket
from financeai.domains.market.instruments import DEMO_INVERSE_A, DEMO_INVERSE_B
from financeai.domains.market.signals import (
    TopOfBook,
    basis_deviation_bps,
    build_signal,
    canonical_spec,
    ewma_time,
    imbalance,
    lead_gap_bps,
    observed_rows,
    ofi,
    spec_id,
    top_of_book,
    trade_flow,
    trailing_return_bps,
)
from financeai.domains.market.stream import BOOK_EVENT, TRADE_EVENT, EventStream, MarketData
from financeai.domains.market.tape import (
    BOOK_CLEAR,
    BOOK_SNAPSHOT,
    DEPTH,
    BookTape,
    MarketTapes,
    TradeTape,
)

from .synthetic import US, random_market, tapes_from_rows

MS = 1_000_000
NAN = math.nan
BMX = DEMO_INVERSE_A.key
DBT = DEMO_INVERSE_B.key
Trade = tuple[int, int, int, int]
"""(recv_us, px_ticks, size_units, side_label)"""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def with_trades(tapes: MarketTapes, trades: Sequence[Trade]) -> MarketTapes:
    tape = TradeTape(
        recv_ns=np.array([t[0] * US for t in trades], np.int64),
        exch_ns=np.array([t[0] * US - 500 * US for t in trades], np.int64),
        px=np.array([t[1] for t in trades], np.int32),
        sz=np.array([t[2] for t in trades], np.int32),
        side_label=np.array([t[3] for t in trades], np.int8),
    )
    return dataclasses.replace(tapes, trades=tape)


def prepare(*venues: MarketTapes) -> PreparedMarket:
    return PreparedMarket.build(MarketData(tuple(venues)))


def rich_market(seed: int, n_rows: int = 160, n_trades: int = 90) -> MarketData:
    """``random_market`` plus unusable book rows (clears, one-sided, crossed) and trade prints.

    A third of the prints share a receipt time with a book row of their venue (cohorts), some
    repeat the previous print's time (multi-print messages) and some precede the first book.
    """
    base = random_market(seed, n_rows)
    rng = np.random.default_rng(seed + 7_919)
    venues = []
    for tapes in base.venues:
        b = tapes.book
        kind, n_bid, n_ask = b.kind.copy(), b.n_bid.copy(), b.n_ask.copy()
        bid_px, bid_sz = b.bid_px.copy(), b.bid_sz.copy()
        ask_px, ask_sz = b.ask_px.copy(), b.ask_sz.copy()
        for r in rng.choice(n_rows, size=n_rows // 12, replace=False).tolist():
            mode = int(rng.integers(0, 3))
            if mode == 0:  # recorder clear: no levels at all
                kind[r], n_bid[r], n_ask[r] = BOOK_CLEAR, 0, 0
                bid_px[r], bid_sz[r], ask_px[r], ask_sz[r] = 0, 0, 0, 0
            elif mode == 1:  # one-sided
                n_ask[r] = 0
                ask_px[r], ask_sz[r] = 0, 0
            else:  # crossed
                ask_px[r] = bid_px[r, 0] - 1 + np.arange(DEPTH)
        book = BookTape(b.recv_ns.copy(), b.exch_ns.copy(), kind, n_bid, n_ask, bid_px, bid_sz, ask_px, ask_sz)
        recv = b.recv_ns
        n_cohort = n_trades // 3
        t = np.sort(
            np.concatenate(
                [
                    rng.choice(recv, size=n_cohort),
                    rng.integers(int(recv[0]) // 2, int(recv[-1]), size=n_trades - n_cohort),
                ]
            )
        )
        for i in range(1, n_trades):
            if rng.random() < 0.15:
                t[i] = t[i - 1]
        trades = TradeTape(
            recv_ns=t.astype(np.int64),
            exch_ns=(t - rng.integers(100 * US, 5 * MS, n_trades)).astype(np.int64),
            px=(40_000 + rng.integers(-40, 40, n_trades)).astype(np.int32),
            sz=(rng.integers(1, 60, n_trades) * tapes.instrument.lot_size).astype(np.int32),
            side_label=rng.choice(np.array([-1, 1], np.int8), n_trades),
        )
        venues.append(dataclasses.replace(tapes, book=book, trades=trades))
    return MarketData(tuple(venues))


def quotes_market() -> PreparedMarket:
    venue_a = tapes_from_rows(
        DEMO_INVERSE_A,
        [
            (1_000, 900, [(40_000, 100)], [(40_002, 300)]),
            (3_000, 2_900, [(40_005, 100)], [(40_004, 100)]),  # crossed -> unusable
            (5_000, 4_900, [(40_001, 200), (40_000, 500)], [(40_002, 200), (40_003, 100)]),
        ],
    )
    venue_b = tapes_from_rows(
        DEMO_INVERSE_B,
        [
            (2_000, 1_000, [(39_990, 50)], []),  # one-sided -> unusable
            (3_000, 2_000, [(39_990, 50)], [(39_992, 150)]),
        ],
    )
    # events: e0 1ms venue_a r0 | e1 2ms venue_b r0 | e2 3ms venue_a r1 | e3 3ms venue_b r1 | e4 5ms venue_a r2
    return prepare(venue_a, venue_b)


def lead_lag_market() -> PreparedMarket:
    venue_a = tapes_from_rows(
        DEMO_INVERSE_A,
        [
            (1_000, 900, [(40_000, 100)], [(40_002, 100)]),  # mid 20,000.5
            (4_000, 3_900, [(40_020, 100)], [(40_022, 100)]),  # leader jumps to 20,010.5
        ],
    )
    venue_b = tapes_from_rows(
        DEMO_INVERSE_B,
        [
            (2_000, 1_000, [(40_000, 100)], [(40_002, 100)]),  # mid 20,000.5
            (6_000, 5_000, [(40_020, 100)], [(40_022, 100)]),  # follower catches up
        ],
    )
    # events: e0 1ms venue_a | e1 2ms venue_b | e2 4ms venue_a | e3 6ms venue_b
    return prepare(venue_a, venue_b)


JUMP_BPS = (20_010.5 / 20_000.5 - 1.0) * 1e4


def cohort_market() -> PreparedMarket:
    """``lead_lag_market`` plus a Venue A print received in the same 4 ms cohort as the leader jump."""
    venue_a, venue_b = lead_lag_market().data.venues
    # events: e0 1ms A book | e1 2ms B book | e2 4ms A book | e3 4ms A trade | e4 6ms B book
    return prepare(with_trades(venue_a, [(4_000, 40_020, 100, 1)]), venue_b)


def skew_market() -> PreparedMarket:
    """Mid and microprice disagree: Venue A keeps its quotes but its level-1 sizes flip."""
    venue_a = tapes_from_rows(
        DEMO_INVERSE_A,
        [
            (1_000, 900, [(40_000, 100)], [(40_002, 300)]),  # mid 20,000.5, microprice 20,000.25
            (4_000, 3_900, [(40_000, 300)], [(40_002, 100)]),  # mid 20,000.5, microprice 20,000.75
        ],
    )
    venue_b = tapes_from_rows(
        DEMO_INVERSE_B,
        [
            (2_000, 1_000, [(40_000, 100)], [(40_002, 100)]),  # mid = microprice = 20,000.5
            (6_000, 5_000, [(40_000, 100)], [(40_002, 100)]),
        ],
    )
    # events: e0 1ms venue_a | e1 2ms venue_b | e2 4ms venue_a | e3 6ms venue_b
    return prepare(venue_a, venue_b)


MICRO_JUMP_BPS = (20_000.75 / 20_000.25 - 1.0) * 1e4


# ---------------------------------------------------------------------------
# Hand-computed examples
# ---------------------------------------------------------------------------


def test_observed_rows_track_the_last_received_book_row_per_venue() -> None:
    rows = observed_rows(quotes_market())
    assert rows.dtype == np.int64 and not rows.flags.writeable
    # e2 and e3 share a receipt time: at e2 the cohort's venue_b row is not yet visible.
    assert rows.tolist() == [[0, 0, 1, 1, 2], [-1, 0, 0, 1, 1]]


def test_top_of_book_is_nan_for_missing_or_unusable_rows() -> None:
    top = top_of_book(quotes_market())
    assert isinstance(top, TopOfBook)
    expected = {
        "bid": [[20_000.0, 20_000.0, NAN, NAN, 20_000.5], [NAN, NAN, NAN, 19_995.0, 19_995.0]],
        "ask": [[20_001.0, 20_001.0, NAN, NAN, 20_001.0], [NAN, NAN, NAN, 19_996.0, 19_996.0]],
        "bid_size": [[100.0, 100.0, NAN, NAN, 200.0], [NAN, NAN, NAN, 50.0, 50.0]],
        "ask_size": [[300.0, 300.0, NAN, NAN, 200.0], [NAN, NAN, NAN, 150.0, 150.0]],
        "mid": [[20_000.5, 20_000.5, NAN, NAN, 20_000.75], [NAN, NAN, NAN, 19_995.5, 19_995.5]],
        # (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
        "microprice": [[20_000.25, 20_000.25, NAN, NAN, 20_000.75], [NAN, NAN, NAN, 19_995.25, 19_995.25]],
        "spread_bps": [
            [1.0 / 20_000.5 * 1e4, 1.0 / 20_000.5 * 1e4, NAN, NAN, 0.5 / 20_000.75 * 1e4],
            [NAN, NAN, NAN, 1.0 / 19_995.5 * 1e4, 1.0 / 19_995.5 * 1e4],
        ],
    }
    for name, want in expected.items():
        got = getattr(top, name)
        assert got.dtype == np.float64 and got.shape == (2, 5) and not got.flags.writeable, name
        np.testing.assert_array_equal(got, np.array(want), err_msg=name)


def test_top_of_book_rejects_inconsistent_arrays() -> None:
    good = np.zeros((2, 3))
    with pytest.raises(InvalidDataError):
        TopOfBook(good, good, good, good, good, good, np.zeros((2, 4)))
    with pytest.raises(InvalidDataError):
        TopOfBook(good, good, good, good, good, good, np.zeros((2, 3), np.float32))


def test_ewma_time_skips_nan_holds_state_and_ignores_same_time_values() -> None:
    x = np.array([NAN, 1.0, NAN, 3.0, 5.0, 7.0])
    t = np.array([0, 10, 20, 30, 30, 40], np.int64)
    out = ewma_time(x, t, 10)
    w = math.exp(-2.0)  # 20 ns since the last finite value (t=10); the NaN at t=20 is skipped
    m3 = w * 1.0 + (1.0 - w) * 3.0
    m5 = math.exp(-1.0) * m3 + (1.0 - math.exp(-1.0)) * 7.0
    # the value at t=30 arriving with dt=0 has weight 1 - exp(0) = 0
    np.testing.assert_allclose(out, [NAN, 1.0, 1.0, m3, m3, m5], rtol=1e-15, atol=0.0)
    assert out.dtype == np.float64 and not out.flags.writeable
    np.testing.assert_array_equal(ewma_time(np.full(3, NAN), np.arange(3), 5), [NAN, NAN, NAN])


def test_ewma_time_validates_inputs() -> None:
    x, t = np.ones(3), np.arange(3, dtype=np.int64)
    with pytest.raises(InvalidConfigError):
        ewma_time(x, t, 0)
    with pytest.raises(InvalidConfigError):
        ewma_time(x, t, True)
    with pytest.raises(InvalidDataError):
        ewma_time(x, t[:2], 5)
    with pytest.raises(InvalidDataError):
        ewma_time(x, np.array([0, 2, 1]), 5)
    with pytest.raises(InvalidDataError):
        ewma_time(x, t.astype(np.float64), 5)  # clocks are integer nanoseconds
    with pytest.raises(InvalidDataError):
        ewma_time(np.array([1.0, math.inf, 2.0]), t, 5)
    with pytest.raises(InvalidDataError):
        ewma_time(np.ones((3, 1)), t, 5)


def test_basis_deviation_measures_a_jump_against_previous_values_only() -> None:
    prep = lead_lag_market()
    g = basis_deviation_bps(prep, BMX, DBT, 2 * MS, price="mid")
    # b = [NaN, 0, JUMP, 0]; EWMA(b) = [NaN, 0, (1 - e^-1) JUMP, ...] with dt = 2 ms = tau.
    np.testing.assert_allclose(g, [NAN, NAN, JUMP_BPS, -(1.0 - math.exp(-1.0)) * JUMP_BPS], rtol=1e-13, atol=0.0)
    assert g[2] > 0.0  # leader rich vs. its recent relation: the follower is expected to rise
    assert basis_deviation_bps(prep, DBT, BMX, 2 * MS, price="mid")[2] < 0.0
    assert not g.flags.writeable


def test_basis_deviation_holds_a_jump_through_its_receipt_cohort() -> None:
    """The replay engine acts after a cohort's last event. A print received together with the leader
    jump must not absorb part of it: the average excludes the whole cohort, not just the event itself.
    (Before: the cohort's first value entered the EWMA at once and the print saw e^-1 * JUMP.)"""
    prep = cohort_market()
    assert prep.stream.recv_ns.tolist() == [MS, 2 * MS, 4 * MS, 4 * MS, 6 * MS]
    g = basis_deviation_bps(prep, BMX, DBT, 2 * MS, price="mid")
    # b = [NaN, 0, JUMP, JUMP, 0]; the state before the 6 ms cohort folds in the 4 ms cohort once (dt = 2 ms).
    np.testing.assert_allclose(
        g, [NAN, NAN, JUMP_BPS, JUMP_BPS, -(1.0 - math.exp(-1.0)) * JUMP_BPS], rtol=1e-13, atol=0.0
    )


def test_a_jump_that_reverts_inside_its_cohort_leaves_no_deviation() -> None:
    """Leader and follower both move in one cohort: the cohort's last basis (0) is what enters the
    average, so neither the cohort's last event nor later ones report an absorbed jump."""
    venue_a = tapes_from_rows(
        DEMO_INVERSE_A,
        [
            (1_000, 900, [(40_000, 100)], [(40_002, 100)]),
            (4_000, 3_900, [(40_020, 100)], [(40_022, 100)]),  # leader jumps ...
        ],
    )
    venue_b = tapes_from_rows(
        DEMO_INVERSE_B,
        [
            (2_000, 1_000, [(40_000, 100)], [(40_002, 100)]),
            (4_000, 3_000, [(40_020, 100)], [(40_022, 100)]),  # ... and the follower catches up in the same cohort
            (6_000, 5_000, [(40_020, 100)], [(40_022, 100)]),
        ],
    )
    prep = prepare(venue_a, venue_b)
    assert prep.stream.recv_ns.tolist() == [MS, 2 * MS, 4 * MS, 4 * MS, 6 * MS]
    g = basis_deviation_bps(prep, BMX, DBT, 2 * MS, price="mid")
    np.testing.assert_array_equal(g, [NAN, NAN, JUMP_BPS, 0.0, 0.0])


def test_micro_price_returns_and_gaps_follow_the_size_weighted_price() -> None:
    prep = skew_market()
    np.testing.assert_array_equal(trailing_return_bps(prep, BMX, 2 * MS, "mid"), [NAN, NAN, 0.0, 0.0])
    np.testing.assert_array_equal(trailing_return_bps(prep, BMX, 2 * MS, "micro"), [NAN, NAN, MICRO_JUMP_BPS, 0.0])
    np.testing.assert_array_equal(lead_gap_bps(prep, BMX, DBT, 2 * MS), [NAN, NAN, 0.0, 0.0])
    np.testing.assert_array_equal(lead_gap_bps(prep, BMX, DBT, 2 * MS, price="micro"), [NAN, NAN, MICRO_JUMP_BPS, 0.0])
    np.testing.assert_array_equal(lead_gap_bps(prep, DBT, BMX, 2 * MS, price="micro"), [NAN, NAN, -MICRO_JUMP_BPS, 0.0])


def test_trailing_return_uses_the_price_observed_as_of_receipt_minus_window() -> None:
    prep = lead_lag_market()
    np.testing.assert_array_equal(trailing_return_bps(prep, BMX, 2 * MS, "mid"), [NAN, NAN, JUMP_BPS, 0.0])
    np.testing.assert_array_equal(trailing_return_bps(prep, DBT, 2 * MS, "mid"), [NAN, NAN, 0.0, JUMP_BPS])
    # one nanosecond more and the event exactly ``window`` earlier is no longer "as of"
    np.testing.assert_array_equal(trailing_return_bps(prep, BMX, 2 * MS + 1, "mid"), [NAN, NAN, JUMP_BPS, JUMP_BPS])
    np.testing.assert_array_equal(lead_gap_bps(prep, BMX, DBT, 2 * MS), [NAN, NAN, JUMP_BPS, -JUMP_BPS])


def flow_market() -> PreparedMarket:
    bids = [(40_000 - k, 100 * (k + 1)) for k in range(6)]  # top-5 bids: 1,500 USD
    asks = [(40_001 + k, 100) for k in range(6)]  # top-5 asks: 500 USD
    book = tapes_from_rows(DEMO_INVERSE_A, [(500, 400, bids, asks), (2_000, 1_900, bids, asks)])
    # events: e0 0.5ms book | e1 1ms buy 100 | e2 2ms book | e3 3ms sell 300
    return prepare(with_trades(book, [(1_000, 40_001, 100, 1), (3_000, 40_000, 300, -1)]))


def test_trade_flow_decays_signed_aggressor_units_to_every_event() -> None:
    prep = flow_market()
    raw = trade_flow(prep, BMX, 2 * MS, normalize=False)
    expected = np.array([0.0, 100.0, 100.0 * math.exp(-0.5), 100.0 * math.exp(-1.0) - 300.0])
    np.testing.assert_allclose(raw, expected, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(trade_flow(prep, BMX, 2 * MS, normalize=True), expected / 2_000.0, rtol=1e-15, atol=0.0)


def test_normalized_flow_is_nan_without_a_usable_book() -> None:
    one_sided = tapes_from_rows(DEMO_INVERSE_A, [(2_000, 1_900, [(40_000, 100)], [])])
    prep = prepare(with_trades(one_sided, [(1_000, 40_000, 100, 1)]))  # trade, then book
    np.testing.assert_array_equal(trade_flow(prep, BMX, MS, normalize=True), [NAN, NAN])
    np.testing.assert_allclose(trade_flow(prep, BMX, MS, normalize=False), [100.0, 100.0 * math.exp(-1.0)], rtol=1e-15)


def test_ofi_hand_computed_cont_kukanov_stoikov_increments() -> None:
    rows = [
        (1_000, 900, [(40_000, 100)], [(40_002, 200)]),
        (2_000, 1_900, [(40_000, 150)], [(40_002, 200)]),  # bid queue +50 -> e = +50
        (3_000, 2_900, [(40_001, 80)], [(40_002, 100)]),  # bid up (+80), ask -200+... -> +180
        (4_000, 3_900, [(40_000, 300)], [(40_001, 50)]),  # bid down (-80), ask down (-50) -> -130
    ]
    out = ofi(prepare(tapes_from_rows(DEMO_INVERSE_A, rows)), BMX, MS)
    s2 = 50.0 * math.exp(-1.0) + 180.0
    s3 = s2 * math.exp(-1.0) - 130.0
    # divided by the observed top-1 depth (bid + ask USD at level 0)
    np.testing.assert_allclose(out, [0.0, 50.0 / 350.0, s2 / 180.0, s3 / 350.0], rtol=1e-14, atol=0.0)


def test_ofi_skips_pairs_that_touch_an_unusable_book() -> None:
    rows = [
        (1_000, 900, [(40_000, 100)], [(40_002, 200)]),
        (2_000, 1_900, [(40_003, 100)], [(40_002, 200)]),  # crossed
        (3_000, 2_900, [(40_001, 500)], [(40_003, 100)]),
    ]
    out = ofi(prepare(tapes_from_rows(DEMO_INVERSE_A, rows)), BMX, MS)
    np.testing.assert_array_equal(out, [0.0, NAN, 0.0])


def test_imbalance_over_the_top_k_observed_levels() -> None:
    rows = [
        (1_000, 900, [(40_000, 100), (39_999, 300)], [(40_001, 200), (40_002, 100)]),
        (2_000, 1_900, [], []),  # empty book -> unusable
    ]
    prep = prepare(tapes_from_rows(DEMO_INVERSE_A, rows))
    np.testing.assert_array_equal(imbalance(prep, BMX, 1), [(100 - 200) / 300, NAN])
    np.testing.assert_array_equal(imbalance(prep, BMX, 2), [(400 - 300) / 700, NAN])
    np.testing.assert_array_equal(imbalance(prep, 0, DEPTH), [(400 - 300) / 700, NAN])


def test_depth_counts_only_the_populated_levels() -> None:
    tapes = tapes_from_rows(DEMO_INVERSE_A, [(1_000, 900, [(40_000, 100)], [(40_001, 300)])])
    book = tapes.book
    bid_px, bid_sz = book.bid_px.copy(), book.bid_sz.copy()
    bid_px[0, 1], bid_sz[0, 1] = 39_999, 5_000  # stale level beyond n_bid = 1
    stale = dataclasses.replace(book, bid_px=bid_px, bid_sz=bid_sz)
    prep = prepare(dataclasses.replace(tapes, book=stale))
    np.testing.assert_array_equal(imbalance(prep, BMX, 2), [(100 - 300) / 400])
    np.testing.assert_array_equal(trade_flow(prep, BMX, MS, normalize=True), [0.0])


def test_a_stream_built_for_other_data_is_rejected() -> None:
    small, large = lead_lag_market(), quotes_market()
    with pytest.raises(InvalidDataError):
        PreparedMarket.build(small.data, stream=large.stream)  # rows beyond small's tapes
    with pytest.raises(InvalidDataError):
        PreparedMarket.build(large.data, stream=small.stream)  # different observation clocks
    # Replay permits genuine subsets; the signal registry requires complete tape coverage.
    stream = EventStream(*(getattr(large.stream, key)[:-1] for key in ("recv_ns", "venue", "kind", "row")))
    short = PreparedMarket.build(large.data, stream=stream)
    with pytest.raises(InvalidDataError):
        build_signal(short, {"kind": "imbalance", "venue": BMX, "levels": 1})
    with pytest.raises(InvalidDataError):
        top_of_book(short)


def test_durations_are_bounded_by_2_pow_53_ns() -> None:
    """Beyond 2**53 ns ``float(dt)`` is no longer exact and clock arithmetic nears int64; the bound is inclusive."""
    prep = lead_lag_market()
    x, t = np.ones(2), np.array([0, 1], np.int64)
    assert trailing_return_bps(prep, BMX, 2**53, "mid").shape == (4,)
    assert ewma_time(x, t, 2**53).shape == (2,)
    too_long: list[Callable[[], object]] = [
        lambda: basis_deviation_bps(prep, BMX, DBT, 2**53 + 1),
        lambda: trailing_return_bps(prep, BMX, 2**53 + 1, "mid"),
        lambda: lead_gap_bps(prep, BMX, DBT, 2**53 + 1),
        lambda: trade_flow(prep, BMX, 2**53 + 1),
        lambda: ofi(prep, BMX, 2**53 + 1),
        lambda: ewma_time(x, t, 2**53 + 1),
    ]
    for i, call in enumerate(too_long):
        with pytest.raises(InvalidConfigError):
            call()
            pytest.fail(f"call {i} was accepted")


@pytest.mark.parametrize(
    "key", [" venue_a/ALPHA-USD", "venue_a/ALPHA-USD ", "venue_a/ALPHA-USD\n", "venue_a/ ALPHA-USD"]
)
def test_instrument_keys_with_whitespace_are_rejected(key: str) -> None:
    spec = {"kind": "imbalance", "venue": key, "levels": 2}
    with pytest.raises(InvalidConfigError):
        canonical_spec(spec)
    with pytest.raises(InvalidConfigError):
        spec_id(spec)
    with pytest.raises(InvalidConfigError):
        build_signal(lead_lag_market(), spec)
    with pytest.raises(InvalidConfigError):
        imbalance(lead_lag_market(), key, 2)


def test_public_functions_validate_their_arguments() -> None:
    prep = lead_lag_market()
    bad_calls: list[Callable[[], object]] = [
        lambda: basis_deviation_bps(prep, BMX, BMX, MS),
        lambda: basis_deviation_bps(prep, BMX, DBT, 0),
        lambda: basis_deviation_bps(prep, BMX, DBT, MS, price="last"),
        lambda: trailing_return_bps(prep, "kraken/ALPHA-USD", MS, "mid"),
        lambda: trailing_return_bps(prep, 2, MS, "mid"),
        lambda: trailing_return_bps(prep, True, MS, "mid"),
        lambda: trailing_return_bps(prep, BMX, -1, "mid"),
        lambda: lead_gap_bps(prep, 0, 0, MS),
        lambda: trade_flow(prep, BMX, 1.5, normalize=False),  # type: ignore[arg-type]
        lambda: trade_flow(prep, BMX, MS, normalize=1),  # type: ignore[arg-type]
        lambda: ofi(prep, BMX, True),
        lambda: imbalance(prep, BMX, 0),
        lambda: imbalance(prep, BMX, DEPTH + 1),
        lambda: observed_rows(prep.data),  # type: ignore[arg-type]
    ]
    for i, call in enumerate(bad_calls):
        with pytest.raises(InvalidConfigError):
            call()
            pytest.fail(f"call {i} was accepted")


# ---------------------------------------------------------------------------
# Independent slow oracle
# ---------------------------------------------------------------------------


def rel_bps(price: float, base: float) -> float:
    return (price / base - 1.0) * 1e4 if price > 0 and base > 0 else NAN


def oracle_ewma(x: Sequence[float], t: Sequence[int], tau: int) -> list[float]:
    """Textbook form ``m = w m + (1 - w) x`` with ``w = exp(-dt / tau)``."""
    out: list[float] = []
    m: float | None = None
    t_last = 0
    for xi, ti in zip(x, t, strict=True):
        if not math.isnan(xi):
            if m is None:
                m = xi
            else:
                w = math.exp(-(ti - t_last) / tau)
                m = w * m + (1.0 - w) * xi
            t_last = ti
        out.append(NAN if m is None else m)
    return out


class Oracle:
    """Slow re-derivation from the raw tapes: explicit sort, explicit validity, direct
    (non-recursive) decayed sums. Uses neither the library kernels nor EventStream."""

    def __init__(self, data: MarketData) -> None:
        self.data = data
        self.keys = [t.instrument.key for t in data.venues]
        events: list[tuple[int, int, int, int]] = []
        for v, tapes in enumerate(data.venues):
            clocks = ((BOOK_EVENT, tapes.book.recv_ns), (TRADE_EVENT, tapes.trades.recv_ns))
            for kind, clock in clocks:
                events.extend((int(t), v, kind, r) for r, t in enumerate(clock.tolist()))
        self.events = sorted(events)
        self.recv = [e[0] for e in self.events]
        n, nv = len(self.events), len(data.venues)
        self.rows = [[-1] * n for _ in range(nv)]
        current = [-1] * nv
        for i, (_, v, kind, r) in enumerate(self.events):
            if kind == BOOK_EVENT:
                current[v] = r
            for u in range(nv):
                self.rows[u][i] = current[u]

    def usable(self, v: int, r: int) -> bool:
        if r < 0:
            return False
        b = self.data.venues[v].book
        return (
            int(b.kind[r]) == BOOK_SNAPSHOT
            and int(b.n_bid[r]) > 0
            and int(b.n_ask[r]) > 0
            and int(b.bid_px[r, 0]) < int(b.ask_px[r, 0])
        )

    def quote(self, v: int, i: int) -> tuple[float, float, float, float] | None:
        r = self.rows[v][i]
        if not self.usable(v, r):
            return None
        b = self.data.venues[v].book
        tick = float(self.data.venues[v].instrument.tick_size)
        return (
            int(b.bid_px[r, 0]) * tick,
            int(b.ask_px[r, 0]) * tick,
            float(int(b.bid_sz[r, 0])),
            float(int(b.ask_sz[r, 0])),
        )

    def price(self, v: int, kind: str) -> list[float]:
        out: list[float] = []
        for i in range(len(self.events)):
            q = self.quote(v, i)
            if q is None:
                out.append(NAN)
                continue
            bid, ask, qb, qa = q
            if kind == "mid":
                out.append(0.5 * (bid + ask))
            else:
                out.append((bid * qa + ask * qb) / (qb + qa) if qb + qa > 0 else NAN)
        return out

    def sides(self, v: int, i: int, levels: int) -> tuple[int, int] | None:
        r = self.rows[v][i]
        if not self.usable(v, r):
            return None
        b = self.data.venues[v].book
        bid = sum(int(b.bid_sz[r, k]) for k in range(min(levels, int(b.n_bid[r]))))
        ask = sum(int(b.ask_sz[r, k]) for k in range(min(levels, int(b.n_ask[r]))))
        return bid, ask

    def imbalance(self, v: int, levels: int) -> list[float]:
        out: list[float] = []
        for i in range(len(self.events)):
            s = self.sides(v, i, levels)
            out.append(NAN if s is None or sum(s) <= 0 else (s[0] - s[1]) / (s[0] + s[1]))
        return out

    def trailing(self, v: int, window: int, kind: str) -> list[float]:
        p = self.price(v, kind)
        out: list[float] = []
        for i, t in enumerate(self.recv):
            j = bisect.bisect_right(self.recv, t - window) - 1
            out.append(rel_bps(p[i], p[j]) if j >= 0 else NAN)
        return out

    def basis(self, leader: str, follower: str, tau: int, kind: str) -> tuple[list[float], float]:
        """``g_i = b_i - M_i`` with ``M_i`` the textbook EWMA over the receipt cohorts strictly
        before event ``i``; a cohort contributes its last finite basis, at the cohort's clock."""
        pl = self.price(self.keys.index(leader), kind)
        pf = self.price(self.keys.index(follower), kind)
        b = [rel_bps(x, y) for x, y in zip(pl, pf, strict=True)]
        g: list[float] = []
        m: float | None = None
        t_last = 0
        pending: float | None = None
        for i, (t, value) in enumerate(zip(self.recv, b, strict=True)):
            if i > 0 and t != self.recv[i - 1] and pending is not None:  # the previous cohort is complete
                if m is None:
                    m = pending
                else:
                    w = math.exp(-(self.recv[i - 1] - t_last) / tau)
                    m = w * m + (1.0 - w) * pending
                t_last = self.recv[i - 1]
                pending = None
            g.append(NAN if m is None else value - m)
            if not math.isnan(value):
                pending = value
        scale = max((abs(x) for x in b if not math.isnan(x)), default=0.0)
        return g, 1e-12 * (1.0 + scale)

    def ofi_increment(self, v: int, r: int) -> float | None:
        if r == 0 or not self.usable(v, r) or not self.usable(v, r - 1):
            return None
        b = self.data.venues[v].book
        b0, b1 = int(b.bid_px[r - 1, 0]), int(b.bid_px[r, 0])
        a0, a1 = int(b.ask_px[r - 1, 0]), int(b.ask_px[r, 0])
        qb0, qb1 = int(b.bid_sz[r - 1, 0]), int(b.bid_sz[r, 0])
        qa0, qa1 = int(b.ask_sz[r - 1, 0]), int(b.ask_sz[r, 0])
        return float((b1 >= b0) * qb1 - (b1 <= b0) * qb0 - (a1 <= a0) * qa1 + (a1 >= a0) * qa0)

    def trade_increment(self, v: int, r: int) -> float | None:
        trades = self.data.venues[v].trades
        return float(int(trades.side_label[r]) * int(trades.sz[r]))

    def decayed(
        self, v: int, kind: int, increment: Callable[[int, int], float | None], tau: int, levels: int | None
    ) -> tuple[list[float], list[float]]:
        """sum_j x_j exp(-(t_i - t_j) / tau) over the venue's ``kind`` events j <= i."""
        seen: list[tuple[int, float]] = []
        values: list[float] = []
        tols: list[float] = []
        for i, (t, u, k, r) in enumerate(self.events):
            if u == v and k == kind:
                x = increment(v, r)
                if x is not None:
                    seen.append((t, x))
            terms = [x * math.exp(-(t - tj) / tau) for tj, x in seen]
            total, scale = math.fsum(terms), math.fsum(abs(z) for z in terms)
            if levels is not None:
                s = self.sides(v, i, levels)
                depth = 0 if s is None else s[0] + s[1]
                total, scale = (total / depth, scale / depth) if depth > 0 else (NAN, 0.0)
            values.append(total)
            tols.append(1e-12 * scale)
        return values, tols

    def signal(self, spec: Mapping[str, Any]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n = len(self.events)
        kind = spec["kind"]
        exact = np.zeros(n)
        if kind == "linear":
            total, tol = np.zeros(n), np.zeros(n)
            for idx, term in enumerate(spec["terms"]):
                part, part_tol = self.signal(term["spec"])
                w = float(term["weight"])
                total = w * part if idx == 0 else total + w * part
                tol = tol + abs(w) * part_tol
            return total, tol
        if kind == "basis_dev":
            g, tol_b = self.basis(spec["leader"], spec["follower"], spec["tau_ms"] * MS, spec["price"])
            return np.array(g), np.full(n, tol_b)
        if kind == "lead_gap":
            window = spec["window_ms"] * MS
            lead = self.trailing(self.keys.index(spec["leader"]), window, "mid")
            follow = self.trailing(self.keys.index(spec["follower"]), window, "mid")
            return np.array(lead) - np.array(follow), exact
        v = self.keys.index(spec["venue"])
        if kind == "return":
            return np.array(self.trailing(v, spec["window_ms"] * MS, "mid")), exact
        if kind == "imbalance":
            return np.array(self.imbalance(v, spec["levels"])), exact
        tau = spec["tau_ms"] * MS
        if kind == "trade_flow":
            values, tols = self.decayed(v, TRADE_EVENT, self.trade_increment, tau, 5 if spec["normalize"] else None)
        else:
            assert kind == "ofi"
            values, tols = self.decayed(v, BOOK_EVENT, self.ofi_increment, tau, 1)
        return np.array(values), np.array(tols)


def assert_close(got: NDArray[np.float64], want: NDArray[np.float64], tol: NDArray[np.float64], label: str) -> None:
    assert got.shape == want.shape, label
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want), err_msg=f"{label}: NaN pattern")
    ok = ~np.isnan(want)
    excess = np.abs(got[ok] - want[ok]) - tol[ok]
    assert excess.size == 0 or float(excess.max()) <= 0.0, f"{label}: off by {excess.max()}"


ALL_SPECS: list[dict[str, Any]] = [
    {"kind": "basis_dev", "leader": BMX, "follower": DBT, "tau_ms": 50, "price": "mid"},
    {"kind": "basis_dev", "leader": DBT, "follower": BMX, "tau_ms": 3, "price": "micro"},
    {"kind": "lead_gap", "leader": BMX, "follower": DBT, "window_ms": 30},
    {"kind": "lead_gap", "leader": DBT, "follower": BMX, "window_ms": 7},
    {"kind": "trade_flow", "venue": BMX, "tau_ms": 40, "normalize": True},
    {"kind": "trade_flow", "venue": DBT, "tau_ms": 2, "normalize": False},
    {"kind": "ofi", "venue": BMX, "tau_ms": 25},
    {"kind": "ofi", "venue": DBT, "tau_ms": 1},
    {"kind": "imbalance", "venue": DBT, "levels": 3},
    {"kind": "imbalance", "venue": BMX, "levels": DEPTH},
    {"kind": "return", "venue": BMX, "window_ms": 15},
    {"kind": "return", "venue": DBT, "window_ms": 120},
    {
        "kind": "linear",
        "terms": [
            {
                "spec": {"kind": "basis_dev", "leader": BMX, "follower": DBT, "tau_ms": 20, "price": "mid"},
                "weight": 0.5,
            },
            {"spec": {"kind": "imbalance", "venue": BMX, "levels": 2}, "weight": -2.0},
            {"spec": {"kind": "trade_flow", "venue": DBT, "tau_ms": 10, "normalize": True}, "weight": 3},
        ],
    },
]
SPEC_IDS = [f"{i}-{spec['kind']}" for i, spec in enumerate(ALL_SPECS)]


@given(seed=st.integers(0, 50_000))
@settings(max_examples=8, deadline=None)
def test_every_kind_matches_an_independent_python_oracle(seed: int) -> None:
    data = rich_market(seed)
    prep = PreparedMarket.build(data)
    oracle = Oracle(data)
    stream = prep.stream
    assert oracle.events == list(
        zip(stream.recv_ns.tolist(), stream.venue.tolist(), stream.kind.tolist(), stream.row.tolist(), strict=True)
    )
    for label, spec in zip(SPEC_IDS, ALL_SPECS, strict=True):
        want, tol = oracle.signal(spec)
        assert_close(build_signal(prep, spec), want, tol, label)
    # The registry evaluates returns and gaps on mids only; the public functions also take the microprice.
    exact = np.zeros(len(oracle.events))
    micro = [np.array(oracle.trailing(v, 9 * MS, "micro")) for v in range(len(oracle.keys))]
    for v, key in enumerate(oracle.keys):
        assert_close(trailing_return_bps(prep, key, 9 * MS, "micro"), micro[v], exact, f"return-micro-{key}")
    assert_close(
        lead_gap_bps(prep, oracle.keys[0], oracle.keys[1], 9 * MS, price="micro"),
        micro[0] - micro[1],
        exact,
        "lead_gap-micro",
    )
    assert_close(
        lead_gap_bps(prep, oracle.keys[1], oracle.keys[0], 9 * MS, price="micro"),
        micro[1] - micro[0],
        exact,
        "lead_gap-micro-reversed",
    )


@given(seed=st.integers(0, 50_000))
@settings(max_examples=8, deadline=None)
def test_observed_rows_and_top_of_book_match_the_oracle_exactly(seed: int) -> None:
    data = rich_market(seed)
    prep = PreparedMarket.build(data)
    oracle = Oracle(data)
    n = len(oracle.events)
    np.testing.assert_array_equal(observed_rows(prep), np.array(oracle.rows, np.int64))
    top = top_of_book(prep)
    for v in range(len(data.venues)):
        quotes = [oracle.quote(v, i) for i in range(n)]
        columns = {
            "bid": [NAN if q is None else q[0] for q in quotes],
            "ask": [NAN if q is None else q[1] for q in quotes],
            "bid_size": [NAN if q is None else q[2] for q in quotes],
            "ask_size": [NAN if q is None else q[3] for q in quotes],
            "mid": oracle.price(v, "mid"),
            "microprice": oracle.price(v, "micro"),
            "spread_bps": [NAN if q is None else (q[1] - q[0]) / (0.5 * (q[0] + q[1])) * 1e4 for q in quotes],
        }
        for name, want in columns.items():
            np.testing.assert_array_equal(getattr(top, name)[v], np.array(want), err_msg=name)


@given(seed=st.integers(0, 50_000), tau=st.integers(1, 5_000), k=st.integers(0, 198))
@settings(max_examples=40, deadline=None)
def test_ewma_time_matches_the_textbook_recursion_and_is_causal(seed: int, tau: int, k: int) -> None:
    rng = np.random.default_rng(seed)
    n = 200
    t = np.cumsum(rng.integers(0, 2, n) * rng.integers(0, 3_000, n)).astype(np.int64)
    x = rng.normal(0.0, 5.0, n)
    x[rng.random(n) < 0.2] = NAN
    got = ewma_time(x, t, tau)
    want = oracle_ewma(x.tolist(), t.tolist(), tau)
    scale = 1.0 + float(np.nanmax(np.abs(x)))
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12 * scale)
    y = x.copy()
    y[k + 1 :] = rng.normal(0.0, 50.0, n - k - 1)
    assert ewma_time(y, t, tau)[: k + 1].tobytes() == got[: k + 1].tobytes()


# ---------------------------------------------------------------------------
# Causality: rewriting the future never changes issued values
# ---------------------------------------------------------------------------


def later(clock: NDArray[np.int64], start: int, rng: np.random.Generator) -> NDArray[np.int64]:
    out = clock.copy()
    out[start:] += 1 + np.cumsum(rng.integers(0, 2 * MS, out.shape[0] - start))
    return out


def mutate_book(book: BookTape, r0: int, rng: np.random.Generator) -> BookTape:
    m = len(book) - r0
    kind = book.kind.copy()
    kind[r0:] = np.where(rng.random(m) < 0.2, BOOK_CLEAR, BOOK_SNAPSHOT)
    n_bid, n_ask = book.n_bid.copy(), book.n_ask.copy()
    n_bid[r0:] = rng.integers(0, DEPTH + 1, m)
    n_ask[r0:] = rng.integers(0, DEPTH + 1, m)
    bid_px, ask_px = book.bid_px.copy(), book.ask_px.copy()
    bid_px[r0:] += rng.integers(-25, 26, (m, 1)).astype(np.int32)
    ask_px[r0:] += rng.integers(-25, 26, (m, 1)).astype(np.int32)
    bid_sz, ask_sz = book.bid_sz.copy(), book.ask_sz.copy()
    bid_sz[r0:] = rng.integers(1, 500, (m, DEPTH)) * 10
    ask_sz[r0:] = rng.integers(1, 500, (m, DEPTH)) * 10
    return BookTape(
        later(book.recv_ns, r0, rng), later(book.exch_ns, r0, rng), kind, n_bid, n_ask, bid_px, bid_sz, ask_px, ask_sz
    )


def mutate_trades(trades: TradeTape, q0: int, rng: np.random.Generator) -> TradeTape:
    m = len(trades) - q0
    px, sz, side = trades.px.copy(), trades.sz.copy(), trades.side_label.copy()
    px[q0:] = rng.integers(39_000, 41_000, m)
    sz[q0:] = rng.integers(1, 1_000, m) * 10
    side[q0:] = -side[q0:]
    return TradeTape(later(trades.recv_ns, q0, rng), later(trades.exch_ns, q0, rng), px, sz, side)


def mutate_after(data: MarketData, k: int, seed: int) -> MarketData:
    """Rewrite every tape row whose stream index is > k (later receipt times, other prices,
    sizes, book kinds and aggressor sides); rows at stream index <= k are untouched."""
    stream = EventStream.build(data)
    rng = np.random.default_rng(seed)
    venue, kind = stream.venue[: k + 1], stream.kind[: k + 1]
    venues = []
    for v, tapes in enumerate(data.venues):
        r0 = int(np.count_nonzero((venue == v) & (kind == BOOK_EVENT)))
        q0 = int(np.count_nonzero((venue == v) & (kind == TRADE_EVENT)))
        venues.append(
            dataclasses.replace(
                tapes, book=mutate_book(tapes.book, r0, rng), trades=mutate_trades(tapes.trades, q0, rng)
            )
        )
    return MarketData(tuple(venues))


def cuts(prep: PreparedMarket) -> list[int]:
    recv = prep.stream.recv_ns
    n = recv.shape[0]
    ties = np.flatnonzero(recv[1:] == recv[:-1])  # k with recv[k] == recv[k + 1]: inside a cohort
    assert ties.size > 0
    return [0, int(ties[ties.size // 2]), n // 3, (2 * n) // 3, n - 2]


def mutated_prep(data: MarketData, prep: PreparedMarket, k: int) -> PreparedMarket:
    altered = PreparedMarket.build(mutate_after(data, k, seed=k))
    for name in ("recv_ns", "venue", "kind", "row"):
        np.testing.assert_array_equal(getattr(altered.stream, name)[: k + 1], getattr(prep.stream, name)[: k + 1])
    return altered


@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
def test_features_receipt_causal(spec: Mapping[str, Any]) -> None:
    """Rewriting everything after stream index k leaves values 0..k byte-identical."""
    data = rich_market(29)
    prep = PreparedMarket.build(data)
    full = build_signal(prep, spec)
    changed = False
    for k in cuts(prep):
        other = build_signal(mutated_prep(data, prep, k), spec)
        assert other[: k + 1].tobytes() == full[: k + 1].tobytes(), f"k={k}"
        changed |= other.tobytes() != full.tobytes()
    assert changed, "the mutation never reached the signal, so the check would be vacuous"


def test_observed_state_is_receipt_causal() -> None:
    data = rich_market(31)
    prep = PreparedMarket.build(data)
    rows, top = observed_rows(prep), top_of_book(prep)
    for k in cuts(prep):
        altered = mutated_prep(data, prep, k)
        assert observed_rows(altered)[:, : k + 1].tobytes() == rows[:, : k + 1].tobytes()
        other = top_of_book(altered)
        for field in dataclasses.fields(TopOfBook):
            a, b = getattr(other, field.name), getattr(top, field.name)
            assert a[:, : k + 1].tobytes() == b[:, : k + 1].tobytes(), (field.name, k)


# ---------------------------------------------------------------------------
# Registry, canonical ids and the pure-Python fallback
# ---------------------------------------------------------------------------


def test_registry_dispatches_to_the_public_functions() -> None:
    prep = PreparedMarket.build(rich_market(4))
    pairs: list[tuple[dict[str, Any], NDArray[np.float64]]] = [
        (
            {"kind": "basis_dev", "leader": BMX, "follower": DBT, "tau_ms": 50, "price": "micro"},
            basis_deviation_bps(prep, BMX, DBT, 50 * MS, price="micro"),
        ),
        ({"kind": "lead_gap", "leader": DBT, "follower": BMX, "window_ms": 7}, lead_gap_bps(prep, 1, 0, 7 * MS)),
        (
            {"kind": "trade_flow", "venue": BMX, "tau_ms": 40, "normalize": True},
            trade_flow(prep, BMX, 40 * MS, normalize=True),
        ),
        ({"kind": "ofi", "venue": DBT, "tau_ms": 0.5}, ofi(prep, DBT, 500_000)),
        ({"kind": "imbalance", "venue": BMX, "levels": 4}, imbalance(prep, 0, 4)),
        ({"kind": "return", "venue": DBT, "window_ms": 12}, trailing_return_bps(prep, DBT, 12 * MS, "mid")),
    ]
    n = len(prep.stream)
    for spec, direct in pairs:
        got = build_signal(prep, spec)
        assert got.dtype == np.float64 and got.shape == (n,) and not got.flags.writeable
        assert got.tobytes() == direct.tobytes(), spec["kind"]
    lin = {"kind": "linear", "terms": [{"spec": pairs[0][0], "weight": 0.5}, {"spec": pairs[4][0], "weight": -2}]}
    got = build_signal(prep, lin)
    np.testing.assert_array_equal(got, 0.5 * pairs[0][1] + -2.0 * pairs[4][1])
    np.testing.assert_array_equal(np.isnan(got), np.isnan(pairs[0][1]) | np.isnan(pairs[4][1]))
    assert np.isnan(got).any() and not np.isnan(got).all()


def test_basis_deviation_is_the_basis_minus_the_ewma_of_earlier_cohorts() -> None:
    """``g_e = b_e - M(t_e^-)``: ``M`` is :func:`ewma_time` over one value per receipt time (the cohort's
    last finite basis at the cohort's clock), read at the previous cohort. The composition is exact."""
    prep = PreparedMarket.build(rich_market(8))
    top = top_of_book(prep)
    b = (top.mid[0] / top.mid[1] - 1.0) * 1e4
    t = prep.stream.recv_ns
    starts = np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
    ends = np.r_[starts[1:], t.shape[0]]
    assert (ends - starts > 1).any()  # the market has cohorts
    last_finite = []
    for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
        finite = [x for x in b[s:e].tolist() if not math.isnan(x)]
        last_finite.append(finite[-1] if finite else NAN)
    m = ewma_time(np.array(last_finite), t[starts], 30 * MS)  # the state after each cohort
    cohort_of = np.repeat(np.arange(starts.shape[0]), ends - starts)
    previous = np.concatenate([[NAN], m])[cohort_of]  # the state before each event's cohort
    np.testing.assert_array_equal(basis_deviation_bps(prep, BMX, DBT, 30 * MS), b - previous)


IMB = {"kind": "imbalance", "venue": BMX, "levels": 2}
MALFORMED_SPECS: list[object] = [
    [("kind", "imbalance")],
    {"kind": "momentum", "venue": BMX},
    {"venue": BMX, "levels": 3},
    {"kind": "imbalance", "venue": BMX},
    {"kind": "imbalance", "venue": BMX, "levels": 3, "extra": 1},
    {"kind": "imbalance", "venue": BMX, "levels": 3, 7: 1},
    {"kind": "imbalance", "venue": BMX, "levels": 0},
    {"kind": "imbalance", "venue": BMX, "levels": DEPTH + 1},
    {"kind": "imbalance", "venue": BMX, "levels": True},
    {"kind": "imbalance", "venue": BMX, "levels": 2.0},
    {"kind": "imbalance", "venue": "", "levels": 2},
    {"kind": "imbalance", "venue": "venue_a", "levels": 2},
    {"kind": "imbalance", "venue": 0, "levels": 2},
    {"kind": "ofi", "venue": BMX, "tau_ms": 0},
    {"kind": "ofi", "venue": BMX, "tau_ms": -5},
    {"kind": "ofi", "venue": BMX, "tau_ms": NAN},
    {"kind": "ofi", "venue": BMX, "tau_ms": math.inf},
    {"kind": "ofi", "venue": BMX, "tau_ms": True},
    {"kind": "ofi", "venue": BMX, "tau_ms": "5"},
    {"kind": "ofi", "venue": BMX, "tau_ms": 1e-9},  # sub-nanosecond
    {"kind": "ofi", "venue": BMX, "tau_ms": 10**12},  # beyond 2**53 ns
    {"kind": "trade_flow", "venue": BMX, "tau_ms": 5, "normalize": 1},
    {"kind": "basis_dev", "leader": BMX, "follower": BMX, "tau_ms": 5, "price": "mid"},
    {"kind": "basis_dev", "leader": BMX, "follower": DBT, "tau_ms": 5, "price": "last"},
    {"kind": "basis_dev", "leader": BMX, "follower": DBT, "tau_ms": 5},
    {"kind": "lead_gap", "leader": BMX, "follower": DBT, "window_ms": 0},
    {"kind": "return", "venue": BMX, "window_ms": 10, "price": "mid"},
    {"kind": "linear", "terms": []},
    {"kind": "linear", "terms": "abc"},
    {"kind": "linear", "terms": [IMB]},
    {"kind": "linear", "terms": [{"spec": IMB}]},
    {"kind": "linear", "terms": [{"spec": IMB, "weight": NAN}]},
    {"kind": "linear", "terms": [{"spec": IMB, "weight": False}]},
    {"kind": "linear", "terms": [{"spec": IMB, "weight": 1.0, "note": "x"}]},
    {"kind": "linear", "terms": [{"spec": {"kind": "nope"}, "weight": 1.0}]},
]


@pytest.mark.parametrize("bad", MALFORMED_SPECS, ids=[str(i) for i in range(len(MALFORMED_SPECS))])
def test_malformed_specs_are_rejected(bad: Any) -> None:
    prep = lead_lag_market()
    with pytest.raises(InvalidConfigError):
        build_signal(prep, bad)
    with pytest.raises(InvalidConfigError):
        spec_id(bad)


def test_unknown_instrument_is_rejected_at_build_time() -> None:
    spec = {"kind": "imbalance", "venue": "kraken/ALPHA-USD", "levels": 1}
    assert len(spec_id(spec)) == 16  # structurally valid
    with pytest.raises(InvalidConfigError):
        build_signal(lead_lag_market(), spec)


def test_linear_nesting_is_bounded() -> None:
    spec: dict[str, Any] = dict(IMB)
    for _ in range(3):
        spec = {"kind": "linear", "terms": [{"spec": spec, "weight": 1.0}]}
    prep = lead_lag_market()
    np.testing.assert_array_equal(build_signal(prep, spec), build_signal(prep, IMB))
    for _ in range(40):
        spec = {"kind": "linear", "terms": [{"spec": spec, "weight": 1.0}]}
    with pytest.raises(InvalidConfigError):
        spec_id(spec)


def test_spec_id_is_canonical_versioned_and_stable() -> None:
    a = {"kind": "ofi", "venue": BMX, "tau_ms": 100}
    b = {"tau_ms": 100.0, "venue": BMX, "kind": "ofi"}
    sid = spec_id(a)
    assert sid == spec_id(b)
    assert len(sid) == 16 and int(sid, 16) >= 0
    assert canonical_spec(b) == {"kind": "ofi", "tau_ms": 100, "venue": BMX}
    assert (
        sid
        == sha256_json({"schema": signals.SPEC_SCHEMA, "spec": canonical_spec(a), "version": signals.SPEC_VERSION})[:16]
    )
    assert sid == GOLDEN_OFI_ID  # changing this breaks every cache keyed by spec_id
    assert sid != spec_id({"kind": "ofi", "venue": BMX, "tau_ms": 101})
    assert sid != spec_id({"kind": "ofi", "venue": DBT, "tau_ms": 100})
    assert spec_id({"kind": "ofi", "venue": BMX, "tau_ms": 0.5}) != spec_id({"kind": "ofi", "venue": BMX, "tau_ms": 1})
    lin_a = {"kind": "linear", "terms": [{"spec": a, "weight": 1}]}
    lin_b = {"kind": "linear", "terms": ({"weight": 1.0, "spec": b},)}
    assert spec_id(lin_a) == spec_id(lin_b)
    two = [{"spec": a, "weight": 1.0}, {"spec": IMB, "weight": 2.0}]
    assert spec_id({"kind": "linear", "terms": two}) != spec_id({"kind": "linear", "terms": two[::-1]})


GOLDEN_OFI_ID = "7406cf719742ac0e"
"""SPEC_VERSION 2 (basis_dev measures against earlier receipt cohorts); version 1 was ec7cea6317a1dd75."""


@pytest.mark.skipif(not HAVE_NUMBA, reason="without Numba the kernels already run as Python")
def test_pure_python_fallback_is_identical_to_the_compiled_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    prep = PreparedMarket.build(rich_market(17, n_rows=60, n_trades=30))
    compiled = [build_signal(prep, spec) for spec in ALL_SPECS]
    compiled_top = top_of_book(prep)
    compiled_rows = observed_rows(prep)
    kernels = [name for name, obj in vars(signals).items() if hasattr(obj, "py_func")]
    assert len(kernels) >= 8
    for name in kernels:
        monkeypatch.setattr(signals, name, getattr(signals, name).py_func)
    for label, spec, want in zip(SPEC_IDS, ALL_SPECS, compiled, strict=True):
        assert build_signal(prep, spec).tobytes() == want.tobytes(), label
    assert observed_rows(prep).tobytes() == compiled_rows.tobytes()
    python_top = top_of_book(prep)
    for field in dataclasses.fields(TopOfBook):
        a, b = getattr(python_top, field.name), getattr(compiled_top, field.name)
        assert a.tobytes() == b.tobytes(), field.name
