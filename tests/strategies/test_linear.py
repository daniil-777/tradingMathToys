"""H4 linear model: ridge exactness, hand-computed features, labels, streaming prediction, causality."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from financeai.core.errors import InvalidDataError
from financeai.domains.market.engine import PreparedMarket
from financeai.domains.market.instruments import DEMO_INVERSE_A, DEMO_INVERSE_B
from financeai.domains.market.stream import MarketData
from financeai.models.ridge import RidgeModel
from financeai.strategies.linear import (
    decayed_trade_flow,
    depth_imbalance,
    feature_columns,
    follower_labels,
    prediction_signal,
)

from ..market.synthetic import ladder, random_market, tapes_from_rows

MS = 1_000_000
S = 1_000_000_000


def test_ridge_recovers_coefficients_and_does_not_penalise_the_intercept() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5_000, 3)) * np.array([1.0, 10.0, 0.1]) + np.array([5.0, -2.0, 1.0])
    y = 3.0 + x @ np.array([2.0, -0.5, 7.0])
    model = RidgeModel(alpha=1e-9).fit(x, y)
    np.testing.assert_allclose(model.predict(x), y, atol=1e-6)
    heavy = RidgeModel(alpha=1e12).fit(x, y)
    np.testing.assert_allclose(heavy.predict(x), np.full(len(y), y.mean()), atol=1e-3)


def test_ridge_rejects_non_finite_training_data() -> None:
    with pytest.raises(InvalidDataError):
        RidgeModel(alpha=1.0).fit(np.array([[1.0], [np.nan]]), np.array([1.0, 2.0]))


def _market_with_trades() -> PreparedMarket:
    b, a = ladder(40_000, 40_001, 10_000)
    rows = [(0, 0, b, a), (2_000_000, 1_999_000, b, a)]
    venue_a = tapes_from_rows(DEMO_INVERSE_A, rows)
    trades = dataclasses.replace(
        venue_a.trades,
        recv_ns=np.array([0, 1_000_000], np.int64) * 1_000,
        exch_ns=np.array([0, 999_000], np.int64) * 1_000,
        px=np.array([40_001, 40_000], np.int32),
        sz=np.array([100, 50], np.int32),
        side_label=np.array([1, -1], np.int8),
    )
    venue_b = tapes_from_rows(DEMO_INVERSE_B, rows)
    return PreparedMarket.build(MarketData((dataclasses.replace(venue_a, trades=trades), venue_b)))


def test_decayed_trade_flow_matches_hand_computation() -> None:
    prep = _market_with_trades()
    flow = decayed_trade_flow(prep, venue=0, tau_ns=S)
    recv = prep.stream.recv_ns
    at_1s = flow[np.flatnonzero(recv == S)[-1]]
    assert at_1s == pytest.approx(100 * math.exp(-1.0) - 50)
    at_2s = flow[np.flatnonzero(recv == 2 * S)[-1]]
    assert at_2s == pytest.approx((100 * math.exp(-1.0) - 50) * math.exp(-1.0))


def test_depth_imbalance_over_top_levels() -> None:
    rows = [(0, 0, [(40_000, 300), (39_999, 100)], [(40_001, 100), (40_002, 100)])]
    prep = PreparedMarket.build(MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),)))
    assert depth_imbalance(prep, venue=0, levels=2)[0] == pytest.approx((400 - 200) / 600)


def test_follower_labels_use_the_exchange_clock_book_at_arrival_and_horizon() -> None:
    b0, a0 = ladder(40_000, 40_002, 10_000)  # mid 20000.5
    b1, a1 = ladder(40_010, 40_012, 10_000)  # mid 20005.5
    rows = [(1_000, 500, b0, a0), (30_000, 20_000, b1, a1)]
    prep = PreparedMarket.build(MarketData((tapes_from_rows(DEMO_INVERSE_A, rows),)))
    labels = follower_labels(prep, follower=0, event_idx=np.array([0]), latency_ns=5 * MS, horizon_ns=100 * MS)
    assert labels[0] == pytest.approx((20_005.5 / 20_000.5 - 1) * 1e4)


def test_streaming_prediction_equals_matrix_product() -> None:
    prep = PreparedMarket.build(random_market(9))
    # The ~1 s synthetic market cannot populate multi-second lookbacks; keep columns it can.
    cols = [c for c in feature_columns(prep, follower=0, leader=1) if np.isfinite(c[1]()).mean() > 0.5]
    assert len(cols) >= 8
    x = np.column_stack([fn() for _, fn in cols])
    ok = np.all(np.isfinite(x), axis=1)
    rng = np.random.default_rng(1)
    model = RidgeModel(alpha=1.0).fit(x[ok], rng.normal(size=int(ok.sum())))
    pred = prediction_signal(prep, model, cols, follower=0, gate_ticks=0)
    expected = np.where(ok, model.predict(np.where(ok[:, None], x, 0.0)), 0.0)
    np.testing.assert_allclose(pred, expected, rtol=1e-10, atol=1e-10)


def test_features_never_use_later_events() -> None:
    data = random_market(13)
    prep = PreparedMarket.build(data)
    k = len(prep.stream) // 2
    cut = int(prep.stream.recv_ns[k])
    mutated = []
    for tapes in data.venues:
        book = tapes.book
        shift = np.where(book.recv_ns > cut, 5, 0).astype(np.int32)[:, None]
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
    other = PreparedMarket.build(MarketData(tuple(mutated)))
    early = prep.stream.recv_ns <= cut
    for (name, fa), (_, fb) in zip(feature_columns(prep, 0, 1), feature_columns(other, 0, 1), strict=True):
        np.testing.assert_array_equal(fa()[early], fb()[early], err_msg=name)
