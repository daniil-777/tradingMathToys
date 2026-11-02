"""Causal activity measures and signal gating."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from financeai.domains.market.engine import PreparedMarket
from financeai.domains.market.instruments import DEMO_INVERSE_A, DEMO_INVERSE_B
from financeai.domains.market.stream import MarketData
from financeai.strategies.leadlag import build_signal
from financeai.strategies.regime import decayed_volume

from ..market.synthetic import ladder, random_market, tapes_from_rows

S = 1_000_000_000


def _prep_with_trades() -> PreparedMarket:
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
    return PreparedMarket.build(
        MarketData((dataclasses.replace(venue_a, trades=trades), tapes_from_rows(DEMO_INVERSE_B, rows)))
    )


def test_decayed_volume_counts_both_sides() -> None:
    prep = _prep_with_trades()
    vol = decayed_volume(prep, venue=0, tau_ns=S)
    recv = prep.stream.recv_ns
    assert vol[np.flatnonzero(recv == S)[-1]] == pytest.approx(100 * math.exp(-1.0) + 50)


def test_volume_gate_zeroes_the_signal_in_quiet_periods() -> None:
    prep = PreparedMarket.build(random_market(31))
    base = {"kind": "sweep", "source": "venue_a/ALPHA-USD", "min_units": 1}
    sig = build_signal(prep, base)
    gated = build_signal(prep, {**base, "gate": {"venue": "venue_a/ALPHA-USD", "tau_ms": 1000, "min_units": 10**12}})
    assert np.all(gated == 0.0)
    open_gate = build_signal(prep, {**base, "gate": {"venue": "venue_a/ALPHA-USD", "tau_ms": 1000, "min_units": 0}})
    np.testing.assert_array_equal(open_gate, sig)


def test_decayed_volume_is_causal() -> None:
    data = random_market(32)
    prep = PreparedMarket.build(data)
    k = len(prep.stream) // 2
    cut = int(prep.stream.recv_ns[k])
    tr = data.venues[0].trades
    assert len(tr) == 0 or True  # synthetic markets have no trades; inject some
    times = np.sort(np.random.default_rng(1).choice(prep.stream.recv_ns, 40, replace=False))
    trades = dataclasses.replace(
        tr,
        recv_ns=times.astype(np.int64),
        exch_ns=times.astype(np.int64),
        px=np.full(40, 40_000, np.int32),
        sz=np.full(40, 1_000, np.int32),
        side_label=np.ones(40, np.int8),
    )
    a = PreparedMarket.build(MarketData((dataclasses.replace(data.venues[0], trades=trades), data.venues[1])))
    late = trades.recv_ns > cut
    sz2 = np.where(late, 77_000, trades.sz).astype(np.int32)
    b_v0 = dataclasses.replace(data.venues[0], trades=dataclasses.replace(trades, sz=sz2))
    b = PreparedMarket.build(MarketData((b_v0, data.venues[1])))
    va, vb = decayed_volume(a, venue=0, tau_ns=S), decayed_volume(b, venue=0, tau_ns=S)
    early = a.stream.recv_ns <= cut
    np.testing.assert_array_equal(va[early], vb[early])
