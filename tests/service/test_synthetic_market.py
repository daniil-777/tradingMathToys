"""Library synthetic market generator: determinism, validation and parity with the test builder."""

from __future__ import annotations

import numpy as np
import pytest

from financeai.core.errors import InvalidConfigError
from financeai.domains.market import synthetic
from financeai.domains.market.instruments import DEMO_INVERSE_A, DEMO_INVERSE_B
from financeai.domains.market.stream import MarketData
from financeai.domains.market.synthetic import (
    DEFAULT_VENUES,
    GENERATOR,
    MAX_ROWS_PER_VENUE,
    MAX_SEED,
    SyntheticMarketSpec,
    max_span_ns,
    synthetic_market,
    synthetic_source,
    synthetic_tapes,
)
from financeai.domains.market.tape import DEPTH

from ..market.synthetic import random_market

_BOOK = ("recv_ns", "exch_ns", "kind", "n_bid", "n_ask", "bid_px", "bid_sz", "ask_px", "ask_sz")
_TRADES = ("recv_ns", "exch_ns", "px", "sz", "side_label")


def _assert_same_market(a: MarketData, b: MarketData) -> None:
    assert [v.instrument for v in a.venues] == [v.instrument for v in b.venues]
    for va, vb in zip(a.venues, b.venues, strict=True):
        for name in _BOOK:
            x, y = getattr(va.book, name), getattr(vb.book, name)
            assert x.dtype == y.dtype, name
            np.testing.assert_array_equal(x, y, err_msg=name)
        for name in _TRADES:
            np.testing.assert_array_equal(getattr(va.trades, name), getattr(vb.trades, name), err_msg=name)
        assert va.report == vb.report


@pytest.mark.parametrize("seed", [0, 5, 123])
def test_default_spec_reproduces_the_test_suite_builder_exactly(seed: int) -> None:
    _assert_same_market(synthetic_market(SyntheticMarketSpec(seed=seed)), random_market(seed))


def test_generation_is_deterministic_and_independent_of_requested_venue_order() -> None:
    forward = SyntheticMarketSpec(seed=9, venues=(DEMO_INVERSE_A.key, DEMO_INVERSE_B.key))
    backward = SyntheticMarketSpec(seed=9, venues=(DEMO_INVERSE_B.key, DEMO_INVERSE_A.key))
    assert forward == backward
    assert forward.venues == tuple(sorted(DEFAULT_VENUES))
    a, b = synthetic_tapes(forward), synthetic_tapes(backward)
    for key in DEFAULT_VENUES:
        np.testing.assert_array_equal(a[key].book.bid_px, b[key].book.bid_px)
    ordered = synthetic_market(forward, order=(DEMO_INVERSE_B.key, DEMO_INVERSE_A.key))
    assert [v.instrument.key for v in ordered.venues] == [DEMO_INVERSE_B.key, DEMO_INVERSE_A.key]
    assert not np.array_equal(
        synthetic_tapes(SyntheticMarketSpec(seed=10))[DEMO_INVERSE_A.key].book.bid_px, a[DEMO_INVERSE_A.key].book.bid_px
    )


def test_books_are_valid_inverse_perpetual_snapshots() -> None:
    spec = SyntheticMarketSpec(seed=3, rows_per_venue=500, start_ns=1_700_000_000_000_000_000)
    for key, tapes in synthetic_tapes(spec).items():
        book = tapes.book
        assert len(book) == 500 and tapes.instrument.key == key
        assert np.all(np.diff(book.recv_ns) > 0) and np.all(np.diff(book.exch_ns) >= 0)
        assert np.all(book.exch_ns < book.recv_ns)
        assert int(book.recv_ns[0]) > spec.start_ns
        assert np.all(book.bid_px[:, 0] < book.ask_px[:, 0])
        assert np.all(np.diff(book.bid_px, axis=1) == -1) and np.all(np.diff(book.ask_px, axis=1) == 1)
        assert np.all(book.bid_sz % tapes.instrument.lot_size == 0) and np.all(book.bid_sz > 0)
        assert np.all(book.n_bid == DEPTH) and np.all(book.n_ask == DEPTH)
        assert len(tapes.trades) == 0
        assert not book.bid_px.flags.writeable


def test_seed_range_is_json_safe() -> None:
    assert MAX_SEED == 2**53 - 1
    assert SyntheticMarketSpec(seed=MAX_SEED).seed == MAX_SEED


def test_large_start_clock_is_exact_int64() -> None:
    start = 4_000_000_000_000_000_123  # > 2**53, <= the 2**62 start cap
    tapes = synthetic_tapes(
        SyntheticMarketSpec(seed=1, venues=(DEMO_INVERSE_A.key,), rows_per_venue=10, start_ns=start)
    )
    first = int(tapes[DEMO_INVERSE_A.key].book.recv_ns[0])
    base = synthetic_tapes(SyntheticMarketSpec(seed=1, venues=(DEMO_INVERSE_A.key,), rows_per_venue=10))
    assert first == start + int(base[DEMO_INVERSE_A.key].book.recv_ns[0])


def test_start_clock_cap_leaves_int64_headroom_for_derived_replay_clocks() -> None:
    cap = synthetic.MAX_START_NS
    assert cap == 2**62  # 2116-02-20 UTC: a realistic epoch bound, part of the API contract
    spec = SyntheticMarketSpec(seed=1, rows_per_venue=MAX_ROWS_PER_VENUE, start_ns=cap)
    last = max(int(t.book.recv_ns[-1]) for t in synthetic_tapes(spec).values())
    assert cap < last <= cap + max_span_ns(MAX_ROWS_PER_VENUE)
    assert 2**63 - 1 - last > 2**61  # > 73 years for latency, acknowledgement, consumption TTL and cooldown
    for start in (cap + 1, 2**63 - 1 - max_span_ns(10)):  # the latter was accepted and wrapped arrival clocks
        with pytest.raises(InvalidConfigError):
            SyntheticMarketSpec(seed=1, rows_per_venue=10, start_ns=start)


def test_mapping_round_trip_and_descriptor() -> None:
    spec = SyntheticMarketSpec(seed=11, venues=(DEMO_INVERSE_B.key,), rows_per_venue=40, start_ns=7)
    doc = spec.to_mapping()
    assert doc == {"seed": 11, "venues": [DEMO_INVERSE_B.key], "rows_per_venue": 40, "start_ns": 7}
    assert SyntheticMarketSpec.from_mapping(doc) == spec
    assert spec.descriptor() == {"generator": dict(GENERATOR), **doc}
    source = synthetic_source(spec, DEMO_INVERSE_B.key)
    assert source.name == "synthetic:venue_b/ALPHA-PERP" and len(source.sha256) == 64 and source.bytes == 0
    assert (
        synthetic_source(
            SyntheticMarketSpec(seed=12, venues=(DEMO_INVERSE_B.key,), rows_per_venue=40, start_ns=7),
            DEMO_INVERSE_B.key,
        ).sha256
        != source.sha256
    )
    with pytest.raises(InvalidConfigError):
        SyntheticMarketSpec.from_mapping({**doc, "extra": 1})
    with pytest.raises(InvalidConfigError):
        synthetic_source(spec, DEMO_INVERSE_A.key)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": True},
        {"seed": 1.0},
        {"seed": 2**63},
        {"seed": MAX_SEED + 1},
        {"seed": 1, "venues": ()},
        {"seed": 1, "venues": ("venue_a/ALPHA-USD", "venue_a/ALPHA-USD")},
        {"seed": 1, "venues": ("binance/ASSETUSD",)},
        {"seed": 1, "venues": "venue_a/ALPHA-USD"},
        {"seed": 1, "rows_per_venue": 1},
        {"seed": 1, "rows_per_venue": MAX_ROWS_PER_VENUE + 1},
        {"seed": 1, "rows_per_venue": 10.0},
        {"seed": 1, "start_ns": -1},
        {"seed": 1, "start_ns": "0"},
        {"seed": 1, "start_ns": 2**63 - 1},
        {"seed": 1, "start_ns": 2**62 + 1},
    ],
)
def test_invalid_specs_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(InvalidConfigError):
        SyntheticMarketSpec(**kwargs)  # type: ignore[arg-type]


def test_unknown_order_key_is_rejected() -> None:
    with pytest.raises(InvalidConfigError):
        synthetic_market(SyntheticMarketSpec(seed=1, venues=(DEMO_INVERSE_A.key,)), order=(DEMO_INVERSE_B.key,))
