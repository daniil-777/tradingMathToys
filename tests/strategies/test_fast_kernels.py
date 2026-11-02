"""Fused signal kernels match the reference implementations bit for bit.

Synthetic markets cover invalid book rows, tied clocks, and the pure-Python fallback.
Comparisons require identical dtype, shape, values, and bytes.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from financeai.core.errors import InvalidDataError
from financeai.core.jit import HAVE_NUMBA
from financeai.domains.market import stream as stream_mod
from financeai.domains.market.engine import PreparedMarket
from financeai.domains.market.stream import EventStream, MarketData
from financeai.domains.market.tape import BOOK_CLEAR, DEPTH, BookTape, TradeTape
from financeai.strategies import leadlag, linear, regime

from ..market.synthetic import US, random_market
from .legacy_kernels import Legacy

MS = 1_000_000
COMPOSITE: Mapping[str, Any] = {
    "kind": "any",
    "components": [
        {
            "scale": 1.0,
            "spec": {
                "kind": "sweep",
                "source": "venue_a/ALPHA-USD",
                "min_units": 2_000,
                "gate": {"venue": "venue_a/ALPHA-USD", "tau_ms": 1_000, "min_units": 500},
            },
        },
        {
            "scale": 2.5,
            "spec": {
                "kind": "lead_gap",
                "leader": "venue_b/ALPHA-PERP",
                "follower": "venue_a/ALPHA-USD",
                "window_ms": 20,
                "depth_units": 2_000,
                "gate_ticks": 1,
                "gate": {"venue": "venue_a/ALPHA-USD", "tau_ms": 1_000, "min_units": 500},
                "fresh": {"max_age_ms": 40},
            },
        },
    ],
}
"""The incumbent's recipe shape (gated sweep or gated lead gap) with a freshness gate added."""


# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------


def rich_market(seed: int, n_rows: int = 150, n_trades: int = 80) -> MarketData:
    """``random_market`` plus clears, one-sided, crossed and locked rows, duplicate receipt
    times and trade prints (a third share a book row's receipt time, some repeat the previous
    print's). The locked rows (best bid == best ask) sit exactly on the ``bid < ask`` boundary
    of ``book_validity`` that the fused mid kernel evaluates inline."""
    base = random_market(seed, n_rows)
    rng = np.random.default_rng(seed + 7_919)
    venues = []
    for tapes in base.venues:
        b = tapes.book
        kind, n_bid, n_ask = b.kind.copy(), b.n_bid.copy(), b.n_ask.copy()
        bid_px, bid_sz, ask_px, ask_sz = b.bid_px.copy(), b.bid_sz.copy(), b.ask_px.copy(), b.ask_sz.copy()
        recv, exch = b.recv_ns.copy(), b.exch_ns.copy()
        for r in rng.choice(n_rows, size=n_rows // 8, replace=False).tolist():
            mode = int(rng.integers(0, 5))
            if mode == 0:
                kind[r], n_bid[r], n_ask[r] = BOOK_CLEAR, 0, 0
                bid_px[r], bid_sz[r], ask_px[r], ask_sz[r] = 0, 0, 0, 0
            elif mode == 1:
                n_ask[r] = 0
                ask_px[r], ask_sz[r] = 0, 0
            elif mode == 2:
                ask_px[r] = bid_px[r, 0] - 1 + np.arange(DEPTH)
            elif mode == 3:
                ask_px[r] = bid_px[r, 0] + np.arange(DEPTH)
            elif r > 0:  # a second row received at the same instant as the previous one
                recv[r], exch[r] = recv[r - 1], exch[r - 1]
        book = BookTape(recv, exch, kind, n_bid, n_ask, bid_px, bid_sz, ask_px, ask_sz)
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
            if rng.random() < 0.2:
                t[i] = t[i - 1]
        trades = TradeTape(
            recv_ns=t.astype(np.int64),
            exch_ns=(t - rng.integers(100 * US, 5 * MS, n_trades)),
            px=(40_000 + rng.integers(-40, 40, n_trades)).astype(np.int32),
            sz=(rng.integers(1, 60, n_trades) * tapes.instrument.lot_size).astype(np.int32),
            side_label=rng.choice(np.array([-1, 1], np.int8), n_trades),
        )
        venues.append(dataclasses.replace(tapes, book=book, trades=trades))
    return MarketData(tuple(venues))


def tied_market(seed: int, n_rows: int = 120) -> MarketData:
    """Both venues' books and trades draw receipt times from the same coarse grid, so most
    events tie across venues, kinds and rows (the tie-break order is what is under test)."""
    base = random_market(seed, n_rows)
    rng = np.random.default_rng(seed + 31)
    venues = []
    for tapes in base.venues:
        grid = np.cumsum(rng.integers(0, 3, size=n_rows)) * MS + MS
        book = dataclasses.replace(tapes.book, recv_ns=grid.astype(np.int64), exch_ns=(grid - 500 * US))
        n_tr = n_rows // 2
        t = np.sort(rng.choice(grid, size=n_tr)).astype(np.int64)
        trades = TradeTape(
            recv_ns=t,
            exch_ns=t - 300 * US,
            px=np.full(n_tr, 40_000, np.int32),
            sz=np.full(n_tr, 1_000, np.int32),
            side_label=np.ones(n_tr, np.int8),
        )
        venues.append(dataclasses.replace(tapes, book=book, trades=trades))
    return MarketData(tuple(venues))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def same(new: NDArray[Any], old: NDArray[Any], *, context: str = "") -> None:
    assert new.dtype == old.dtype and new.shape == old.shape, context
    assert np.array_equal(new, old, equal_nan=new.dtype.kind == "f"), context
    assert new.tobytes() == old.tobytes(), context


def same_stream(new: EventStream, old: EventStream) -> None:
    for name in ("recv_ns", "venue", "kind", "row"):
        a, b = getattr(new, name), getattr(old, name)
        same(a, b)
        assert a.flags.c_contiguous and not a.flags.writeable


def every_kernel_output(prep: PreparedMarket, depth_units: int, window_ns: int) -> dict[str, NDArray[Any]]:
    """Every array the fast paths produce for one market (the pure-Python parity fixture)."""
    s = prep.stream
    out: dict[str, NDArray[Any]] = {}
    for v, tapes in enumerate(prep.data.venues):
        mid = leadlag.depth_weighted_mid(tapes, depth_units)
        x = leadlag.event_values(s, v, mid)
        out[f"mid{v}"] = mid
        out[f"obs{v}"] = leadlag.observed_rows(s, v)
        out[f"x{v}"] = x
        out[f"chg{v}"] = leadlag.window_change_bps(x, s.recv_ns, window_ns)
        out[f"lag{v}"] = leadlag.delayed(x, s.recv_ns, window_ns // 2)
        out[f"fresh{v}"] = regime.freshness_mask(prep, venues=(v,), max_age_ns=window_ns)
        out[f"vol{v}"] = regime.decayed_volume(prep, venue=v, tau_ns=window_ns)
        out[f"flow{v}"] = linear.decayed_trade_flow(prep, venue=v, tau_ns=window_ns)
    out["ewma"] = leadlag.ewma_deviation_signal(prep, leader=1, follower=0, tau_ns=window_ns, depth_units=depth_units)
    out["norm"] = regime.normalise(out["chg0"], prep, window_ns)
    out["composite"] = leadlag.build_signal(prep, COMPOSITE)
    return out


def compare_with_legacy(data: MarketData, *, depths: tuple[int, ...], windows_ns: tuple[int, ...]) -> None:
    prep = PreparedMarket.build(data)
    same_stream(prep.stream, Legacy.build_stream(data))
    s = prep.stream
    for v, tapes in enumerate(data.venues):
        same(stream_mod.book_validity(tapes), Legacy.book_validity(tapes))
        same(leadlag.observed_rows(s, v), Legacy.observed_rows(s, v))
        for depth in depths:
            mid = leadlag.depth_weighted_mid(tapes, depth)
            same(mid, Legacy.depth_weighted_mid(tapes, depth))
            x = leadlag.event_values(s, v, mid)
            same(x, Legacy.event_values(s, v, mid))
            for w in windows_ns:
                same(leadlag.window_change_bps(x, s.recv_ns, w), Legacy.window_change_bps(x, s.recv_ns, w))
                same(leadlag.delayed(x, s.recv_ns, w), Legacy.delayed(x, s.recv_ns, w))
        for w in windows_ns:
            same(
                regime.freshness_mask(prep, venues=(v,), max_age_ns=w),
                Legacy.freshness_mask(prep, venues=(v,), max_age_ns=w),
            )
    all_venues = tuple(range(len(data)))
    same(
        regime.freshness_mask(prep, venues=all_venues, max_age_ns=windows_ns[0]),
        Legacy.freshness_mask(prep, venues=all_venues, max_age_ns=windows_ns[0]),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fast_paths_equal_legacy_on_random_markets(seed: int) -> None:
    compare_with_legacy(random_market(seed), depths=(1_000, 50_000), windows_ns=(0, 20 * MS, 10**12))


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_fast_paths_equal_legacy_on_markets_with_unusable_rows_and_trades(seed: int) -> None:
    compare_with_legacy(rich_market(seed), depths=(500, 5_000, 10**7), windows_ns=(1, 7 * MS, 300 * MS))


@pytest.mark.parametrize("seed", [7, 8])
def test_fast_paths_equal_legacy_on_heavily_tied_markets(seed: int) -> None:
    compare_with_legacy(tied_market(seed), depths=(1_000,), windows_ns=(0, MS, 3 * MS))


def test_window_change_and_delay_equal_legacy_on_awkward_values() -> None:
    """Zeros, infinities, NaN runs and negative values example every IEEE corner of the
    fused change kernel (0/0, x/0, inf/inf) and leading-NaN handling of the delay."""
    rng = np.random.default_rng(11)
    n = 5_000
    recv = np.cumsum(rng.integers(0, 4, size=n)).astype(np.int64) * MS
    values = rng.normal(size=n)
    values[rng.random(n) < 0.1] = 0.0
    values[rng.random(n) < 0.05] = np.nan
    values[rng.random(n) < 0.02] = np.inf
    values[rng.random(n) < 0.02] = -np.inf
    values[:7] = np.nan
    for w in (0, 1, MS, 5 * MS, 10**15):
        same(leadlag.window_change_bps(values, recv, w), Legacy.window_change_bps(values, recv, w))
        same(leadlag.delayed(values, recv, w), Legacy.delayed(values, recv, w))
    assert leadlag.delayed(values, recv, 0) is values


def test_depth_weighted_mid_treats_locked_and_zero_priced_books_like_legacy() -> None:
    """Deterministic boundary rows: a locked book (best bid == best ask) is unusable, a crossed
    one too, a zero best bid against a positive ask is usable (``0 < ask``), and a usable
    row thinner than the depth is NaN. ``rich_market`` draws these at random; this pins them."""
    tapes = random_market(19, n_rows=6).venues[0]
    b = tapes.book
    bid_px, ask_px, ask_sz = b.bid_px.copy(), b.ask_px.copy(), b.ask_sz.copy()
    ask_px[1] = bid_px[1, 0] + np.arange(DEPTH)  # locked
    ask_px[2] = bid_px[2, 0] - 1 + np.arange(DEPTH)  # crossed
    bid_px[3] = 0  # zero best bid (usable: 0 < ask)
    ask_sz[4] = 1  # usable but too thin on the ask side
    book = dataclasses.replace(b, bid_px=bid_px, ask_px=ask_px, ask_sz=ask_sz)
    tapes = dataclasses.replace(tapes, book=book)
    for depth in (1, tapes.instrument.lot_size, 5_000):
        new, old = leadlag.depth_weighted_mid(tapes, depth), Legacy.depth_weighted_mid(tapes, depth)
        same(new, old)
        assert np.isnan(new[1]) and np.isnan(new[2]) and np.isfinite(new[0])
    assert np.isfinite(leadlag.depth_weighted_mid(tapes, 1)[3])
    assert np.isnan(leadlag.depth_weighted_mid(tapes, 5_000)[4])


def test_composite_signal_equals_the_legacy_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end: the incumbent-shaped recipe built with the fast kernels equals the same
    recipe built from the replaced implementations (stream, kernels and gates)."""
    data = rich_market(9, n_rows=200, n_trades=120)
    new = leadlag.build_signal(PreparedMarket.build(data), COMPOSITE)
    monkeypatch.setattr(EventStream, "build", staticmethod(Legacy.build_stream))
    for name in ("depth_weighted_mid", "observed_rows", "event_values", "window_change_bps", "delayed"):
        monkeypatch.setattr(leadlag, name, getattr(Legacy, name))
    monkeypatch.setattr(regime, "freshness_mask", Legacy.freshness_mask)
    old = leadlag.build_signal(PreparedMarket.build(data), COMPOSITE)
    same(new, old)
    assert np.count_nonzero(new) > 0


def test_stream_merge_kernel_equals_the_numpy_merge() -> None:
    """The compiled k-way merge and the stable-argsort fallback are two implementations of
    one unique order (``(recv, venue, kind, row)`` never repeats), so they must agree exactly."""
    for data in (tied_market(12), rich_market(13), random_market(14), MarketData(random_market(15).venues[:1])):
        clocks, offsets = stream_mod._clock_segments(data)
        n = int(offsets[-1])
        got: list[tuple[NDArray[Any], ...]] = [
            (np.empty(n, np.int64), np.empty(n, np.int8), np.empty(n, np.int8), np.empty(n, np.int64)) for _ in range(3)
        ]
        kernel: Any = stream_mod._merge_clocks
        kernel(clocks, offsets, *got[0])
        (kernel.py_func if HAVE_NUMBA else kernel)(clocks, offsets, *got[1])
        stream_mod._merge_clocks_numpy(clocks, offsets, *got[2])
        for arrays in got[1:]:
            for a, b in zip(arrays, got[0], strict=True):
                same(a, b)


def test_event_stream_of_an_empty_market_is_empty() -> None:
    tapes = random_market(16, n_rows=3).venues[0]
    empty_book = BookTape(*(getattr(tapes.book, f.name)[:0] for f in dataclasses.fields(BookTape)))
    data = MarketData((dataclasses.replace(tapes, book=empty_book),))
    stream = EventStream.build(data)
    assert len(stream) == 0
    same_stream(stream, Legacy.build_stream(data))


def test_event_values_rejects_row_values_that_do_not_cover_the_tape() -> None:
    prep = PreparedMarket.build(random_market(17, n_rows=20))
    short = np.zeros(len(prep.data.venues[0].book) - 1)
    with pytest.raises(IndexError):
        leadlag.event_values(prep.stream, 0, short)
    with pytest.raises(IndexError):
        Legacy.event_values(prep.stream, 0, short)


def test_event_values_rejects_row_values_that_are_not_one_dimensional() -> None:
    prep = PreparedMarket.build(random_market(17, n_rows=20))
    mid = leadlag.depth_weighted_mid(prep.data.venues[0], 1_000)
    with pytest.raises(InvalidDataError, match="one value per book row"):
        leadlag.event_values(prep.stream, 0, np.stack([mid, mid], axis=1))
    with pytest.raises(InvalidDataError, match="one value per book row"):
        leadlag.event_values(prep.stream, 0, np.float64(1.0))  # type: ignore[arg-type]


def test_freshness_mask_with_fractional_or_negative_age_equals_legacy() -> None:
    """Out-of-contract ages (the contract is an int): ``age <= x`` for integer ages is
    ``age <= floor(x)``, so a negative fraction never counts as fresh and ``2.5 ms`` admits
    ages up to 2 ms, exactly as the comparison with a float did."""
    prep = PreparedMarket.build(rich_market(20, n_rows=60, n_trades=30))
    for age in (-0.5, -1, 0, 0.5, 2.5 * MS, 7 * MS + 0.25, 2.0**70, -(2.0**70)):
        same(
            regime.freshness_mask(prep, venues=(0, 1), max_age_ns=age),  # type: ignore[arg-type]
            Legacy.freshness_mask(prep, venues=(0, 1), max_age_ns=age),
        )  # type: ignore[arg-type]
    assert not regime.freshness_mask(prep, venues=(0,), max_age_ns=-0.5).any()  # type: ignore[arg-type]
    assert regime.freshness_mask(prep, venues=(0,), max_age_ns=0).any()


def test_window_change_rejects_misaligned_inputs() -> None:
    recv = np.arange(5, dtype=np.int64) * MS
    with pytest.raises(InvalidDataError, match="aligned"):
        leadlag.window_change_bps(np.zeros(4), recv, MS)
    with pytest.raises(InvalidDataError, match="aligned"):
        leadlag.delayed(np.zeros(4), recv, MS)


def _kernels(module: ModuleType) -> list[str]:
    return [name for name, obj in vars(module).items() if hasattr(obj, "py_func")]


@pytest.mark.skipif(not HAVE_NUMBA, reason="without Numba the kernels already run as Python")
def test_pure_python_fallback_matches_the_compiled_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    data = rich_market(18, n_rows=80, n_trades=50)
    compiled = every_kernel_output(PreparedMarket.build(data), 2_000, 20 * MS)
    names = {module: _kernels(module) for module in (leadlag, linear, regime, stream_mod)}
    assert {"_depth_weighted_mid", "_observed_rows", "_event_values", "_window_change_bps", "_lagged_values"} <= set(
        names[leadlag]
    )
    assert "_fresh_mask" in names[regime] and "_merge_clocks" in names[stream_mod]
    for module, kernels in names.items():
        for name in kernels:
            monkeypatch.setattr(module, name, getattr(module, name).py_func)
    python = every_kernel_output(PreparedMarket.build(data), 2_000, 20 * MS)
    assert python.keys() == compiled.keys()
    for key, want in compiled.items():
        same(python[key], want, context=key)
