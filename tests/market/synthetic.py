"""Builders for small synthetic markets used by engine tests (invented values)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from financeai.domains.market.instruments import DEMO_INVERSE_A, DEMO_INVERSE_B, InstrumentSpec
from financeai.domains.market.stream import MarketData
from financeai.domains.market.tape import (
    DEPTH,
    BookTape,
    MarketTapes,
    ParseReport,
    TradeTape,
)

US = 1_000

Level = tuple[int, int]
Row = tuple[int, int, Sequence[Level], Sequence[Level]]


def tapes_from_rows(spec: InstrumentSpec, rows: Sequence[Row]) -> MarketTapes:
    """rows: (recv_us, exch_us, bids best-first, asks best-first) with prices in ticks."""
    n = len(rows)
    bid_px = np.zeros((n, DEPTH), np.int32)
    bid_sz = np.zeros((n, DEPTH), np.int32)
    ask_px = np.zeros((n, DEPTH), np.int32)
    ask_sz = np.zeros((n, DEPTH), np.int32)
    for r, (_, _, bids, asks) in enumerate(rows):
        for k, (p, s) in enumerate(bids):
            bid_px[r, k], bid_sz[r, k] = p, s
        for k, (p, s) in enumerate(asks):
            ask_px[r, k], ask_sz[r, k] = p, s
    book = BookTape(
        recv_ns=np.array([r[0] * US for r in rows], np.int64),
        exch_ns=np.array([r[1] * US for r in rows], np.int64),
        kind=np.zeros(n, np.int8),
        n_bid=np.array([len(r[2]) for r in rows], np.int8),
        n_ask=np.array([len(r[3]) for r in rows], np.int8),
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
    )
    trades = TradeTape(
        np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int8)
    )
    report = ParseReport(snapshots=n, clears=0, trade_lines=0, trades=0, skipped_lines=0, bytes_parsed=0)
    return MarketTapes(instrument=spec, book=book, trades=trades, report=report)


def ladder(best_bid: int, best_ask: int, size: int, depth: int = DEPTH) -> tuple[list[Level], list[Level]]:
    bids = [(best_bid - k, size) for k in range(depth)]
    asks = [(best_ask + k, size) for k in range(depth)]
    return bids, asks


def random_market(seed: int, n_rows: int = 300) -> MarketData:
    """Two venues with random-walk mids, 1-2 tick spreads and random feed latency."""
    rng = np.random.default_rng(seed)
    venues = []
    for spec in (DEMO_INVERSE_A, DEMO_INVERSE_B):
        recv = np.cumsum(rng.integers(300, 6_000, size=n_rows)) + 1_000  # microseconds
        latency = rng.integers(200, 9_000, size=n_rows)
        exch = np.maximum.accumulate(recv - latency)
        mid2 = 2 * 40_000 + np.cumsum(rng.integers(-2, 3, size=n_rows))  # in half-ticks
        rows: list[Row] = []
        for i in range(n_rows):
            spread = int(rng.integers(1, 3))
            best_bid = int(mid2[i] // 2)
            sizes = rng.integers(1, 40, size=2 * DEPTH) * spec.lot_size
            bids = [(best_bid - k, int(sizes[k])) for k in range(DEPTH)]
            asks = [(best_bid + spread + k, int(sizes[DEPTH + k])) for k in range(DEPTH)]
            rows.append((int(recv[i]), int(exch[i]), bids, asks))
        venues.append(tapes_from_rows(spec, rows))
    return MarketData(tuple(venues))
