"""Reference signal-kernel and stream-builder implementations.

They are the oracle for the bit-equality tests in ``test_fast_kernels.py`` and the ``before``
side of ``benchmarks/bench_signals.py``; this module is deliberately free of pytest so the
benchmark can import it without the test machinery. Like the originals, the depth-weighted
mid loop is compiled when Numba is available (``njit`` from :mod:`financeai.core.jit`): a
pure-Python copy would be hundreds of times slower and would turn every before/after
comparison into a measurement of the interpreter rather than of the replaced algorithm.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from financeai.core.jit import njit
from financeai.domains.market.engine import PreparedMarket
from financeai.domains.market.stream import BOOK_EVENT, TRADE_EVENT, EventStream, MarketData
from financeai.domains.market.tape import BOOK_SNAPSHOT, MarketTapes


@njit()
def _depth_weighted_mid(
    bpx: Any, bsz: Any, apx: Any, asz: Any, nb: Any, na: Any, valid: Any, tick: float, depth: int, out: Any
) -> None:
    for i in range(bpx.shape[0]):
        out[i] = np.nan
        if not valid[i]:
            continue
        rem = depth
        acc = 0.0
        for k in range(nb[i]):
            take = min(rem, int(bsz[i, k]))
            acc += take * float(bpx[i, k])
            rem -= take
            if rem <= 0:
                break
        if rem > 0:
            continue
        bid_vwap = acc / depth
        rem = depth
        acc = 0.0
        for k in range(na[i]):
            take = min(rem, int(asz[i, k]))
            acc += take * float(apx[i, k])
            rem -= take
            if rem <= 0:
                break
        if rem > 0:
            continue
        out[i] = 0.5 * (bid_vwap + acc / depth) * tick


class Legacy:
    """The replaced implementations (lexsort stream build, ``maximum.accumulate`` observed
    rows, fancy-indexed event values, ``searchsorted`` window changes and delays, the
    validity-array depth-weighted mid and the gather-based freshness mask)."""

    @staticmethod
    def build_stream(data: MarketData) -> EventStream:
        recv: list[NDArray[np.int64]] = []
        venue: list[NDArray[np.int8]] = []
        kind: list[NDArray[np.int8]] = []
        row: list[NDArray[np.int64]] = []
        for v, tapes in enumerate(data.venues):
            for k, clock in ((BOOK_EVENT, tapes.book.recv_ns), (TRADE_EVENT, tapes.trades.recv_ns)):
                n = clock.shape[0]
                recv.append(np.asarray(clock, dtype=np.int64))
                venue.append(np.full(n, v, np.int8))
                kind.append(np.full(n, k, np.int8))
                row.append(np.arange(n, dtype=np.int64))
        r, v_, k_, w = (np.concatenate(recv), np.concatenate(venue), np.concatenate(kind), np.concatenate(row))
        order = np.lexsort((w, k_, v_, r))
        arrays: list[Any] = [np.ascontiguousarray(a[order]) for a in (r, v_, k_, w)]
        for a in arrays:
            a.setflags(write=False)
        return EventStream(*arrays)

    @staticmethod
    def book_validity(tapes: MarketTapes) -> NDArray[np.bool_]:
        b = tapes.book
        ok = (b.kind == BOOK_SNAPSHOT) & (b.n_bid > 0) & (b.n_ask > 0) & (b.bid_px[:, 0] < b.ask_px[:, 0])
        out = np.ascontiguousarray(ok)
        out.setflags(write=False)
        return out

    _depth_weighted_mid = staticmethod(_depth_weighted_mid)

    @classmethod
    def depth_weighted_mid(cls, tapes: MarketTapes, depth_units: int) -> NDArray[np.float64]:
        b = tapes.book
        out = np.empty(len(b), np.float64)
        cls._depth_weighted_mid(
            b.bid_px,
            b.bid_sz,
            b.ask_px,
            b.ask_sz,
            b.n_bid,
            b.n_ask,
            cls.book_validity(tapes),
            tapes.instrument.tick_float,
            int(depth_units),
            out,
        )
        return out

    @staticmethod
    def observed_rows(stream: EventStream, venue: int) -> NDArray[np.int64]:
        mask = (stream.venue == venue) & (stream.kind == BOOK_EVENT)
        rows = np.where(mask, stream.row, -1)
        return np.maximum.accumulate(rows)

    @classmethod
    def event_values(cls, stream: EventStream, venue: int, row_values: NDArray[np.float64]) -> NDArray[np.float64]:
        obs = cls.observed_rows(stream, venue)
        out = np.full(len(stream), np.nan)
        have = obs >= 0
        out[have] = row_values[obs[have]]
        return out

    @staticmethod
    def window_change_bps(
        values: NDArray[np.float64], recv_ns: NDArray[np.int64], window_ns: int
    ) -> NDArray[np.float64]:
        idx = np.searchsorted(recv_ns, recv_ns - window_ns, side="right") - 1
        past = np.where(idx >= 0, values[np.maximum(idx, 0)], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (values / past - 1.0) * 1e4

    @staticmethod
    def delayed(values: NDArray[np.float64], recv_ns: NDArray[np.int64], delay_ns: int) -> NDArray[np.float64]:
        if delay_ns <= 0:
            return values
        idx = np.searchsorted(recv_ns, recv_ns - delay_ns, side="right") - 1
        return np.where(idx >= 0, values[np.maximum(idx, 0)], np.nan)

    @classmethod
    def freshness_mask(cls, prep: PreparedMarket, *, venues: tuple[int, ...], max_age_ns: int) -> NDArray[np.bool_]:
        recv = prep.stream.recv_ns
        mask = np.ones(len(prep.stream), bool)
        for v in venues:
            obs = cls.observed_rows(prep.stream, v)
            have = obs >= 0
            age = np.full(len(recv), np.iinfo(np.int64).max)
            age[have] = recv[have] - prep.book_recv[v][obs[have]]
            mask &= have & (age <= max_age_ns)
        return mask
