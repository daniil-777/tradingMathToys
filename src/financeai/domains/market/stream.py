"""Multi-venue market data and the merged, receipt-ordered event stream.

The controller observes events strictly in local receipt order. Ties are broken
deterministically by ``(recv_ns, venue, kind, row)``; all events sharing one receipt
timestamp form a *cohort* that is delivered before any decision at that time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.core.jit import HAVE_NUMBA, njit
from financeai.domains.market.instruments import InstrumentSpec
from financeai.domains.market.tape import BOOK_SNAPSHOT, MarketTapes

BOOK_EVENT: Final = 0
TRADE_EVENT: Final = 1

Clocks = tuple[NDArray[np.int64], NDArray[np.int64]]
"""The receipt clocks of every ``(venue, kind)`` tape concatenated in that order, and the
start offset of each segment (one more entry than segments; the last is the event count)."""


def _clock_segments(data: MarketData) -> Clocks:
    clocks = [
        np.asarray(c, dtype=np.int64) for tapes in data.venues for c in (tapes.book.recv_ns, tapes.trades.recv_ns)
    ]
    offsets = np.zeros(len(clocks) + 1, np.int64)
    np.cumsum([c.shape[0] for c in clocks], out=offsets[1:])
    return np.concatenate(clocks), offsets


@njit()
def _merge_clocks(
    cat: NDArray[np.int64],
    offsets: NDArray[np.int64],
    recv: NDArray[np.int64],
    venue: NDArray[np.int8],
    kind: NDArray[np.int8],
    row: NDArray[np.int64],
) -> None:
    """k-way merge of the per-``(venue, kind)`` segments of ``cat`` (each non-decreasing)
    into receipt order. A tie goes to the earliest segment, i.e. the lowest ``(venue, kind)``,
    and a segment emits its rows in tape order: exactly the ``(recv, venue, kind, row)``
    order, which is unique because no two events share all four keys."""
    n_seq = offsets.shape[0] - 1
    pos = offsets[:-1].copy()
    for e in range(recv.shape[0]):
        best = -1
        best_t = 0
        for s in range(n_seq):
            p = pos[s]
            if p < offsets[s + 1]:
                t = cat[p]
                if best < 0 or t < best_t:
                    best = s
                    best_t = t
        recv[e] = best_t
        venue[e] = best // 2
        kind[e] = best % 2
        row[e] = pos[best] - offsets[best]
        pos[best] += 1


def _merge_clocks_numpy(
    cat: NDArray[np.int64],
    offsets: NDArray[np.int64],
    recv: NDArray[np.int64],
    venue: NDArray[np.int8],
    kind: NDArray[np.int8],
    row: NDArray[np.int64],
) -> None:
    """The same order without a compiled kernel: the segments are concatenated in ``(venue,
    kind, row)`` order, so a *stable* sort by receipt time alone is the full lexicographic
    sort (and runs in near-linear time on the already sorted segments)."""
    order = np.argsort(cat, kind="stable")
    lengths = np.diff(offsets)
    seq = np.repeat(np.arange(lengths.shape[0]), lengths)
    recv[:] = cat[order]
    venue[:] = (seq // 2).astype(np.int8)[order]
    kind[:] = (seq % 2).astype(np.int8)[order]
    row[:] = (np.arange(cat.shape[0], dtype=np.int64) - offsets[seq])[order]


@dataclass(frozen=True)
class MarketData:
    """Tapes for distinct instruments (index order defines logical venue IDs 0..n-1).

    Multiple instruments may share an exchange name; identity is the full venue/symbol key."""

    venues: tuple[MarketTapes, ...]

    def __post_init__(self) -> None:
        if not self.venues:
            raise InvalidConfigError("market data needs at least one venue")
        if len(self.venues) > np.iinfo(np.int8).max + 1:
            raise InvalidConfigError("market data supports at most 128 venues (int8 venue IDs)")
        keys = [v.instrument.key for v in self.venues]
        if len(set(keys)) != len(keys):
            raise InvalidConfigError("duplicate instrument in market data", context={"keys": keys})
        for tapes in self.venues:
            for name, clock in (("book", tapes.book.recv_ns), ("trades", tapes.trades.recv_ns)):
                if clock.size and int(clock[0]) < 0:
                    raise InvalidDataError(
                        "receipt clocks must be non-negative",
                        context={"instrument": tapes.instrument.key, "tape": name},
                    )
                if clock.shape[0] > 1 and bool(np.any(clock[1:] < clock[:-1])):
                    raise InvalidDataError(
                        "receipt clock is not monotone", context={"instrument": tapes.instrument.key, "tape": name}
                    )

    @property
    def instruments(self) -> tuple[InstrumentSpec, ...]:
        return tuple(v.instrument for v in self.venues)

    def __len__(self) -> int:
        return len(self.venues)


@dataclass(frozen=True)
class EventStream:
    """Receipt-ordered observations; use ``build`` or validate supplied columns against their tapes."""

    recv_ns: NDArray[np.int64]
    venue: NDArray[np.int8]
    kind: NDArray[np.int8]
    row: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.recv_ns.shape[0])

    def validate_for(self, data: MarketData) -> None:
        """Check and freeze a supplied stream before replay preparation.

        A subset of genuine observations is permitted, but duplicates, retimed observations,
        foreign row indexes, and changes to the deterministic tie order are not. Built-in
        stream construction already guarantees this contract and does not need another scan.
        """
        length = None
        for name, dtype in (("recv_ns", np.int64), ("venue", np.int8), ("kind", np.int8), ("row", np.int64)):
            column = getattr(self, name)
            if not isinstance(column, np.ndarray) or column.ndim != 1 or column.dtype != dtype:
                raise InvalidDataError("event stream column has wrong dtype or dimensions", context={"field": name})
            if length is not None and len(column) != length:
                raise InvalidDataError("event stream columns have different lengths", context={"field": name})
            length = len(column)
        if np.any((self.venue < 0) | (self.venue >= len(data))):
            raise InvalidDataError("event stream references an unknown venue")
        if np.any((self.kind != BOOK_EVENT) & (self.kind != TRADE_EVENT)):
            raise InvalidDataError("event stream contains an unknown event kind")
        if (len(self) and self.recv_ns[0] < 0) or np.any(self.recv_ns[1:] < self.recv_ns[:-1]):
            raise InvalidDataError("event stream receipt clocks must be non-negative and monotone")
        ties = self.recv_ns[1:] == self.recv_ns[:-1]
        if np.any(
            ties
            & (
                (self.venue[1:] < self.venue[:-1])
                | ((self.venue[1:] == self.venue[:-1]) & (self.kind[1:] < self.kind[:-1]))
            )
        ):
            raise InvalidDataError("event stream does not preserve deterministic tie order")
        for venue, tapes in enumerate(data.venues):
            for kind, clocks in ((BOOK_EVENT, tapes.book.recv_ns), (TRADE_EVENT, tapes.trades.recv_ns)):
                mask = (self.venue == venue) & (self.kind == kind)
                rows = self.row[mask]
                if np.any((rows < 0) | (rows >= len(clocks))):
                    raise InvalidDataError(
                        "event stream row is outside its source tape", context={"venue": venue, "kind": kind}
                    )
                if np.any(rows[1:] <= rows[:-1]):
                    raise InvalidDataError("event stream repeats or reorders source rows")
                if not np.array_equal(self.recv_ns[mask], clocks[rows]):
                    raise InvalidDataError("event stream clock differs from its source observation")
        for column in (self.recv_ns, self.venue, self.kind, self.row):
            column.setflags(write=False)

    @classmethod
    def build(cls, data: MarketData) -> EventStream:
        """Merge the per-tape receipt clocks (each monotone, see :class:`MarketData`) into
        ``(recv_ns, venue, kind, row)`` order.

        A linear k-way merge replaces the former ``np.lexsort`` over four 16M-element keys,
        which was most of ``PreparedMarket.build``; without Numba a stable sort by receipt
        time produces the identical (unique) order.
        """
        cat, offsets = _clock_segments(data)
        n = int(offsets[-1])
        recv, row = np.empty(n, np.int64), np.empty(n, np.int64)
        venue, kind = np.empty(n, np.int8), np.empty(n, np.int8)
        merge = _merge_clocks if HAVE_NUMBA else _merge_clocks_numpy
        merge(cat, offsets, recv, venue, kind, row)
        for a in (recv, venue, kind, row):
            a.setflags(write=False)
        return cls(recv, venue, kind, row)


def book_validity(tapes: MarketTapes) -> NDArray[np.bool_]:
    """Rows usable for pricing/execution: snapshot, both sides present, not crossed."""
    b = tapes.book
    ok = (b.kind == BOOK_SNAPSHOT) & (b.n_bid > 0) & (b.n_ask > 0) & (b.bid_px[:, 0] < b.ask_px[:, 0])
    out = np.ascontiguousarray(ok)
    out.setflags(write=False)
    return out


def exchange_envelope(tapes: MarketTapes) -> NDArray[np.int64]:
    """Monotone running maximum of the exchange clock over book rows.

    A book-clear without an exchange clock (``MISSING_NS``) inherits the previous value,
    i.e. it is treated as effective as early as possible (conservative).
    """
    env = np.maximum.accumulate(np.asarray(tapes.book.exch_ns, dtype=np.int64))
    env = np.ascontiguousarray(env)
    env.setflags(write=False)
    return env
