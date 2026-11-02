"""Columnar (structure-of-arrays) market tapes.

Each tape owns read-only NumPy arrays. Level arrays have shape ``(n, DEPTH)`` with
index 0 the best level on each side (bids descending, asks ascending); absent levels
are zero-filled and ``n_bid``/``n_ask`` give the populated count. Prices are integer
ticks of the instrument's tick size; sizes are integer stored quantity units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidDataError
from financeai.domains.market.instruments import InstrumentSpec

DEPTH: Final = 10
TIME_DTYPE: Final = np.dtype(np.int64)
PRICE_DTYPE: Final = np.dtype(np.int32)
SIZE_DTYPE: Final = np.dtype(np.int32)
COUNT_DTYPE: Final = np.dtype(np.int8)
KIND_DTYPE: Final = np.dtype(np.int8)
SIDE_DTYPE: Final = np.dtype(np.int8)

BOOK_SNAPSHOT: Final = 0
"""A full top-of-book state (up to DEPTH levels per side)."""
BOOK_CLEAR: Final = 1
"""The recorder cleared its book; no liquidity is known from this event on."""


def _freeze(name: str, arr: NDArray[np.generic], dtype: np.dtype[np.generic], ndim: int) -> None:
    if not isinstance(arr, np.ndarray):
        raise InvalidDataError(f"{name} must be a NumPy array")
    if arr.dtype != dtype:
        raise InvalidDataError(f"{name} must have dtype {dtype}", context={"got": str(arr.dtype)})
    if arr.ndim != ndim:
        raise InvalidDataError(f"{name} must be {ndim}-D", context={"shape": arr.shape})
    arr.setflags(write=False)


@dataclass(frozen=True, slots=True)
class BookTape:
    recv_ns: NDArray[np.int64]
    exch_ns: NDArray[np.int64]
    kind: NDArray[np.int8]
    n_bid: NDArray[np.int8]
    n_ask: NDArray[np.int8]
    bid_px: NDArray[np.int32]
    bid_sz: NDArray[np.int32]
    ask_px: NDArray[np.int32]
    ask_sz: NDArray[np.int32]

    def __post_init__(self) -> None:
        _freeze("recv_ns", self.recv_ns, TIME_DTYPE, 1)
        n = self.recv_ns.shape[0]
        dtype: np.dtype[Any]
        for name, dtype in (
            ("recv_ns", TIME_DTYPE),
            ("exch_ns", TIME_DTYPE),
            ("kind", KIND_DTYPE),
            ("n_bid", COUNT_DTYPE),
            ("n_ask", COUNT_DTYPE),
        ):
            arr = getattr(self, name)
            _freeze(name, arr, dtype, 1)
            if arr.shape[0] != n:
                raise InvalidDataError(f"{name} length mismatch", context={"want": n, "got": arr.shape[0]})
        for name, dtype in (
            ("bid_px", PRICE_DTYPE),
            ("bid_sz", SIZE_DTYPE),
            ("ask_px", PRICE_DTYPE),
            ("ask_sz", SIZE_DTYPE),
        ):
            arr = getattr(self, name)
            _freeze(name, arr, dtype, 2)
            if arr.shape != (n, DEPTH):
                raise InvalidDataError(f"{name} shape mismatch", context={"shape": arr.shape})
        for name, counts in (("n_bid", self.n_bid), ("n_ask", self.n_ask)):
            if np.any((counts < 0) | (counts > DEPTH)):
                raise InvalidDataError(f"{name} must be in [0, {DEPTH}]")
        if np.any((self.kind != BOOK_SNAPSHOT) & (self.kind != BOOK_CLEAR)):
            raise InvalidDataError("unknown book event kind")

    def __len__(self) -> int:
        return int(self.recv_ns.shape[0])


@dataclass(frozen=True, slots=True)
class TradeTape:
    recv_ns: NDArray[np.int64]
    exch_ns: NDArray[np.int64]
    px: NDArray[np.int32]
    sz: NDArray[np.int32]
    side_label: NDArray[np.int8]
    """Raw recorder label: +1 for 'Bid', -1 for 'Ask' (aggressor meaning verified separately)."""

    def __post_init__(self) -> None:
        _freeze("recv_ns", self.recv_ns, TIME_DTYPE, 1)
        n = self.recv_ns.shape[0]
        for name, dtype in (
            ("recv_ns", TIME_DTYPE),
            ("exch_ns", TIME_DTYPE),
            ("px", PRICE_DTYPE),
            ("sz", SIZE_DTYPE),
            ("side_label", SIDE_DTYPE),
        ):
            arr = getattr(self, name)
            _freeze(name, arr, dtype, 1)
            if arr.shape[0] != n:
                raise InvalidDataError(f"{name} length mismatch", context={"want": n, "got": arr.shape[0]})

    def __len__(self) -> int:
        return int(self.recv_ns.shape[0])


@dataclass(frozen=True, slots=True)
class ParseReport:
    snapshots: int
    clears: int
    trade_lines: int
    trades: int
    skipped_lines: int
    """Recorder log lines (``[INFO ...]``); any other unrecognised line is an error."""
    bytes_parsed: int


@dataclass(frozen=True, slots=True)
class MarketTapes:
    instrument: InstrumentSpec
    book: BookTape
    trades: TradeTape
    report: ParseReport
