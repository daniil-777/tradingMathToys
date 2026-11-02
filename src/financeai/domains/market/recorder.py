"""Ingestion of the recorder's depth-10 text format.

Grammar (one event per header line; levels follow snapshot headers)::

    YY-MM-DD HH:MM:SS.ffffff type: <Type> symbol: .. symbol_id: .. receive_time: <ts> [server_time: <us>] ...
    Ask price: <p> volume: <v>      # printed worst -> best (descending price)
    Bid price: <p> volume: <v>      # printed best -> worst (descending price)
    <blank line>                    # terminates a snapshot block

``<Type>`` is ``OrderbookSnapshot`` (``orderbook: bid_size: <n> ask_size: <m>``, the
*declared* full depth; ``min(10, declared)`` levels are printed), ``Trades``
(``trades: [ { Bid|Ask <price> <qty> } ... ]``) or ``OrderbookClear`` (no exchange
clock). Leading ``[INFO ...]`` recorder log lines are counted and skipped; every other
unrecognised line is an error, never silently dropped.

Two independent implementations exist: :func:`parse_recorder_bytes_reference`
(line-oriented Python using ``datetime``/``Decimal``) is the oracle, and the streaming
Numba kernel behind :func:`parse_recorder_bytes` / :func:`read_recorder_file` is the
fast path (roughly 1 GB/s of text). Tests assert they agree exactly.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterable, Iterator
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Final

import numpy as np
from numpy.typing import NDArray

from financeai.core.clock import MISSING_NS, NS_PER_DAY, NS_PER_S, NS_PER_US, days_from_civil
from financeai.core.clock import parse_recorder_datetime_ns as _parse_ts_reference
from financeai.core.errors import InvalidDataError
from financeai.core.jit import njit
from financeai.domains.market.instruments import InstrumentSpec
from financeai.domains.market.tape import (
    BOOK_CLEAR,
    BOOK_SNAPSHOT,
    COUNT_DTYPE,
    DEPTH,
    KIND_DTYPE,
    PRICE_DTYPE,
    SIDE_DTYPE,
    SIZE_DTYPE,
    TIME_DTYPE,
    BookTape,
    MarketTapes,
    ParseReport,
    TradeTape,
)

DEFAULT_CHUNK_BYTES: Final = 64 << 20
_INT32_MAX: Final = 2**31 - 1

# Kernel status codes -> human-readable reasons.
_OK = 0
_ERRORS: Final = {
    1: "unknown event type",
    2: "unrecognised line",
    3: "price level outside a snapshot block",
    4: "more than ten printed levels on one side",
    5: "malformed number",
    6: "price is not on the instrument tick grid",
    7: "size is not on the instrument quantity grid",
    8: "value overflows int32 storage",
    9: "printed level count differs from min(10, declared depth)",
    10: "malformed timestamp",
    11: "malformed trade group",
    12: "output capacity exceeded (internal error)",
    13: "missing server_time on a snapshot or trade event",
}

# ---------------------------------------------------------------------------
# Fast path: Numba kernel over a uint8 buffer of complete records.
# ---------------------------------------------------------------------------


@njit()
def _digit(c: int) -> int:
    # int() keeps arithmetic in int64 both compiled and in the pure-Python fallback, where
    # buffer elements are NumPy uint8 scalars that would otherwise overflow.
    v = int(c)
    return v - 48 if 48 <= v <= 57 else -1


@njit()
def _parse_ts(buf: NDArray[np.uint8], i: int) -> int:
    """Parse ``YY-MM-DD HH:MM:SS.ffffff`` at offset ``i``; return ns or -1."""
    vals = np.empty(7, dtype=np.int64)
    widths = (2, 2, 2, 2, 2, 2, 6)
    seps = (45, 45, 32, 58, 58, 46, 0)  # '-', '-', ' ', ':', ':', '.', end
    pos = i
    for f in range(7):
        acc = 0
        for _ in range(widths[f]):
            d = _digit(buf[pos])
            if d < 0:
                return -1
            acc = acc * 10 + d
            pos += 1
        vals[f] = acc
        if f < 6:
            if buf[pos] != seps[f]:
                return -1
            pos += 1
    yy, mo, dd, hh, mi, ss, us = vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6]
    if mo < 1 or mo > 12 or dd < 1 or dd > 31 or hh > 23 or mi > 59 or ss > 60:
        return -1
    days = days_from_civil(2000 + yy, mo, dd)
    return int(days * NS_PER_DAY + (hh * 3600 + mi * 60 + ss) * NS_PER_S + us * NS_PER_US)


@njit()
def _parse_fixed6(buf: NDArray[np.uint8], i: int, end: int) -> tuple[int, int]:
    """Parse ``<digits>[.<up to 6 digits>]`` at ``i``; return (value * 1e6, next_pos).

    ``next_pos`` is -1 on malformed input.
    """
    pos = i
    whole = 0
    nd = 0
    while pos < end and _digit(buf[pos]) >= 0:
        whole = whole * 10 + _digit(buf[pos])
        pos += 1
        nd += 1
        if nd > 15:
            return 0, -1
    if nd == 0:
        return 0, -1
    frac = 0
    scale = 100_000
    if pos < end and buf[pos] == 46:  # '.'
        pos += 1
        while pos < end and _digit(buf[pos]) >= 0:
            d = _digit(buf[pos])
            if scale == 0:
                if d != 0:
                    return 0, -1  # more than six significant decimals
            else:
                frac += d * scale
                scale //= 10
            pos += 1
    return whole * 1_000_000 + frac, pos


@njit()
def _match(buf: NDArray[np.uint8], i: int, end: int, pat: NDArray[np.uint8]) -> bool:
    if i + pat.shape[0] > end:
        return False
    for k in range(pat.shape[0]):  # noqa: SIM110 - Numba loop avoids Python generator overhead
        if buf[i + k] != pat[k]:
            return False
    return True


@njit()
def _find(buf: NDArray[np.uint8], i: int, end: int, pat: NDArray[np.uint8]) -> int:
    m = pat.shape[0]
    last = end - m
    p = i
    while p <= last:
        if buf[p] == pat[0] and _match(buf, p, end, pat):
            return p
        p += 1
    return -1


@njit()
def _close_snapshot(
    row: int,
    cnt_b: int,
    cnt_a: int,
    decl_b: int,
    decl_a: int,
    tmp_apx: NDArray[np.int64],
    tmp_asz: NDArray[np.int64],
    b_apx: NDArray[np.int32],
    b_asz: NDArray[np.int32],
    b_nbid: NDArray[np.int8],
    b_nask: NDArray[np.int8],
) -> bool:
    """Validate printed depth and store asks best-first. False if the counts disagree."""
    if cnt_b != min(DEPTH, decl_b) or cnt_a != min(DEPTH, decl_a):
        return False
    for k in range(cnt_a):  # asks were printed worst -> best
        b_apx[row, k] = tmp_apx[cnt_a - 1 - k]
        b_asz[row, k] = tmp_asz[cnt_a - 1 - k]
    b_nbid[row] = cnt_b
    b_nask[row] = cnt_a
    return True


@njit()
def _parse_block(
    buf: NDArray[np.uint8],
    tick_micro: int,
    pats: tuple[NDArray[np.uint8], ...],
    b_recv: NDArray[np.int64],
    b_exch: NDArray[np.int64],
    b_kind: NDArray[np.int8],
    b_nbid: NDArray[np.int8],
    b_nask: NDArray[np.int8],
    b_bpx: NDArray[np.int32],
    b_bsz: NDArray[np.int32],
    b_apx: NDArray[np.int32],
    b_asz: NDArray[np.int32],
    t_recv: NDArray[np.int64],
    t_exch: NDArray[np.int64],
    t_px: NDArray[np.int32],
    t_sz: NDArray[np.int32],
    t_side: NDArray[np.int8],
    counts: NDArray[np.int64],
) -> tuple[int, int]:
    """Parse complete records in ``buf``. Returns (status, byte offset of the failure)."""
    (p_ask, p_bid, p_type, p_server, p_bidsize, p_asksize, p_snap, p_trades, p_clear, p_volume, p_brace) = pats
    n = buf.shape[0]
    cap_book = b_recv.shape[0]
    cap_trade = t_recv.shape[0]
    nb = 0
    nt = 0
    skipped = 0
    tlines = 0
    open_snap = False
    row = -1
    decl_b = 0
    decl_a = 0
    cnt_b = 0
    cnt_a = 0
    tmp_apx = np.zeros(DEPTH, dtype=np.int64)
    tmp_asz = np.zeros(DEPTH, dtype=np.int64)
    i = 0
    while i < n:
        j = i
        while j < n and buf[j] != 10:
            j += 1
        end = j
        if end > i and buf[end - 1] == 13:  # tolerate CRLF
            end -= 1
        if end == i:  # blank line closes a snapshot block
            if open_snap:
                if not _close_snapshot(
                    row, cnt_b, cnt_a, decl_b, decl_a, tmp_apx, tmp_asz, b_apx, b_asz, b_nbid, b_nask
                ):
                    return 9, i
                open_snap = False
            i = j + 1
            continue
        is_ask = _match(buf, i, end, p_ask)
        if is_ask or _match(buf, i, end, p_bid):
            if not open_snap:
                return 3, i
            price, pos = _parse_fixed6(buf, i + p_ask.shape[0], end)
            if pos < 0 or not _match(buf, pos, end, p_volume):
                return 5, i
            size, pos2 = _parse_fixed6(buf, pos + p_volume.shape[0], end)
            if pos2 != end:
                return 5, i
            if price % tick_micro != 0:
                return 6, i
            if size % 1_000_000 != 0:
                return 7, i
            ticks = price // tick_micro
            usd = size // 1_000_000
            if ticks > _INT32_MAX or usd > _INT32_MAX:
                return 8, i
            if is_ask:
                if cnt_a >= DEPTH:
                    return 4, i
                tmp_apx[cnt_a] = ticks
                tmp_asz[cnt_a] = usd
                cnt_a += 1
            else:
                if cnt_b >= DEPTH:
                    return 4, i
                b_bpx[row, cnt_b] = ticks
                b_bsz[row, cnt_b] = usd
                cnt_b += 1
            i = j + 1
            continue
        if end - i >= 26 and buf[i + 2] == 45 and buf[i + 25] == 91:  # 'YY-..' + '[' -> log line
            skipped += 1
            i = j + 1
            continue
        if not (end - i >= 31 and buf[i + 2] == 45 and _match(buf, i + 24, end, p_type)):
            return 2, i
        # ---- event header ----
        if open_snap:  # a header also terminates a block that lacked its blank line
            if not _close_snapshot(row, cnt_b, cnt_a, decl_b, decl_a, tmp_apx, tmp_asz, b_apx, b_asz, b_nbid, b_nask):
                return 9, i
            open_snap = False
        recv = _parse_ts(buf, i)
        if recv < 0:
            return 10, i
        tpos = i + 24 + p_type.shape[0]
        sp = _find(buf, tpos, end, p_server)
        exch = MISSING_NS
        if sp >= 0:
            q = sp + p_server.shape[0]
            acc = 0
            nd = 0
            while q < end and _digit(buf[q]) >= 0:
                acc = acc * 10 + _digit(buf[q])
                q += 1
                nd += 1
            if nd == 0:
                return 5, i
            exch = acc * NS_PER_US
        if _match(buf, tpos, end, p_snap):
            if sp < 0:
                return 13, i
            if nb >= cap_book:
                return 12, i
            row = nb
            nb += 1
            b_recv[row] = recv
            b_exch[row] = exch
            b_kind[row] = BOOK_SNAPSHOT
            bp = _find(buf, tpos, end, p_bidsize)
            ap = _find(buf, tpos, end, p_asksize)
            if bp < 0 or ap < 0:
                return 5, i
            v, q = _parse_fixed6(buf, bp + p_bidsize.shape[0], end)
            if q < 0:
                return 5, i
            decl_b = v // 1_000_000
            v, q = _parse_fixed6(buf, ap + p_asksize.shape[0], end)
            if q < 0:
                return 5, i
            decl_a = v // 1_000_000
            cnt_b = 0
            cnt_a = 0
            open_snap = True
        elif _match(buf, tpos, end, p_trades):
            if sp < 0:
                return 13, i
            tlines += 1
            q = _find(buf, tpos, end, p_brace)
            if q < 0:
                return 11, i
            while q >= 0:
                q += p_brace.shape[0]
                if _match(buf, q, end, p_bid[:4]):  # "Bid "
                    side = 1
                elif _match(buf, q, end, p_ask[:4]):  # "Ask "
                    side = -1
                else:
                    return 11, i
                price, q = _parse_fixed6(buf, q + 4, end)
                if q < 0 or q >= end or buf[q] != 32:
                    return 11, i
                size, q = _parse_fixed6(buf, q + 1, end)
                if q < 0:
                    return 11, i
                if price % tick_micro != 0:
                    return 6, i
                if size % 1_000_000 != 0:
                    return 7, i
                ticks = price // tick_micro
                usd = size // 1_000_000
                if ticks > _INT32_MAX or usd > _INT32_MAX:
                    return 8, i
                if nt >= cap_trade:
                    return 12, i
                t_recv[nt] = recv
                t_exch[nt] = exch
                t_px[nt] = ticks
                t_sz[nt] = usd
                t_side[nt] = side
                nt += 1
                q = _find(buf, q, end, p_brace)
        elif _match(buf, tpos, end, p_clear):
            if nb >= cap_book:
                return 12, i
            b_recv[nb] = recv
            b_exch[nb] = exch
            b_kind[nb] = BOOK_CLEAR
            b_nbid[nb] = 0
            b_nask[nb] = 0
            nb += 1
        else:
            return 1, i
        i = j + 1
    if open_snap and not _close_snapshot(
        row, cnt_b, cnt_a, decl_b, decl_a, tmp_apx, tmp_asz, b_apx, b_asz, b_nbid, b_nask
    ):
        return 9, n
    counts[0] = nb
    counts[1] = nt
    counts[2] = skipped
    counts[3] = tlines
    return _OK, -1


def _pattern(text: bytes) -> NDArray[np.uint8]:
    return np.frombuffer(text, dtype=np.uint8).copy()


_PATTERNS: Final = (
    _pattern(b"Ask price: "),
    _pattern(b"Bid price: "),
    _pattern(b" type: "),
    _pattern(b"server_time: "),
    _pattern(b"bid_size: "),
    _pattern(b"ask_size: "),
    _pattern(b"OrderbookSnapshot "),
    _pattern(b"Trades "),
    _pattern(b"OrderbookClear"),
    _pattern(b" volume: "),
    _pattern(b"{ "),
)


class _StreamingParser:
    """Feeds arbitrary byte chunks; parses only complete records (cut after a blank line)."""

    def __init__(self, instrument: InstrumentSpec) -> None:
        self._instrument = instrument
        self._pending = b""
        self._offset = 0  # absolute offset of ``_pending[0]`` in the input stream
        self._book_parts: list[tuple[NDArray[np.generic], ...]] = []
        self._trade_parts: list[tuple[NDArray[np.generic], ...]] = []
        self._skipped = 0
        self._trade_lines = 0
        self._bytes = 0

    def feed(self, data: bytes) -> None:
        if not data:
            return
        buf = self._pending + data if self._pending else data
        cut = buf.rfind(b"\n\n")
        if cut < 0:
            self._pending = buf
            return
        cut += 2
        self._parse(buf[:cut])
        self._offset += cut
        self._pending = buf[cut:]

    def finish(self) -> MarketTapes:
        if self._pending:
            self._parse(self._pending)
            self._offset += len(self._pending)
            self._pending = b""
        return self._assemble()

    def _parse(self, block: bytes) -> None:
        n_book = block.count(b" type: OrderbookSnapshot ") + block.count(b" type: OrderbookClear")
        n_trade = block.count(b"{ ")
        b_recv = np.zeros(n_book, TIME_DTYPE)
        b_exch = np.zeros(n_book, TIME_DTYPE)
        b_kind = np.zeros(n_book, KIND_DTYPE)
        b_nbid = np.zeros(n_book, COUNT_DTYPE)
        b_nask = np.zeros(n_book, COUNT_DTYPE)
        b_bpx = np.zeros((n_book, DEPTH), PRICE_DTYPE)
        b_bsz = np.zeros((n_book, DEPTH), SIZE_DTYPE)
        b_apx = np.zeros((n_book, DEPTH), PRICE_DTYPE)
        b_asz = np.zeros((n_book, DEPTH), SIZE_DTYPE)
        t_recv = np.zeros(n_trade, TIME_DTYPE)
        t_exch = np.zeros(n_trade, TIME_DTYPE)
        t_px = np.zeros(n_trade, PRICE_DTYPE)
        t_sz = np.zeros(n_trade, SIZE_DTYPE)
        t_side = np.zeros(n_trade, SIDE_DTYPE)
        counts = np.zeros(4, np.int64)
        arr = np.frombuffer(block, dtype=np.uint8)
        status, pos = _parse_block(
            arr,
            self._instrument.tick_micro,
            _PATTERNS,
            b_recv,
            b_exch,
            b_kind,
            b_nbid,
            b_nask,
            b_bpx,
            b_bsz,
            b_apx,
            b_asz,
            t_recv,
            t_exch,
            t_px,
            t_sz,
            t_side,
            counts,
        )
        if status != _OK:
            line_end = block.find(b"\n", pos)
            excerpt = block[pos : line_end if line_end >= 0 else pos + 160][:160]
            raise InvalidDataError(
                f"recorder parse error: {_ERRORS.get(int(status), 'unknown')}",
                context={
                    "instrument": self._instrument.key,
                    "byte_offset": self._offset + int(pos),
                    "line": excerpt.decode("ascii", errors="replace"),
                },
            )
        nb, nt = int(counts[0]), int(counts[1])
        self._book_parts.append(
            (
                b_recv[:nb],
                b_exch[:nb],
                b_kind[:nb],
                b_nbid[:nb],
                b_nask[:nb],
                b_bpx[:nb],
                b_bsz[:nb],
                b_apx[:nb],
                b_asz[:nb],
            )
        )
        self._trade_parts.append((t_recv[:nt], t_exch[:nt], t_px[:nt], t_sz[:nt], t_side[:nt]))
        self._skipped += int(counts[2])
        self._trade_lines += int(counts[3])
        self._bytes += len(block)

    def _assemble(self) -> MarketTapes:
        book_cols = _concat_columns(
            self._book_parts,
            (
                TIME_DTYPE,
                TIME_DTYPE,
                KIND_DTYPE,
                COUNT_DTYPE,
                COUNT_DTYPE,
                PRICE_DTYPE,
                SIZE_DTYPE,
                PRICE_DTYPE,
                SIZE_DTYPE,
            ),
            level_cols=(5, 6, 7, 8),
        )
        trade_cols = _concat_columns(
            self._trade_parts, (TIME_DTYPE, TIME_DTYPE, PRICE_DTYPE, SIZE_DTYPE, SIDE_DTYPE), level_cols=()
        )
        book = BookTape(*book_cols)
        trades = TradeTape(*trade_cols)
        kinds = book.kind
        report = ParseReport(
            snapshots=int(np.count_nonzero(kinds == BOOK_SNAPSHOT)),
            clears=int(np.count_nonzero(kinds == BOOK_CLEAR)),
            trade_lines=self._trade_lines,
            trades=len(trades),
            skipped_lines=self._skipped,
            bytes_parsed=self._bytes,
        )
        return MarketTapes(instrument=self._instrument, book=book, trades=trades, report=report)


def _concat_columns(
    parts: list[tuple[NDArray[np.generic], ...]],
    dtypes: tuple[np.dtype[np.generic], ...],
    level_cols: tuple[int, ...],
) -> list[NDArray[Any]]:
    cols: list[NDArray[Any]] = []
    for c, dtype in enumerate(dtypes):
        pieces = [p[c] for p in parts]
        if pieces:
            cols.append(np.ascontiguousarray(np.concatenate(pieces)))
        else:
            shape: tuple[int, ...] = (0, DEPTH) if c in level_cols else (0,)
            cols.append(np.zeros(shape, dtype))
    return cols


def parse_recorder_bytes(
    data: bytes, instrument: InstrumentSpec, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES
) -> MarketTapes:
    """Fast-path parse of in-memory recorder text, streamed in ``chunk_bytes`` pieces."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if not instrument.fast_text_compatible:
        return parse_recorder_bytes_reference(data, instrument)
    parser = _StreamingParser(instrument)
    for start in range(0, len(data), chunk_bytes):
        parser.feed(data[start : start + chunk_bytes])
    return parser.finish()


def _open_stream(path: Path) -> gzip.GzipFile | BinaryIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def iter_file_chunks(path: Path, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> Iterator[bytes]:
    with _open_stream(path) as stream:
        while chunk := stream.read(chunk_bytes):
            yield chunk


def read_recorder_file(
    path: str | Path, instrument: InstrumentSpec, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES
) -> MarketTapes:
    """Stream-decompress and parse a recorder file (``.txt`` or ``.txt.gz``)."""
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    if not instrument.fast_text_compatible:
        with _open_stream(Path(path)) as stream:
            return _parse_reference_lines(stream, instrument)
    parser = _StreamingParser(instrument)
    for chunk in iter_file_chunks(Path(path), chunk_bytes):
        parser.feed(chunk)
    return parser.finish()


# ---------------------------------------------------------------------------
# Oracle: straightforward line-oriented Python implementation.
# ---------------------------------------------------------------------------

_TEN = Decimal(10)


def _reference_number(token: str, *, what: str, line: str) -> Decimal:
    try:
        value = Decimal(token)
    except InvalidOperation as exc:
        raise InvalidDataError(f"malformed {what}", context={"line": line[:160]}) from exc
    if not value.is_finite() or value < 0:
        raise InvalidDataError(f"invalid {what}", context={"line": line[:160]})
    return value


def _reference_ticks(token: str, instrument: InstrumentSpec, line: str) -> int:
    price = _reference_number(token, what="price", line=line)
    ticks = price / instrument.tick_size
    if ticks != ticks.to_integral_value():
        raise InvalidDataError("price is not on the instrument tick grid", context={"line": line[:160]})
    if ticks > _INT32_MAX:
        raise InvalidDataError("value overflows int32 storage", context={"line": line[:160]})
    return int(ticks)


def _reference_units(token: str, instrument: InstrumentSpec, line: str) -> int:
    size = _reference_number(token, what="size", line=line) / instrument.quantity_step
    if size != size.to_integral_value():
        raise InvalidDataError("size is not on the instrument quantity grid", context={"line": line[:160]})
    if size > _INT32_MAX:
        raise InvalidDataError("value overflows int32 storage", context={"line": line[:160]})
    return int(size)


def parse_recorder_bytes_reference(data: bytes, instrument: InstrumentSpec) -> MarketTapes:
    """Exact decimal parser and independent oracle for the optimized integer-unit path."""
    return _parse_reference_lines(BytesIO(data), instrument)


def _parse_reference_lines(lines: Iterable[bytes], instrument: InstrumentSpec) -> MarketTapes:
    book_rows: list[tuple[int, int, int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    trades: list[tuple[int, int, int, int, int]] = []
    skipped = 0
    trade_lines = 0
    current: tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None = None
    declared = (0, 0)

    def close() -> None:
        nonlocal current
        if current is None:
            return
        recv, exch, asks, bids = current
        if len(bids) != min(DEPTH, declared[0]) or len(asks) != min(DEPTH, declared[1]):
            raise InvalidDataError("printed level count differs from min(10, declared depth)")
        book_rows.append((recv, exch, BOOK_SNAPSHOT, list(reversed(asks)), bids))
        current = None

    byte_count = 0
    for raw in lines:
        byte_count += len(raw)
        line = raw.rstrip(b"\r\n").decode("ascii", errors="strict")
        if not line:
            close()
            continue
        if line.startswith(("Ask price: ", "Bid price: ")):
            if current is None:
                raise InvalidDataError("price level outside a snapshot block", context={"line": line})
            fields = line.split()
            if len(fields) != 5 or fields[3] != "volume:":
                raise InvalidDataError("malformed number", context={"line": line})
            level = (_reference_ticks(fields[2], instrument, line), _reference_units(fields[4], instrument, line))
            side_levels = current[2] if line[0] == "A" else current[3]
            if len(side_levels) >= DEPTH:
                raise InvalidDataError("more than ten printed levels on one side")
            side_levels.append(level)
            continue
        if len(line) > 25 and line[25] == "[":
            skipped += 1
            continue
        if len(line) < 31 or line[24:31] != " type: ":
            raise InvalidDataError("unrecognised line", context={"line": line[:160]})
        close()
        recv = _parse_ts_reference(line[:24].encode())
        event_type = line[31:].split(" ", 1)[0]
        exch = MISSING_NS
        if "server_time: " in line:
            exch = int(line.split("server_time: ", 1)[1].split(" ", 1)[0]) * NS_PER_US
        if event_type == "OrderbookSnapshot":
            if exch == MISSING_NS:
                raise InvalidDataError("missing server_time on a snapshot or trade event")
            n_bid = int(line.split("bid_size: ", 1)[1].split(" ", 1)[0])
            n_ask = int(line.split("ask_size: ", 1)[1].split(" ", 1)[0])
            declared = (n_bid, n_ask)
            current = (recv, exch, [], [])
        elif event_type == "Trades":
            if exch == MISSING_NS:
                raise InvalidDataError("missing server_time on a snapshot or trade event")
            trade_lines += 1
            body = line.split("trades: [", 1)[1]
            for group in body.split("{")[1:]:
                tokens = group.split("}")[0].split()
                if len(tokens) != 3 or tokens[0] not in ("Bid", "Ask"):
                    raise InvalidDataError("malformed trade group", context={"line": line[:160]})
                side = 1 if tokens[0] == "Bid" else -1
                trades.append(
                    (
                        recv,
                        exch,
                        _reference_ticks(tokens[1], instrument, line),
                        _reference_units(tokens[2], instrument, line),
                        side,
                    )
                )
        elif event_type == "OrderbookClear":
            book_rows.append((recv, exch, BOOK_CLEAR, [], []))
        else:
            raise InvalidDataError("unknown event type", context={"type": event_type})
    close()

    n = len(book_rows)
    bid_px = np.zeros((n, DEPTH), PRICE_DTYPE)
    bid_sz = np.zeros((n, DEPTH), SIZE_DTYPE)
    ask_px = np.zeros((n, DEPTH), PRICE_DTYPE)
    ask_sz = np.zeros((n, DEPTH), SIZE_DTYPE)
    for r, (_, _, _, asks, bids) in enumerate(book_rows):
        for k, (px, sz) in enumerate(bids):
            bid_px[r, k], bid_sz[r, k] = px, sz
        for k, (px, sz) in enumerate(asks):
            ask_px[r, k], ask_sz[r, k] = px, sz
    book = BookTape(
        recv_ns=np.array([r[0] for r in book_rows], TIME_DTYPE),
        exch_ns=np.array([r[1] for r in book_rows], TIME_DTYPE),
        kind=np.array([r[2] for r in book_rows], KIND_DTYPE),
        n_bid=np.array([len(r[4]) for r in book_rows], COUNT_DTYPE),
        n_ask=np.array([len(r[3]) for r in book_rows], COUNT_DTYPE),
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
    )
    trade_tape = TradeTape(
        recv_ns=np.array([t[0] for t in trades], TIME_DTYPE),
        exch_ns=np.array([t[1] for t in trades], TIME_DTYPE),
        px=np.array([t[2] for t in trades], PRICE_DTYPE),
        sz=np.array([t[3] for t in trades], SIZE_DTYPE),
        side_label=np.array([t[4] for t in trades], SIDE_DTYPE),
    )
    report = ParseReport(
        snapshots=sum(1 for r in book_rows if r[2] == BOOK_SNAPSHOT),
        clears=sum(1 for r in book_rows if r[2] == BOOK_CLEAR),
        trade_lines=trade_lines,
        trades=len(trades),
        skipped_lines=skipped,
        bytes_parsed=byte_count,
    )
    return MarketTapes(instrument=instrument, book=book, trades=trade_tape, report=report)
