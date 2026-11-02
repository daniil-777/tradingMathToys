"""Strict normalized Tardis snapshot-25/trade CSV ingestion for any instrument grid.

Only the first ten levels are retained, but every snapshot row is preserved.
Receipt and exchange clocks remain separate. Decimal values are parsed exactly,
without floating-point quantization. Normalized CSV has no disconnect markers;
this adapter cannot reconstruct them or the original cross-file message order.
"""

from __future__ import annotations

import gzip
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.core.jit import njit
from financeai.domains.market.instruments import InstrumentSpec
from financeai.domains.market.recorder import _reference_ticks, _reference_units
from financeai.domains.market.tape import BookTape, MarketTapes, ParseReport, TradeTape

_MAX32 = 2**31 - 1
_MAX_US = (2**63 - 1) // 1000


@njit(cache=False)
def _same(buf: NDArray[np.uint8], start: int, end: int, word: NDArray[np.uint8]) -> bool:
    if end - start != len(word):
        return False
    for j in range(len(word)):  # noqa: SIM110 - Numba kernel avoids Python generators
        if buf[start + j] != word[j]:
            return False
    return True


@njit(cache=False)
def _number(buf: NDArray[np.uint8], start: int, end: int, factor: int, limit: int) -> int:
    """Exact nonnegative decimal times 1 or 2; -1 absent, -2 invalid."""
    if start == end:
        return -1
    i, whole, digits = start, 0, 0
    while i < end and 48 <= buf[i] <= 57:
        digit = int(buf[i]) - 48
        if whole > (limit - digit) // 10:
            return -2
        whole = whole * 10 + digit
        i += 1
        digits += 1
    if digits == 0 or whole > limit // factor:
        return -2
    result = whole * factor
    if i < end and buf[i] == 46:
        i += 1
        if i == end:
            return -2
        first = int(buf[i]) - 48
        if factor == 2 and first == 5:
            result += 1
        elif first != 0:
            return -2
        i += 1
        while i < end:
            if buf[i] != 48:
                return -2
            i += 1
    if i != end or result > limit:
        return -2
    return result


@njit(cache=False)
def _parse_block(
    buf: NDArray[np.uint8],
    codes: NDArray[np.int64],
    slots: NDArray[np.int64],
    exchange: NDArray[np.uint8],
    symbol: NDArray[np.uint8],
    width: int,
    rows: int,
) -> tuple[NDArray[np.int64], int, int]:
    out = np.full((rows, width), -1, np.int64)
    row, col, start = 0, 0, 0
    for i in range(len(buf)):
        c = buf[i]
        if c != 44 and c != 10:
            if c == 34 or c == 13:  # quoted fields / CRLF are outside this normalized dialect
                return out, 1, row
            continue
        if row >= rows or col >= len(codes):
            return out, 2, row
        code, slot = codes[col], slots[col]
        if code == 1:  # receipt/exchange integer microseconds
            for j in range(start, i):
                if buf[j] < 48 or buf[j] > 57:
                    return out, 3, row
            value = _number(buf, start, i, 1, _MAX_US)
            if value < 0:
                return out, 3, row
            out[row, slot] = value * 1000
        elif code == 2 or code == 3:
            value = _number(buf, start, i, 2 if code == 2 else 1, _MAX32)
            if value == -2:
                return out, 4, row
            out[row, slot] = value
        elif code == 4:
            if not _same(buf, start, i, exchange):
                return out, 5, row
        elif code == 5:
            if not _same(buf, start, i, symbol):
                return out, 5, row
        elif code == 6:
            if i - start == 3 and buf[start] == 98 and buf[start + 1] == 117 and buf[start + 2] == 121:
                out[row, slot] = 1
            elif (
                i - start == 4
                and buf[start] == 115
                and buf[start + 1] == 101
                and buf[start + 2] == 108
                and buf[start + 3] == 108
            ):
                out[row, slot] = -1
            else:
                return out, 6, row
        if c == 10:
            if col != len(codes) - 1:
                return out, 2, row
            row += 1
            col = 0
        else:
            col += 1
        start = i + 1
    if row != rows or start != len(buf):
        return out, 2, row
    return out, 0, row


def _parse_decimal_block(
    data: bytes, codes: NDArray[np.int64], slots: NDArray[np.int64], spec: InstrumentSpec, width: int, rows: int
) -> tuple[NDArray[np.int64], int, int]:
    """Exact fallback for arbitrary tick and quantity grids; retain the normalized CSV dialect."""
    out = np.full((rows, width), -1, np.int64)
    for row, raw in enumerate(data.splitlines(keepends=True)):
        if b'"' in raw or b"\r" in raw:
            return out, 1, row
        try:
            fields = raw.rstrip(b"\n").decode("ascii").split(",")
        except UnicodeDecodeError:
            return out, 4, row
        if len(fields) != len(codes):
            return out, 2, row
        for token, code, slot in zip(fields, codes, slots, strict=True):
            if code == 1:
                if not token.isdigit() or len(token) > 19 or int(token) > _MAX_US:
                    return out, 3, row
                out[row, slot] = int(token) * 1000
            elif code in (2, 3) and token:
                if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", token) is None:
                    return out, 4, row
                try:
                    out[row, slot] = (
                        _reference_ticks(token, spec, token) if code == 2 else _reference_units(token, spec, token)
                    )
                except InvalidDataError:
                    return out, 4, row
            elif (code == 4 and token != spec.venue) or (code == 5 and token != spec.symbol):
                return out, 5, row
            elif code == 6:
                if token not in ("buy", "sell"):
                    return out, 6, row
                out[row, slot] = 1 if token == "buy" else -1
    return out, 0, rows


def _open(path: Path) -> BinaryIO:
    return cast(BinaryIO, gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb"))


def _read(path: Path, spec: InstrumentSpec, book: bool, chunk_bytes: int) -> tuple[list[NDArray[Any]], int]:
    exchange = np.frombuffer(spec.venue.encode("ascii"), np.uint8)
    symbol = np.frombuffer(spec.symbol.encode("ascii"), np.uint8)
    chunks: list[list[NDArray[Any]]] = [[] for _ in range(8 if book else 5)]
    previous = -1
    byte_count, row_count = 0, 0
    with _open(path) as handle:
        raw_header = handle.readline()
        byte_count += len(raw_header)
        try:
            header = raw_header.decode("ascii").strip("\n").split(",")
        except UnicodeDecodeError as exc:
            raise InvalidDataError("Tardis header is not ASCII") from exc
        if len(set(header)) != len(header):
            raise InvalidDataError("duplicate Tardis header fields")
        required: dict[str, tuple[int, int]] = {
            "exchange": (4, -1),
            "symbol": (5, -1),
            "timestamp": (1, 1),
            "local_timestamp": (1, 0),
        }
        if book:
            for k in range(10):
                for name, code, slot in [("bids", 2, 2 + 4 * k), ("asks", 2, 4 + 4 * k)]:
                    required[f"{name}[{k}].price"] = (code, slot)
                    required[f"{name}[{k}].amount"] = (3, slot + 1)
            # Require the declared source format, even though deeper levels are projected away.
            for k in range(10, 25):
                for side in ("bids", "asks"):
                    required[f"{side}[{k}].price"] = (0, -1)
                    required[f"{side}[{k}].amount"] = (0, -1)
        else:
            required.update({"price": (2, 2), "amount": (3, 3), "side": (6, 4), "id": (0, -1)})
        if not required.keys() <= set(header):
            raise InvalidDataError("Tardis CSV lacks required schema fields", context={"path": str(path)})
        codes = np.array([required.get(h, (0, -1))[0] for h in header], np.int64)
        slots = np.array([required.get(h, (0, -1))[1] for h in header], np.int64)
        carry = b""
        while True:
            block = handle.read(chunk_bytes)
            byte_count += len(block)
            data = carry + block
            if not block:
                if not data:
                    break
                if not data.endswith(b"\n"):
                    data += b"\n"
            last = data.rfind(b"\n")
            if last < 0:
                carry = data
                if len(carry) > 1 << 20:
                    raise InvalidDataError("Tardis row exceeds maximum length")
                continue
            complete, carry = data[: last + 1], data[last + 1 :]
            n = complete.count(b"\n")
            if spec.tick_size == Decimal("0.5") and spec.quantity_step == Decimal(1):
                values, error, bad_row = _parse_block(
                    np.frombuffer(complete, np.uint8),
                    codes,
                    slots,
                    exchange,
                    symbol,
                    42 if book else 5,
                    n,
                )
            else:
                values, error, bad_row = _parse_decimal_block(complete, codes, slots, spec, 42 if book else 5, n)
            if error:
                reasons = {
                    1: "unsupported quoted/CRLF dialect",
                    2: "wrong field count",
                    3: "invalid timestamp",
                    4: "fractional/off-grid/overflowing numeric value",
                    5: "wrong exchange or symbol",
                    6: "unknown aggressor side",
                }
                raise InvalidDataError(reasons[error], context={"path": str(path), "row": row_count + bad_row + 2})
            recv = values[:, 0]
            if len(recv) and (recv[0] < previous or np.any(np.diff(recv) < 0)):
                raise InvalidDataError("Tardis receipt timestamps must be nondecreasing")
            if len(recv):
                previous = int(recv[-1])
            if book:
                bp, bs, ap, az = [values[:, j::4] for j in (2, 3, 4, 5)]
                counts = []
                for prices, sizes, sign in ((bp, bs, 1), (ap, az, -1)):
                    present = prices >= 0
                    if np.any(present != (sizes >= 0)) or np.any(prices[present] <= 0):
                        raise InvalidDataError("book price/amount pair missing or nonpositive price")
                    if np.any(np.diff(present.astype(np.int8), axis=1) > 0):
                        raise InvalidDataError("missing level inside populated book prefix")
                    pairs = present[:, 1:] & present[:, :-1]
                    if np.any((np.diff(prices, axis=1) * sign)[pairs] >= 0):
                        raise InvalidDataError("book levels are not strictly ordered")
                    counts.append(np.sum(present, axis=1).astype(np.int8))
                arrays: list[NDArray[Any]] = [
                    recv.copy(),
                    values[:, 1].copy(),
                    *counts,
                    *[np.maximum(x, 0).astype(np.int32) for x in (bp, bs, ap, az)],
                ]
            else:
                if np.any(values[:, 2:4] <= 0):
                    raise InvalidDataError("trade price and amount must be positive")
                arrays = [
                    recv.copy(),
                    values[:, 1].copy(),
                    values[:, 2].astype(np.int32),
                    values[:, 3].astype(np.int32),
                    values[:, 4].astype(np.int8),
                ]
            for target, arr in zip(chunks, arrays, strict=True):
                target.append(arr)
            row_count += n
            if not block:
                break
    if row_count == 0:
        raise InvalidDataError("empty Tardis dataset")
    return [np.concatenate(c) for c in chunks], byte_count


def read_tardis_snapshot_trades(
    book_path: str | Path,
    trades_path: str | Path,
    instrument: InstrumentSpec,
    *,
    chunk_bytes: int = 8 << 20,
) -> MarketTapes:
    """Read exact decimal CSV into immutable local tapes; no forecasts or fitting.

    Supports normalized LF-delimited, unquoted CSV with instrument-defined tick
    and quantity grids, currencies, and contract types. Unknown trade
    sides, off-grid values, malformed books and decreasing receipt clocks raise
    errors. Crossed top quotes are retained for the existing validity rules.
    Missing trailing levels remain missing. Duplicate receipt cohorts retain
    source row order. Scientific notation is outside the supported dialect.
    """
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise InvalidConfigError("positive integer chunk_bytes required")
    b, bbytes = _read(Path(book_path), instrument, True, chunk_bytes)
    t, tbytes = _read(Path(trades_path), instrument, False, chunk_bytes)
    book = BookTape(b[0], b[1], np.zeros(len(b[0]), np.int8), *b[2:])
    trades = TradeTape(*t)
    return MarketTapes(
        instrument, book, trades, ParseReport(len(book), 0, len(trades), len(trades), 0, bbytes + tbytes)
    )
