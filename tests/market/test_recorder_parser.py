"""Recorder text ingestion: semantics, strictness, chunk invariance and kernel/reference parity."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from financeai.core.clock import MISSING_NS, parse_recorder_datetime_ns
from financeai.core.errors import InvalidDataError
from financeai.domains.market.instruments import DEMO_INVERSE_A
from financeai.domains.market.recorder import (
    parse_recorder_bytes,
    parse_recorder_bytes_reference,
    read_recorder_file,
)
from financeai.domains.market.tape import (
    BOOK_CLEAR,
    BOOK_SNAPSHOT,
    PRICE_DTYPE,
    SIZE_DTYPE,
    MarketTapes,
)

from .conftest import SYNTHETIC_RECORDER_TEXT

US = 1_000
T0 = 1_675_036_800_000_000_000  # 2024-01-01T00:00:00Z as UTC epoch nanoseconds


def assert_tapes_equal(got: MarketTapes, want: MarketTapes) -> None:
    for name in ("recv_ns", "exch_ns", "n_bid", "n_ask", "bid_px", "bid_sz", "ask_px", "ask_sz"):
        a, b = getattr(got.book, name), getattr(want.book, name)
        assert a.dtype == b.dtype, name
        np.testing.assert_array_equal(a, b, err_msg=f"book.{name}")
    for name in ("recv_ns", "exch_ns", "px", "sz", "side_label"):
        a, b = getattr(got.trades, name), getattr(want.trades, name)
        assert a.dtype == b.dtype, name
        np.testing.assert_array_equal(a, b, err_msg=f"trades.{name}")
    assert got.report.skipped_lines == want.report.skipped_lines


def test_recorder_datetime_is_utc_epoch_nanoseconds() -> None:
    assert parse_recorder_datetime_ns(b"23-01-30 00:00:00.141480") == T0 + 141_480 * US
    assert parse_recorder_datetime_ns(b"23-03-02 14:18:33.372532") == 1_677_766_713_372_532_000


def test_reference_parser_book_semantics(synthetic_recorder_text: bytes) -> None:
    book = parse_recorder_bytes_reference(synthetic_recorder_text, DEMO_INVERSE_A).book
    assert len(book) == 4
    assert book.recv_ns.tolist() == [T0 + 100 * US, T0 + 1_000 * US, T0 + 3_000 * US, T0 + 10**6 * US]
    assert book.exch_ns.tolist() == [T0, T0 - 2_000 * US, T0 + 2_000 * US, T0 + 995_000 * US]
    assert book.n_bid.tolist() == [0, 2, 10, 1]
    assert book.n_ask.tolist() == [0, 2, 1, 1]
    # Best level first; prices are integer ticks of 0.5 USD; sizes are stored quantity units.
    assert book.bid_px[1, :2].tolist() == [40_000, 39_998]
    assert book.bid_sz[1, :2].tolist() == [2_000, 500]
    assert book.ask_px[1, :2].tolist() == [40_001, 40_003]
    assert book.ask_sz[1, :2].tolist() == [1_000, 300]
    # Absent levels are zero-filled.
    assert book.bid_px[0].tolist() == [0] * 10
    assert book.ask_sz[1, 2:].tolist() == [0] * 8
    # A declared depth above ten keeps the ten printed (best) levels.
    assert book.bid_px[2].tolist() == [2 * (19_999 - k) for k in range(10)]
    assert book.bid_sz[2, -1] == 1_000
    assert book.bid_px.dtype == PRICE_DTYPE and book.bid_sz.dtype == SIZE_DTYPE


def test_reference_parser_trade_prints(synthetic_recorder_text: bytes) -> None:
    tapes = parse_recorder_bytes_reference(synthetic_recorder_text, DEMO_INVERSE_A)
    trades = tapes.trades
    assert len(trades) == 3
    assert trades.recv_ns.tolist() == [T0 + 2_500 * US] * 3
    assert trades.exch_ns.tolist() == [T0 + 1_000 * US] * 3
    assert trades.px.tolist() == [40_001, 40_000, 39_998]
    assert trades.sz.tolist() == [100, 200, 300]
    # Raw recorder label: 'Bid' -> +1, 'Ask' -> -1. Aggressor meaning is verified on data separately.
    assert trades.side_label.tolist() == [1, -1, -1]
    assert tapes.report.skipped_lines == 2


def test_fast_parser_matches_reference(synthetic_recorder_text: bytes) -> None:
    ref = parse_recorder_bytes_reference(synthetic_recorder_text, DEMO_INVERSE_A)
    assert_tapes_equal(parse_recorder_bytes(synthetic_recorder_text, DEMO_INVERSE_A), ref)


@given(chunk=st.integers(min_value=1, max_value=4_096))
@settings(max_examples=60, deadline=None)
def test_chunk_size_does_not_change_the_result(chunk: int) -> None:
    ref = parse_recorder_bytes_reference(SYNTHETIC_RECORDER_TEXT, DEMO_INVERSE_A)
    got = parse_recorder_bytes(SYNTHETIC_RECORDER_TEXT, DEMO_INVERSE_A, chunk_bytes=chunk)
    assert_tapes_equal(got, ref)


def test_read_gzip_file_streams_in_chunks(tmp_path: Path, synthetic_recorder_text: bytes) -> None:
    path = tmp_path / "synthetic.txt.gz"
    path.write_bytes(gzip.compress(synthetic_recorder_text))
    got = read_recorder_file(path, DEMO_INVERSE_A, chunk_bytes=97)
    assert_tapes_equal(got, parse_recorder_bytes_reference(synthetic_recorder_text, DEMO_INVERSE_A))


PARSERS = [parse_recorder_bytes, parse_recorder_bytes_reference]


@pytest.mark.parametrize("parser", PARSERS)
def test_orderbook_clear_is_a_typed_empty_book_without_exchange_clock(parser) -> None:  # type: ignore[no-untyped-def]
    text = SYNTHETIC_RECORDER_TEXT + (
        b"23-01-30 23:59:59.957913 type: OrderbookClear symbol: none symbol_id: none "
        b"receive_time: 23-01-30 23:59:59.957913\n"
    )
    book = parser(text, DEMO_INVERSE_A).book
    assert book.kind.tolist() == [BOOK_SNAPSHOT] * 4 + [BOOK_CLEAR]
    assert book.recv_ns[-1] == T0 + 86_399_957_913 * US
    assert book.exch_ns[-1] == MISSING_NS  # never fabricated from the receipt clock
    assert book.n_bid[-1] == 0 and book.n_ask[-1] == 0
    assert book.bid_px[-1].tolist() == [0] * 10


@pytest.mark.parametrize("parser", PARSERS)
def test_off_tick_price_is_rejected(parser, synthetic_recorder_text: bytes) -> None:  # type: ignore[no-untyped-def]
    bad = synthetic_recorder_text.replace(b"20001.500000 volume: 300", b"20001.250000 volume: 300")
    with pytest.raises(InvalidDataError) as err:
        parser(bad, DEMO_INVERSE_A)
    assert err.value.code == "invalid_data"


@pytest.mark.parametrize("parser", PARSERS)
def test_fractional_size_is_rejected(parser, synthetic_recorder_text: bytes) -> None:  # type: ignore[no-untyped-def]
    bad = synthetic_recorder_text.replace(b"volume: 300.000000", b"volume: 300.500000")
    with pytest.raises(InvalidDataError):
        parser(bad, DEMO_INVERSE_A)


@pytest.mark.parametrize("parser", PARSERS)
def test_unknown_event_type_is_rejected(parser, synthetic_recorder_text: bytes) -> None:  # type: ignore[no-untyped-def]
    bad = synthetic_recorder_text.replace(b"type: Trades", b"type: Liquidation", 1)
    with pytest.raises(InvalidDataError):
        parser(bad, DEMO_INVERSE_A)
