"""Synthetic exact schema/clock/ordering checks before opening external data."""

from __future__ import annotations

import csv
import dataclasses
import gzip
from decimal import Decimal

import numpy as np
import pytest

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.domains.market.instruments import DEMO_INVERSE_A, DEMO_INVERSE_B
from financeai.domains.market.tardis import read_tardis_snapshot_trades


def examples(spec):
    book = []
    trades = []
    for i in range(4):
        row = {
            "exchange": spec.venue,
            "symbol": spec.symbol,
            "timestamp": str(1677628800000000 + i),
            "local_timestamp": str(1677628800000001 + i // 2),
        }
        for k in range(25):
            for side, price in [("asks", Decimal("20000.5") + k), ("bids", Decimal("20000") - k)]:
                row[f"{side}[{k}].price"] = str(price)
                row[f"{side}[{k}].amount"] = str((k + 1) * 100)
        book.append(row)
        trades.append(
            {
                "exchange": spec.venue,
                "symbol": spec.symbol,
                "timestamp": row["timestamp"],
                "local_timestamp": row["local_timestamp"],
                "id": f"trade-{i}",
                "side": "buy" if i % 2 == 0 else "sell",
                "price": "20000.5000",
                "amount": "300.0",
            }
        )
    return book, trades


def write_csv(path, records, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, "wt", newline="") as f:
        writer = csv.DictWriter(f, list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def load(tmp_path, books, trades, spec=DEMO_INVERSE_A, chunk=97, compressed=False):
    suffix = ".csv.gz" if compressed else ".csv"
    b, t = tmp_path / f"book{suffix}", tmp_path / f"trades{suffix}"
    write_csv(b, books, compressed)
    write_csv(t, trades, compressed)
    return read_tardis_snapshot_trades(b, t, spec, chunk_bytes=chunk)


@pytest.mark.parametrize("spec", [DEMO_INVERSE_A, DEMO_INVERSE_B])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("chunk", [1, 97, 1 << 20])
def test_exact_projection_clocks_and_tied_cohorts(tmp_path, spec, compressed, chunk):
    books, trades = examples(spec)
    data = load(tmp_path, books, trades, spec, chunk, compressed)
    assert len(data.book) == len(data.trades) == 4
    assert data.book.recv_ns.tolist() == [int(b["local_timestamp"]) * 1000 for b in books]
    assert data.book.exch_ns.tolist() == [int(b["timestamp"]) * 1000 for b in books]
    assert data.trades.side_label.tolist() == [1, -1, 1, -1]
    assert data.trades.px.tolist() == [40001] * 4
    assert data.trades.sz.tolist() == [300] * 4
    for i, row in enumerate(books):
        for source, px, sz in [
            ("bids", data.book.bid_px, data.book.bid_sz),
            ("asks", data.book.ask_px, data.book.ask_sz),
        ]:
            for k in range(10):
                assert px[i, k] == int(Decimal(row[f"{source}[{k}].price"]) / spec.tick_size)
                assert sz[i, k] == int(Decimal(row[f"{source}[{k}].amount"]))
    assert not data.book.recv_ns.flags.writeable
    assert not data.trades.side_label.flags.writeable
    assert data.report.snapshots == data.report.trades == 4


def test_missing_trailing_levels_and_crossed_quote_are_preserved(tmp_path):
    books, trades = examples(DEMO_INVERSE_A)
    books[0]["bids[0].price"] = "20001"  # downstream validity handles crossed top
    for k in range(3, 25):
        for side in ["bids", "asks"]:
            books[1][f"{side}[{k}].price"] = books[1][f"{side}[{k}].amount"] = ""
    data = load(tmp_path, books, trades)
    assert data.book.n_bid.tolist() == [10, 3, 10, 10]
    assert data.book.n_ask.tolist() == [10, 3, 10, 10]
    assert np.all(data.book.bid_px[1, 3:] == 0)
    assert data.book.bid_px[0, 0] > data.book.ask_px[0, 0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("bids[0].price", "20000.1"),
        ("asks[0].amount", "100.1"),
        ("asks[0].amount", "-1"),
        ("asks[0].amount", "2147483648"),
        ("bids[0].price", "0"),
        ("bids[0].price", "NaN"),
        ("bids[0].price", "Infinity"),
        ("bids[0].price", "1e10"),
        ("bids[0].price", ""),
        ("bids[1].price", "20000"),
        ("asks[1].price", "19999"),
        ("symbol", "WRONG"),
        ("exchange", "WRONG"),
        ("local_timestamp", "9223372036854776"),
        ("timestamp", "1.5"),
        ("local_timestamp", ""),
    ],
)
def test_bad_book_values_rejected(tmp_path, field, value):
    books, trades = examples(DEMO_INVERSE_A)
    books[0][field] = value
    with pytest.raises(InvalidDataError):
        load(tmp_path, books, trades)


@pytest.mark.parametrize(
    "field,value",
    [("side", "unknown"), ("side", "Bid"), ("amount", "0"), ("amount", "10.5"), ("price", "0.1"), ("timestamp", "-1")],
)
def test_bad_trade_values_rejected(tmp_path, field, value):
    books, trades = examples(DEMO_INVERSE_A)
    trades[0][field] = value
    with pytest.raises(InvalidDataError):
        load(tmp_path, books, trades)


def test_receipt_regression_and_missing_inside_book_rejected(tmp_path):
    books, trades = examples(DEMO_INVERSE_A)
    books[3]["local_timestamp"] = "1"
    with pytest.raises(InvalidDataError, match="nondecreasing"):
        load(tmp_path, books, trades)
    books, trades = examples(DEMO_INVERSE_A)
    books[0]["bids[1].price"] = books[0]["bids[1].amount"] = ""
    with pytest.raises(InvalidDataError, match="prefix"):
        load(tmp_path, books, trades)


def test_header_contract_and_bad_chunk(tmp_path):
    books, trades = examples(DEMO_INVERSE_A)
    for row in books:
        del row["bids[24].price"]
    with pytest.raises(InvalidDataError, match="schema"):
        load(tmp_path, books, trades)
    with pytest.raises(InvalidConfigError):
        read_tardis_snapshot_trades("missing", "missing", DEMO_INVERSE_A, chunk_bytes=0)
    other = dataclasses.replace(DEMO_INVERSE_A, tick_size=Decimal("1"))
    # Arbitrary instrument grids are supported; a missing source is an I/O error.
    with pytest.raises(FileNotFoundError):
        read_tardis_snapshot_trades("missing", "missing", other)


@pytest.mark.parametrize("tick,step,price_scale", [("0.25", "0.001", "1"), ("0.00000001", "0.000001", "1000000")])
def test_arbitrary_instruments_and_fractional_quantity_grids(tmp_path, tick, step, price_scale):
    from financeai.domains.market.instruments import ContractType, InstrumentSpec

    spec = InstrumentSpec(
        "custom",
        "ASSET-EUR",
        Decimal(tick),
        contract_type=ContractType.SPOT,
        quantity_step=Decimal(step),
        quote_currency="EUR",
    )
    books, trades = examples(spec)
    for row in books:
        for key, value in row.items():
            if key.endswith(".price"):
                row[key] = str(Decimal(value) / Decimal(price_scale))
            elif key.endswith(".amount"):
                row[key] = format(Decimal(value) * Decimal(step), "f")
    for row in trades:
        row["price"] = format(Decimal(row["price"]) / Decimal(price_scale), "f")
        row["amount"] = format(Decimal(row["amount"]) * Decimal(step), "f")
    data = load(tmp_path, books, trades, spec)
    assert data.trades.sz.tolist() == [300] * 4
    assert data.book.bid_px[0, 0] == int(Decimal("20000") / Decimal(price_scale) / Decimal(tick))


def test_cli_provenance_hashes_both_inputs_and_names_the_parser(tmp_path) -> None:
    import json

    from financeai.cli import main

    book, trades = examples(DEMO_INVERSE_A)
    book_path, trade_path = tmp_path / "book.csv", tmp_path / "trades.csv"
    write_csv(book_path, book)
    write_csv(trade_path, trades)
    manifests = []
    for iteration in range(3):
        if iteration == 1:
            trades[0]["amount"] = "400.0"
            write_csv(trade_path, trades)
        elif iteration == 2:
            book[0]["asks[0].amount"] = "500"
            write_csv(book_path, book)
        out = tmp_path / f"cache-{iteration}"
        assert (
            main(
                [
                    "ingest",
                    str(book_path),
                    "--instrument",
                    DEMO_INVERSE_A.key,
                    "--format",
                    "tardis",
                    "--trades",
                    str(trade_path),
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["source"]["bytes"] == book_path.stat().st_size + trade_path.stat().st_size
        assert manifest["parser"] == {"name": "tardis-normalized-csv", "version": "1.0"}
        manifests.append(manifest)
    assert len({m["source"]["sha256"] for m in manifests}) == 3
