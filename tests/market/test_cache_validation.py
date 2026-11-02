"""Corrupt caches must fail at the boundary without changing contract semantics."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from financeai.core.errors import IncompatibleSchemaError, InvalidConfigError, InvalidDataError
from financeai.domains.market.cache import SourceInfo, load_tapes, save_tapes
from financeai.domains.market.instruments import DEMO_INVERSE_A
from financeai.domains.market.recorder import parse_recorder_bytes_reference
from financeai.domains.market.stream import MarketData
from financeai.domains.market.tape import BookTape, MarketTapes, TradeTape

from .conftest import SYNTHETIC_RECORDER_TEXT
from .test_tape_cache import SOURCE


@pytest.fixture
def tapes() -> MarketTapes:
    return parse_recorder_bytes_reference(SYNTHETIC_RECORDER_TEXT, DEMO_INVERSE_A)


def write_manifest(root: Path, doc: dict[str, Any]) -> None:
    (root / "manifest.json").write_text(json.dumps(doc))


@pytest.fixture
def cache(tmp_path: Path, tapes: MarketTapes) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "cache"
    return root, save_tapes(tapes, root, source=SOURCE)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lot_size", 1.5),
        ("lot_size", "100"),
        ("lot_size", True),
        ("shortable", "false"),
        ("shortable", 1),
        ("tick_size", "NaN"),
        ("tick_size", "not-a-decimal"),
    ],
)
def test_cache_instrument_values_are_not_coerced(cache: tuple[Path, dict[str, Any]], field: str, value: object) -> None:
    root, doc = cache
    doc["instrument"][field] = value
    write_manifest(root, doc)
    with pytest.raises(InvalidDataError):
        load_tapes(root)


@pytest.mark.parametrize("section", ["instrument", "arrays", "report", "source", "parser"])
@pytest.mark.parametrize("value", [None, [], "invalid"])
def test_malformed_manifest_sections_raise_data_errors(
    cache: tuple[Path, dict[str, Any]], section: str, value: object
) -> None:
    root, doc = cache
    doc[section] = value
    write_manifest(root, doc)
    with pytest.raises(InvalidDataError):
        load_tapes(root)


@pytest.mark.parametrize("version", ["1.x", "1.", "1.-1", "1.01"])
def test_malformed_cache_version_is_rejected(cache: tuple[Path, dict[str, Any]], version: str) -> None:
    root, doc = cache
    doc["schema_version"] = version
    write_manifest(root, doc)
    with pytest.raises(IncompatibleSchemaError):
        load_tapes(root)


@pytest.mark.parametrize("filename", ["../outside.npy", "/tmp/outside.npy", "subdir/column.npy", "subdir\\column.npy"])
def test_column_paths_stay_inside_the_cache(
    cache: tuple[Path, dict[str, Any]], filename: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, doc = cache
    doc["arrays"]["book.recv_ns"]["file"] = filename
    write_manifest(root, doc)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("an invalid column path reached NumPy")

    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(InvalidDataError):
        load_tapes(root)


def test_symlink_columns_are_rejected(cache: tuple[Path, dict[str, Any]]) -> None:
    root, _ = cache
    column = root / "book.recv_ns.npy"
    outside = root.parent / "outside.npy"
    column.rename(outside)
    column.symlink_to(outside)
    with pytest.raises(InvalidDataError):
        load_tapes(root)


@pytest.mark.parametrize("entry", [None, [], {"file": "book.recv_ns.npy"}])
def test_incomplete_column_metadata_is_rejected(cache: tuple[Path, dict[str, Any]], entry: object) -> None:
    root, doc = cache
    doc["arrays"]["book.recv_ns"] = entry
    write_manifest(root, doc)
    with pytest.raises(InvalidDataError):
        load_tapes(root)


@pytest.mark.parametrize("content", [b"", b"not a numpy file", b"\x93NUMPY\x01\x00"])
def test_truncated_columns_raise_data_errors(cache: tuple[Path, dict[str, Any]], content: bytes) -> None:
    root, _ = cache
    (root / "book.recv_ns.npy").write_bytes(content)
    with pytest.raises(InvalidDataError):
        load_tapes(root)


@pytest.mark.parametrize("count", [-1, 11, 127])
def test_book_depth_counts_cannot_index_outside_level_arrays(tapes: MarketTapes, count: int) -> None:
    with pytest.raises(InvalidDataError):
        dataclasses.replace(tapes.book, n_bid=np.full(len(tapes.book), count, np.int8))


@pytest.mark.parametrize("book", [False, True])
def test_scalar_receipt_columns_raise_data_errors(tapes: MarketTapes, book: bool) -> None:
    original: BookTape | TradeTape = tapes.book if book else tapes.trades
    with pytest.raises(InvalidDataError):
        dataclasses.replace(original, recv_ns=np.array(0, dtype=np.int64))


@pytest.mark.parametrize("mmap", [True, False])
def test_a_numpy_archive_cannot_disguise_itself_as_a_column(cache: tuple[Path, dict[str, Any]], mmap: bool) -> None:
    root, _ = cache
    with (root / "book.recv_ns.npy").open("wb") as file:
        np.savez(file, recv_ns=np.arange(4, dtype=np.int64))
    with pytest.raises(InvalidDataError, match="one NumPy array"):
        load_tapes(root, mmap=mmap)


@pytest.mark.parametrize(("field", "value"), [("name", ""), ("sha256", "not-a-digest"), ("bytes", True), ("bytes", -1)])
def test_invalid_source_metadata_cannot_be_written(field: str, value: Any) -> None:
    with pytest.raises(InvalidConfigError):
        SourceInfo(**{**dataclasses.asdict(SOURCE), field: value})


def test_receipt_order_validation_does_not_overflow(tapes: MarketTapes) -> None:
    clocks = np.arange(len(tapes.book), dtype=np.int64) - 2
    clocks[0] = np.iinfo(np.int64).max
    book = dataclasses.replace(tapes.book, recv_ns=clocks)
    with pytest.raises(InvalidDataError, match="not monotone"):
        MarketData((dataclasses.replace(tapes, book=book),))


def test_venue_ids_cannot_wrap_to_negative_int8_values(tapes: MarketTapes) -> None:
    venues = tuple(
        dataclasses.replace(tapes, instrument=dataclasses.replace(tapes.instrument, symbol=str(i))) for i in range(129)
    )
    with pytest.raises(InvalidConfigError, match="128 venues"):
        MarketData(venues)
