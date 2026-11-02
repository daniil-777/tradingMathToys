"""Immutable on-disk tape cache: exact round trip, immutability and integrity checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from financeai.core.errors import IncompatibleSchemaError, InvalidConfigError, InvalidDataError
from financeai.domains.market.cache import SourceInfo, load_tapes, save_tapes
from financeai.domains.market.instruments import DEMO_INVERSE_A
from financeai.domains.market.recorder import parse_recorder_bytes_reference
from financeai.domains.market.tape import MarketTapes

from .conftest import SYNTHETIC_RECORDER_TEXT
from .test_recorder_parser import assert_tapes_equal

SOURCE = SourceInfo(name="synthetic.txt.gz", sha256="0" * 64, bytes=123)


@pytest.fixture
def tapes() -> MarketTapes:
    return parse_recorder_bytes_reference(SYNTHETIC_RECORDER_TEXT, DEMO_INVERSE_A)


def test_round_trip_is_exact_and_read_only(tmp_path: Path, tapes: MarketTapes) -> None:
    target = tmp_path / "cache"
    manifest = save_tapes(tapes, target, source=SOURCE)
    loaded = load_tapes(target)
    assert_tapes_equal(loaded, tapes)
    assert loaded.instrument == tapes.instrument
    assert loaded.report == tapes.report
    assert manifest["source"]["sha256"] == SOURCE.sha256
    with pytest.raises(ValueError):
        loaded.book.bid_px[0, 0] = 1  # memory-mapped read-only


def test_existing_cache_is_never_overwritten(tmp_path: Path, tapes: MarketTapes) -> None:
    target = tmp_path / "cache"
    save_tapes(tapes, target, source=SOURCE)
    with pytest.raises(InvalidConfigError):
        save_tapes(tapes, target, source=SOURCE)


def test_unknown_major_schema_is_rejected(tmp_path: Path, tapes: MarketTapes) -> None:
    target = tmp_path / "cache"
    save_tapes(tapes, target, source=SOURCE)
    manifest_path = target / "manifest.json"
    doc = json.loads(manifest_path.read_text())
    doc["schema_version"] = "99.0"
    manifest_path.write_text(json.dumps(doc))
    with pytest.raises(IncompatibleSchemaError):
        load_tapes(target)


def test_verification_detects_a_tampered_column(tmp_path: Path, tapes: MarketTapes) -> None:
    target = tmp_path / "cache"
    save_tapes(tapes, target, source=SOURCE)
    column = target / "book.bid_sz.npy"
    raw = bytearray(column.read_bytes())
    raw[-1] ^= 0xFF
    column.write_bytes(bytes(raw))
    load_tapes(target)  # cheap load trusts sizes/dtypes only
    with pytest.raises(InvalidDataError):
        load_tapes(target, verify=True)


def test_column_with_wrong_dtype_is_rejected(tmp_path: Path, tapes: MarketTapes) -> None:
    target = tmp_path / "cache"
    save_tapes(tapes, target, source=SOURCE)
    np.save(target / "trades.px.npy", np.asarray(tapes.trades.px, dtype=np.int64))
    with pytest.raises(InvalidDataError):
        load_tapes(target)


def test_cache_records_an_alternate_producer(tmp_path: Path, tapes: MarketTapes) -> None:
    parser = {"name": "synthetic", "version": "1"}
    target = tmp_path / "cache"
    manifest = save_tapes(tapes, target, source=SOURCE, parser=parser)
    parser["name"] = "changed_after_write"
    assert manifest["parser"] == {"name": "synthetic", "version": "1"}
    assert json.loads((target / "manifest.json").read_text())["parser"] == manifest["parser"]
    assert_tapes_equal(load_tapes(target, verify=True), tapes)


@pytest.mark.parametrize("parser", [{}, {"name": "custom"}, {"name": "custom", "version": ""}])
def test_cache_rejects_incomplete_provenance(tmp_path: Path, tapes: MarketTapes, parser: dict[str, str]) -> None:
    target = tmp_path / "cache"
    with pytest.raises(InvalidConfigError):
        save_tapes(tapes, target, source=SOURCE, parser=parser)
    assert not target.exists()
