"""Immutable on-disk cache of parsed tapes (one ``.npy`` per column + JSON manifest).

A cache directory is written once into a staging directory and published by an atomic
rename; existing caches are never overwritten. Loading memory-maps the columns
read-only (``allow_pickle=False``) and checks dtypes/shapes against the manifest;
``verify=True`` additionally re-hashes every column.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Final

import numpy as np

from financeai.core.errors import IncompatibleSchemaError, InvalidConfigError, InvalidDataError
from financeai.core.hashing import sha256_file
from financeai.core.serialization import loads_strict
from financeai.domains.market.instruments import InstrumentSpec
from financeai.domains.market.tape import BookTape, MarketTapes, ParseReport, TradeTape

SCHEMA_NAME: Final = "financeai.market_tapes"
SCHEMA_VERSION: Final = "2.0"
PARSER_ID: Final = {"name": "recorder_text", "version": "1"}

_BOOK_COLUMNS: Final = ("recv_ns", "exch_ns", "kind", "n_bid", "n_ask", "bid_px", "bid_sz", "ask_px", "ask_sz")
_TRADE_COLUMNS: Final = ("recv_ns", "exch_ns", "px", "sz", "side_label")
_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_VERSION: Final = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Identity of the raw input a cache was derived from."""

    name: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidConfigError("source name must be a non-empty string")
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise InvalidConfigError("source sha256 must be a lowercase SHA-256 digest")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 0:
            raise InvalidConfigError("source bytes must be a non-negative integer")

    @classmethod
    def from_file(cls, path: str | Path) -> SourceInfo:
        p = Path(path)
        return cls(name=p.name, sha256=sha256_file(p), bytes=p.stat().st_size)


def save_tapes(
    tapes: MarketTapes, directory: str | Path, *, source: SourceInfo, parser: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Write tapes to a new cache, recording the producing parser or generator.

    ``parser`` defaults to the recorder parser identity for backward compatibility.
    Other producers supply their own ``name`` and ``version``; this does not change
    the column schema or the raw-input identity in ``source``.
    """
    if parser is not None and not isinstance(parser, Mapping):
        raise InvalidConfigError("parser identity must be an object")
    producer = dict(PARSER_ID if parser is None else parser)
    if set(producer) != {"name", "version"} or any(not isinstance(v, str) or not v.strip() for v in producer.values()):
        raise InvalidConfigError("parser identity requires non-empty name and version strings")
    target = Path(directory)
    if target.exists():
        raise InvalidConfigError("cache directory already exists; caches are immutable", context={"path": str(target)})
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        arrays: dict[str, dict[str, Any]] = {}
        for prefix, tape, columns in (("book", tapes.book, _BOOK_COLUMNS), ("trades", tapes.trades, _TRADE_COLUMNS)):
            for column in columns:
                arr = np.ascontiguousarray(getattr(tape, column))
                filename = f"{prefix}.{column}.npy"
                np.save(staging / filename, arr, allow_pickle=False)
                arrays[f"{prefix}.{column}"] = {
                    "file": filename,
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "sha256": sha256_file(staging / filename),
                }
        spec = tapes.instrument
        manifest: dict[str, Any] = {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "instrument": spec.to_mapping(),
            "source": asdict(source),
            "parser": producer,
            "report": asdict(tapes.report),
            "arrays": arrays,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        with manifest_path.open("rb") as handle:
            os.fsync(handle.fileno())
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _check_version(version: object) -> None:
    if not isinstance(version, str) or _VERSION.fullmatch(version) is None:
        raise IncompatibleSchemaError("malformed schema_version", context={"got": version})
    major = version.split(".")[0]
    if major != SCHEMA_VERSION.split(".")[0]:
        raise IncompatibleSchemaError(
            "unsupported cache schema major version", context={"got": version, "supported": SCHEMA_VERSION}
        )


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidDataError("cache field must be an object", context={"field": field})
    return value


def _nonnegative_int(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidDataError("cache field must be a non-negative integer", context={"field": field})


def _text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidDataError("cache field must be a non-empty string", context={"field": field})


def _digest(value: object, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise InvalidDataError("cache field must be a lowercase SHA-256 digest", context={"field": field})


def _instrument(value: object) -> InstrumentSpec:
    inst = _object(value, "instrument")
    required = {
        "venue",
        "symbol",
        "tick_size",
        "lot_size",
        "contract_type",
        "quantity_step",
        "contract_multiplier",
        "base_currency",
        "quote_currency",
        "settlement_currency",
        "shortable",
    }
    if not required <= inst.keys():
        raise InvalidDataError(
            "cache instrument metadata is incomplete", context={"missing": sorted(required - inst.keys())}
        )
    try:
        return InstrumentSpec.from_mapping(inst, allow_unknown=True)
    except InvalidConfigError as exc:
        raise InvalidDataError("invalid cache instrument", context={"field": "instrument"}) from exc


def _column_metadata(value: object, key: str) -> dict[str, Any]:
    entry = _object(value, f"arrays.{key}")
    filename = entry.get("file")
    if (
        not isinstance(filename, str)
        or not filename.endswith(".npy")
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or Path(filename).name != filename
    ):
        raise InvalidDataError("cache column must name a local .npy file", context={"column": key})
    _text(entry.get("dtype"), f"arrays.{key}.dtype")
    shape = entry.get("shape")
    if not isinstance(shape, list):
        raise InvalidDataError("cache column shape must be an array", context={"column": key})
    for size in shape:
        _nonnegative_int(size, f"arrays.{key}.shape")
    _digest(entry.get("sha256"), f"arrays.{key}.sha256")
    return entry


def read_manifest(directory: str | Path) -> dict[str, Any]:
    """Read and validate metadata before any array paths are accessed.

    Extra fields are retained for compatible minor versions and producer provenance.
    Required fields are never coerced into a different contract or array description.
    """
    doc = loads_strict((Path(directory) / "manifest.json").read_bytes())
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA_NAME:
        raise IncompatibleSchemaError("not a market-tapes cache", context={"path": str(directory)})
    _check_version(doc.get("schema_version"))
    _instrument(doc.get("instrument"))
    source = _object(doc.get("source"), "source")
    _text(source.get("name"), "source.name")
    _digest(source.get("sha256"), "source.sha256")
    _nonnegative_int(source.get("bytes"), "source.bytes")
    parser = _object(doc.get("parser"), "parser")
    for key in ("name", "version"):
        _text(parser.get(key), f"parser.{key}")
    report = _object(doc.get("report"), "report")
    for field in fields(ParseReport):
        _nonnegative_int(report.get(field.name), f"report.{field.name}")
    arrays = _object(doc.get("arrays"), "arrays")
    for prefix, columns in (("book", _BOOK_COLUMNS), ("trades", _TRADE_COLUMNS)):
        for name in columns:
            key = f"{prefix}.{name}"
            _column_metadata(arrays.get(key), key)
    return doc


def load_tapes(directory: str | Path, *, verify: bool = False, mmap: bool = True) -> MarketTapes:
    """Load a cache written by :func:`save_tapes` (memory-mapped, read-only by default)."""
    root = Path(directory)
    manifest = read_manifest(root)
    instrument = _instrument(manifest["instrument"])

    def column(key: str) -> np.ndarray[Any, Any]:
        entry = manifest["arrays"][key]
        path = root / entry["file"]
        if path.is_symlink():
            raise InvalidDataError("cache columns must not be symbolic links", context={"column": key})
        try:
            if verify and sha256_file(path) != entry["sha256"]:
                raise InvalidDataError("cache column hash mismatch", context={"column": key})
            loaded = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        except (OSError, ValueError, EOFError) as exc:
            raise InvalidDataError("cannot read cache column", context={"column": key}) from exc
        if not isinstance(loaded, np.ndarray):
            loaded.close()  # np.load can return an NpzFile, even with a .npy filename
            raise InvalidDataError("cache column must contain one NumPy array", context={"column": key})
        arr: np.ndarray[Any, Any] = loaded
        if str(arr.dtype) != entry["dtype"] or list(arr.shape) != entry["shape"]:
            raise InvalidDataError(
                "cache column dtype/shape differs from manifest",
                context={"column": key, "dtype": str(arr.dtype), "shape": arr.shape},
            )
        return arr

    book = BookTape(*(column(f"book.{c}") for c in _BOOK_COLUMNS))
    trades = TradeTape(*(column(f"trades.{c}") for c in _TRADE_COLUMNS))
    report = ParseReport(**{field.name: manifest["report"][field.name] for field in fields(ParseReport)})
    return MarketTapes(instrument=instrument, book=book, trades=trades, report=report)
