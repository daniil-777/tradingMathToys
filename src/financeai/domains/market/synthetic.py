"""Deterministic synthetic multi-venue markets (invented values, never market data).

``synthetic_market(SyntheticMarketSpec(seed=7))`` builds random-walk order books for the
requested inverse-perpetual venues: per venue, receipt gaps of 0.3-6 ms, feed latencies of
0.2-9 ms (the exchange clock is kept monotone), a half-tick random-walk mid around 40,000 USD,
1-2 tick spreads and ten levels per side with lot-multiple sizes. There are no trade prints.

Venues are always generated in sorted key order from one seeded NumPy ``Generator``, so a spec
names one market regardless of the order in which its venues were requested; ``order=``
only arranges the resulting :class:`MarketData`. With the default venues and ``start_ns=0``
the output is bit-identical to the test-suite builder it was ported from. Determinism holds
for a given NumPy version (``Generator`` streams are not frozen across NumPy releases).

The generator is shared by examples, the CLI/Python/HTTP parity checks and the local job
service; its :meth:`SyntheticMarketSpec.descriptor` is the derivation identity of the data.

``start_ns`` is capped at :data:`MAX_START_NS` (``2**62``, 2116-02-20 UTC). Every generated
clock is then below ``2**62 + max_span_ns(rows)``, which leaves about 146 years of int64
headroom for the clocks a replay derives from them (order latency, acknowledgement,
consumption TTL, cooldown): near the int64 ceiling those sums would wrap silently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError
from financeai.core.hashing import sha256_json
from financeai.domains.market.cache import SourceInfo
from financeai.domains.market.instruments import (
    DEMO_INVERSE_A,
    DEMO_INVERSE_B,
    INSTRUMENTS,
    InstrumentSpec,
)
from financeai.domains.market.stream import MarketData
from financeai.domains.market.tape import DEPTH, BookTape, MarketTapes, ParseReport, TradeTape

GENERATOR: Final = {"name": "financeai.synthetic.random_walk_books", "version": "1"}
DEFAULT_VENUES: Final = (DEMO_INVERSE_A.key, DEMO_INVERSE_B.key)
MAX_ROWS_PER_VENUE: Final = 20_000
"""Keeps every generated price strictly positive (the mid moves at most 2 half-ticks per row)."""
MAX_SEED: Final = 2**53 - 1
"""Seeds stay exactly representable for every JSON consumer (including JavaScript numbers)."""
MAX_START_NS: Final = 2**62
"""Latest ``start_ns`` (2116-02-20 UTC): keeps generated *and* replay-derived clocks inside int64."""

_US: Final = 1_000
_GAP_US: Final = (300, 6_000)
_LATENCY_US: Final = (200, 9_000)
_FIRST_US: Final = 1_000
_BASE_HALF_TICKS: Final = 2 * 40_000
_SIZE_LOTS: Final = (1, 40)
_KEYS: Final = frozenset({"seed", "venues", "rows_per_venue", "start_ns"})


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def max_span_ns(rows_per_venue: int) -> int:
    """Upper bound of the last receipt clock relative to ``start_ns``."""
    return (rows_per_venue * (_GAP_US[1] - 1) + _FIRST_US) * _US


@dataclass(frozen=True, slots=True)
class SyntheticMarketSpec:
    """Validated description of one synthetic market (venues are normalised to sorted order)."""

    seed: int
    venues: tuple[str, ...] = DEFAULT_VENUES
    rows_per_venue: int = 300
    start_ns: int = 0
    instruments: tuple[InstrumentSpec, ...] = ()
    """UTC epoch nanoseconds added to every receipt/exchange clock, in ``[0, MAX_START_NS]``."""

    def __post_init__(self) -> None:
        if not _is_int(self.seed) or not 0 <= self.seed <= MAX_SEED:
            raise InvalidConfigError("seed must be an integer in [0, 2**53)", context={"got": self.seed})
        if not isinstance(self.venues, tuple | list) or not self.venues:
            raise InvalidConfigError("venues must be a non-empty sequence of instrument keys")
        if not isinstance(self.instruments, tuple | list) or any(
            not isinstance(i, InstrumentSpec) for i in self.instruments
        ):
            raise InvalidConfigError("instruments must contain InstrumentSpec values")
        registry = {i.key: i for i in self.instruments} if self.instruments else INSTRUMENTS
        if self.instruments and len(registry) != len(self.instruments):
            raise InvalidConfigError("synthetic instrument keys must be distinct")
        object.__setattr__(self, "instruments", tuple(sorted(self.instruments, key=lambda i: i.key)))
        venues = tuple(registry) if self.instruments and tuple(self.venues) == DEFAULT_VENUES else tuple(self.venues)
        unknown = [v for v in venues if not isinstance(v, str) or v not in registry]
        if unknown or len(set(venues)) != len(venues):
            raise InvalidConfigError(
                "venues must be distinct known instrument keys", context={"unknown": unknown, "known": sorted(registry)}
            )
        object.__setattr__(self, "venues", tuple(sorted(venues)))
        if not _is_int(self.rows_per_venue) or not 2 <= self.rows_per_venue <= MAX_ROWS_PER_VENUE:
            raise InvalidConfigError(
                "rows_per_venue must be an integer in [2, MAX_ROWS_PER_VENUE]",
                context={"got": self.rows_per_venue, "max": MAX_ROWS_PER_VENUE},
            )
        if not _is_int(self.start_ns) or not 0 <= self.start_ns <= MAX_START_NS:
            raise InvalidConfigError(
                "start_ns must be an integer number of nanoseconds in [0, 2**62]; later "
                "clocks leave no int64 room for the clocks a replay derives from them",
                context={"got": self.start_ns, "max": MAX_START_NS},
            )

    @classmethod
    def from_mapping(cls, doc: Mapping[str, Any]) -> SyntheticMarketSpec:
        if not isinstance(doc, Mapping) or not (_KEYS <= set(doc) <= _KEYS | {"instruments"}):
            raise InvalidConfigError(
                "synthetic market keys differ",
                context={"expected": sorted(_KEYS), "got": sorted(doc) if isinstance(doc, Mapping) else None},
            )
        venues = doc["venues"]
        if not isinstance(venues, list | tuple):
            raise InvalidConfigError("venues must be a list of instrument keys")
        custom = doc.get("instruments", [])
        if not isinstance(custom, list | tuple):
            raise InvalidConfigError("instruments must be an array of instrument specifications")
        return cls(
            instruments=tuple(InstrumentSpec.from_mapping(i) for i in custom),
            seed=doc["seed"],
            venues=tuple(venues),
            rows_per_venue=doc["rows_per_venue"],
            start_ns=doc["start_ns"],
        )

    def to_mapping(self) -> dict[str, Any]:
        out = {
            "seed": self.seed,
            "venues": list(self.venues),
            "rows_per_venue": self.rows_per_venue,
            "start_ns": self.start_ns,
        }
        if self.instruments:
            out["instruments"] = [i.to_mapping() for i in self.instruments]
        return out

    def descriptor(self) -> dict[str, Any]:
        """Derivation identity: generator name/version plus every parameter that shapes the data."""
        return {"generator": dict(GENERATOR), **self.to_mapping()}


def _frozen(arr: NDArray[Any]) -> NDArray[Any]:
    out = np.ascontiguousarray(arr)
    out.setflags(write=False)
    return out


def _venue_tapes(spec: InstrumentSpec, rng: np.random.Generator, n: int, start_ns: int) -> MarketTapes:
    # Keep RNG draw order deterministic: clocks, latencies, mids, then spread and depth.
    if spec.lot_size * (_SIZE_LOTS[1] - 1) > np.iinfo(np.int32).max:
        raise InvalidConfigError("synthetic lot sizes would overflow tape quantity storage")
    recv_us = np.cumsum(rng.integers(*_GAP_US, size=n)) + _FIRST_US
    latency_us = rng.integers(*_LATENCY_US, size=n)
    exch_us = np.maximum.accumulate(recv_us - latency_us)
    mid2 = _BASE_HALF_TICKS + np.cumsum(rng.integers(-2, 3, size=n))
    bid_px = np.zeros((n, DEPTH), np.int32)
    bid_sz = np.zeros((n, DEPTH), np.int32)
    ask_px = np.zeros((n, DEPTH), np.int32)
    ask_sz = np.zeros((n, DEPTH), np.int32)
    steps = np.arange(DEPTH, dtype=np.int64)
    for i in range(n):
        spread = int(rng.integers(1, 3))
        best_bid = int(mid2[i] // 2)
        sizes = rng.integers(*_SIZE_LOTS, size=2 * DEPTH) * spec.lot_size
        bid_px[i] = best_bid - steps
        ask_px[i] = best_bid + spread + steps
        bid_sz[i] = sizes[:DEPTH]
        ask_sz[i] = sizes[DEPTH:]
    book = BookTape(
        recv_ns=_frozen(start_ns + recv_us.astype(np.int64) * _US),
        exch_ns=_frozen(start_ns + exch_us.astype(np.int64) * _US),
        kind=_frozen(np.zeros(n, np.int8)),
        n_bid=_frozen(np.full(n, DEPTH, np.int8)),
        n_ask=_frozen(np.full(n, DEPTH, np.int8)),
        bid_px=_frozen(bid_px),
        bid_sz=_frozen(bid_sz),
        ask_px=_frozen(ask_px),
        ask_sz=_frozen(ask_sz),
    )
    trades = TradeTape(*(_frozen(np.zeros(0, dtype)) for dtype in (np.int64, np.int64, np.int32, np.int32, np.int8)))
    report = ParseReport(snapshots=n, clears=0, trade_lines=0, trades=0, skipped_lines=0, bytes_parsed=0)
    return MarketTapes(instrument=spec, book=book, trades=trades, report=report)


def synthetic_tapes(spec: SyntheticMarketSpec) -> dict[str, MarketTapes]:
    """Tapes per instrument key, generated in sorted key order from one seeded generator."""
    rng = np.random.default_rng(spec.seed)
    registry = {i.key: i for i in spec.instruments} if spec.instruments else INSTRUMENTS
    return {key: _venue_tapes(registry[key], rng, spec.rows_per_venue, spec.start_ns) for key in spec.venues}


def synthetic_market(spec: SyntheticMarketSpec, order: Sequence[str] | None = None) -> MarketData:
    """The synthetic :class:`MarketData`; ``order`` fixes the venue ids (defaults to sorted keys)."""
    tapes = synthetic_tapes(spec)
    keys = tuple(order) if order is not None else spec.venues
    missing = [k for k in keys if k not in tapes]
    if missing:
        raise InvalidConfigError(
            "order names venues the spec does not generate", context={"missing": missing, "venues": list(spec.venues)}
        )
    return MarketData(tuple(tapes[k] for k in keys))


def synthetic_source(spec: SyntheticMarketSpec, key: str) -> SourceInfo:
    """Cache provenance for one generated venue: the descriptor hash stands in for a raw file."""
    if key not in spec.venues:
        raise InvalidConfigError("venue is not part of the synthetic spec", context={"venue": key})
    return SourceInfo(name=f"synthetic:{key}", sha256=sha256_json({**spec.descriptor(), "venue": key}), bytes=0)


__all__ = [
    "DEFAULT_VENUES",
    "GENERATOR",
    "MAX_ROWS_PER_VENUE",
    "MAX_SEED",
    "MAX_START_NS",
    "SyntheticMarketSpec",
    "max_span_ns",
    "synthetic_market",
    "synthetic_source",
    "synthetic_tapes",
]
