"""Causal per-event market signals aligned with the receipt-ordered event stream.

Every signal is one ``float64`` per :class:`EventStream` event, and the value at event ``e``
is a function of events ``0..e`` only. This *index* causality is stricter than receipt-time
causality: inside a receipt cohort (events sharing ``recv_ns``) an early event does not see
the rest of its cohort; the replay engine acts on the value after a cohort's last event.
Reference averages therefore end at the *previous cohort*: :func:`basis_deviation_bps`
measures the basis against the EWMA of strictly earlier receipt times (a cohort enters the
average only once it is complete, with its last finite value), so a jump is reported in full
at the event the engine acts on instead of being partly absorbed by same-cohort events.

Conventions shared by every feature:

* Book quantities come from each venue's *observed* row, the last book row received at or
  before ``e``. They are NaN while that row is missing or unusable (a clear, one-sided or
  crossed book, see :func:`book_validity`): NaN means "not available" and is never replaced
  by zero or by an older book. A sum over an empty set (no trade printed yet) is exactly 0.
* Time decay is ``exp(-dt / tau)`` on integer receipt-clock differences. ``tau`` and windows
  are positive integer nanoseconds (milliseconds in specs, converted exactly).
* Prices use the instrument quote currency (``ticks * tick_size``); sizes use stored quantity units.
  ``side_label`` +1 is an aggressive
  buy according to the normalized tape contract, so positive flow is net buying pressure.
* Each signal is one fused O(n) pass over the stream in the Numba subset. Book-derived values
  are computed when the venue's row arrives, so the output is the only full-length array
  allocated (at 16M events, temporaries and their page faults cost more than the arithmetic).
  Without Numba the identical Python runs; array elements are wrapped in ``int()``/``float()``
  wherever NumPy scalar arithmetic could otherwise diverge from the compiled path.

:func:`build_signal` evaluates a strictly validated JSON-like spec and :func:`spec_id` names
it for caches.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from financeai.core.clock import NS_PER_MS
from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.core.hashing import sha256_json
from financeai.core.jit import njit
from financeai.domains.market.engine import PreparedMarket
from financeai.domains.market.stream import BOOK_EVENT, TRADE_EVENT
from financeai.domains.market.tape import DEPTH

type VenueRef = int | str
"""A venue index into ``prep.data.venues`` or an instrument key such as ``"venue_a/ALPHA-USD"``."""
type Book = tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int8],
    NDArray[np.int8],
    NDArray[np.bool_],
]
"""One venue's ``(bid_px, bid_sz, ask_px, ask_sz, n_bid, n_ask, usable)`` row arrays."""

PRICE_KINDS: Final = ("mid", "micro")
SIGNAL_KINDS: Final = ("basis_dev", "lead_gap", "trade_flow", "ofi", "imbalance", "return", "linear")
FLOW_DEPTH_LEVELS: Final = 5
"""Levels per side whose resting stored quantity normalises :func:`trade_flow`."""
MAX_DURATION_NS: Final = 2**53
"""Largest ``tau``/window: keeps ``float(dt)`` exact and clock arithmetic far from overflow."""
MAX_SPEC_NESTING: Final = 8
SPEC_SCHEMA: Final = "financeai.market.signal_spec"
SPEC_VERSION: Final = "2"
"""Part of every :func:`spec_id`; bump it whenever a kernel's numerical meaning changes.

Version 2: ``basis_dev`` subtracts the EWMA of strictly earlier receipt cohorts (version 1
subtracted the EWMA of earlier *events*, which let a same-cohort event absorb part of a jump).
"""

_CHECKED: weakref.WeakValueDictionary[int, PreparedMarket] = weakref.WeakValueDictionary()
"""Prepared markets whose stream already passed :meth:`_Context._check_stream` (they are
frozen with read-only arrays, so one check per object suffices; entries die with it)."""


# ---------------------------------------------------------------------------
# Row-level helpers (Numba subset)
# ---------------------------------------------------------------------------

_helper = njit(inline="always")
"""Helpers are inlined into the stream kernels at Numba-IR level: otherwise the 7-array
``book`` tuple is passed by value on every row and the kernels run up to 2x slower."""


@_helper
def _mid(bid: float, ask: float) -> float:
    return 0.5 * (bid + ask)


@_helper
def _micro(bid: float, ask: float, bid_size: float, ask_size: float) -> float:
    """Size-weighted mid; it leans towards the thinner side, the one more likely to give way."""
    den = bid_size + ask_size
    if den > 0.0:
        return (bid * ask_size + ask * bid_size) / den
    return np.nan


@_helper
def _rel_bps(price: float, base: float) -> float:
    """``(price / base - 1) * 1e4``; NaN unless both prices are positive (so NaN-safe)."""
    if price > 0.0 and base > 0.0:
        return (price / base - 1.0) * 1e4
    return np.nan


@_helper
def _ewma_step(m: float, value: float, dt: int, tau: float) -> float:
    """Move ``m`` towards ``value`` by ``alpha = 1 - exp(-dt / tau)``.

    Written as ``m + alpha * (value - m)`` with ``alpha = -expm1(-dt / tau)`` rather than
    ``w * m + (1 - w) * value``: a repeated value or ``dt == 0`` then leaves ``m`` exactly
    unchanged, and a small ``alpha`` keeps full relative precision.
    """
    return m - math.expm1(-float(dt) / tau) * (value - m)


@_helper
def _decay(total: float, dt: int, tau: float) -> float:
    """``total * exp(-dt / tau)``; exact (and free) for a zero total or ``dt == 0``."""
    if total == 0.0 or dt == 0:
        return total
    return total * math.exp(-float(dt) / tau)


@_helper
def _price_at(book: Book, r: int, tick: float, micro: bool) -> float:
    """Mid or microprice in quote currency of book row ``r``; NaN for an unusable row."""
    bpx, bsz, apx, asz, _nb, _na, valid = book
    if not valid[r]:
        return np.nan
    bid = float(bpx[r, 0]) * tick
    ask = float(apx[r, 0]) * tick
    if micro:
        return _micro(bid, ask, float(bsz[r, 0]), float(asz[r, 0]))
    return _mid(bid, ask)


@_helper
def _side_units(sz: NDArray[np.int32], populated: int, r: int, levels: int) -> float:
    total = 0.0
    for k in range(min(levels, populated, sz.shape[1])):
        total += float(sz[r, k])
    return total


@_helper
def _depth_at(book: Book, r: int, levels: int) -> float:
    """Bid + ask stored quantity units on the first ``levels`` populated levels of row ``r``; NaN when the row
    is unusable or empty, so that a ratio over it is NaN rather than infinite."""
    _bpx, bsz, _apx, asz, nb, na, valid = book
    if not valid[r]:
        return np.nan
    total = _side_units(bsz, int(nb[r]), r, levels) + _side_units(asz, int(na[r]), r, levels)
    return total if total > 0.0 else np.nan


@_helper
def _imbalance_at(book: Book, r: int, levels: int) -> float:
    _bpx, bsz, _apx, asz, nb, na, valid = book
    if not valid[r]:
        return np.nan
    bid = _side_units(bsz, int(nb[r]), r, levels)
    ask = _side_units(asz, int(na[r]), r, levels)
    total = bid + ask
    return (bid - ask) / total if total > 0.0 else np.nan


@_helper
def _ofi_at(book: Book, r: int) -> float:
    """Level-1 order-flow increment of row ``r`` against row ``r - 1`` (Cont, Kukanov and
    Stoikov 2014). A pair touching an unusable row (clear, one-sided, crossed) contributes 0:
    the book after a clear is a fresh state, not flow."""
    bpx, bsz, apx, asz, _nb, _na, valid = book
    if r == 0 or not valid[r] or not valid[r - 1]:
        return 0.0
    b0 = int(bpx[r - 1, 0])
    b1 = int(bpx[r, 0])
    a0 = int(apx[r - 1, 0])
    a1 = int(apx[r, 0])
    e = 0.0
    if b1 >= b0:
        e += float(bsz[r, 0])
    if b1 <= b0:
        e -= float(bsz[r - 1, 0])
    if a1 <= a0:
        e -= float(asz[r, 0])
    if a1 >= a0:
        e += float(asz[r - 1, 0])
    return e


# ---------------------------------------------------------------------------
# Stream kernels: one O(n) pass each
# ---------------------------------------------------------------------------


@njit()
def _stream_extent(
    ev_venue: NDArray[np.int8],
    ev_kind: NDArray[np.int8],
    ev_row: NDArray[np.int64],
    n_venues: int,
    extent: NDArray[np.int64],
) -> bool:
    """``extent[v, k]`` = 1 + largest row of the (venue, kind) events; False on a venue, kind
    or row that cannot index any tape."""
    for e in range(ev_venue.shape[0]):
        v = int(ev_venue[e])
        k = int(ev_kind[e])
        r = int(ev_row[e])
        if v < 0 or v >= n_venues or k < 0 or k > 1 or r < 0:
            return False
        if r + 1 > extent[v, k]:
            extent[v, k] = r + 1
    return True


@njit()
def _venue_rows(
    ev_venue: NDArray[np.int8], ev_kind: NDArray[np.int8], ev_row: NDArray[np.int64], venue: int, out: NDArray[np.int64]
) -> None:
    """``out[e]``: last book row of ``venue`` among events ``0..e`` (-1 before the first)."""
    current = -1
    for e in range(ev_venue.shape[0]):
        if ev_venue[e] == venue and ev_kind[e] == BOOK_EVENT:
            current = ev_row[e]
        out[e] = current


@njit()
def _quotes_kernel(
    ev_venue: NDArray[np.int8],
    ev_kind: NDArray[np.int8],
    ev_row: NDArray[np.int64],
    venue: int,
    book: Book,
    tick: float,
    out: NDArray[np.float64],
) -> None:
    """``out[0..6, e]``: bid, ask, bid size, ask size, mid, microprice and spread (bps) of
    the venue's observed row (NaN while it is missing or unusable)."""
    bpx, bsz, apx, asz, _nb, _na, valid = book
    bid = np.nan
    ask = np.nan
    bid_size = np.nan
    ask_size = np.nan
    mid = np.nan
    micro = np.nan
    spread = np.nan
    for e in range(ev_venue.shape[0]):
        if ev_venue[e] == venue and ev_kind[e] == BOOK_EVENT:
            r = ev_row[e]
            if valid[r]:
                bid = float(bpx[r, 0]) * tick
                ask = float(apx[r, 0]) * tick
                bid_size = float(bsz[r, 0])
                ask_size = float(asz[r, 0])
                mid = _mid(bid, ask)
                micro = _micro(bid, ask, bid_size, ask_size)
                spread = (ask - bid) / mid * 1e4 if mid > 0.0 else np.nan
            else:
                bid = np.nan
                ask = np.nan
                bid_size = np.nan
                ask_size = np.nan
                mid = np.nan
                micro = np.nan
                spread = np.nan
        out[0, e] = bid
        out[1, e] = ask
        out[2, e] = bid_size
        out[3, e] = ask_size
        out[4, e] = mid
        out[5, e] = micro
        out[6, e] = spread


@njit()
def _ewma(x: NDArray[np.float64], t: NDArray[np.int64], tau: float, out: NDArray[np.float64]) -> None:
    have = False
    m = 0.0
    t_last = 0
    for e in range(x.shape[0]):
        value = x[e]
        if value == value:
            if have:
                m = _ewma_step(m, value, t[e] - t_last, tau)
            else:
                m = value
                have = True
            t_last = t[e]
        out[e] = m if have else np.nan


@njit()
def _basis_kernel(
    ev_recv: NDArray[np.int64],
    ev_venue: NDArray[np.int8],
    ev_kind: NDArray[np.int8],
    ev_row: NDArray[np.int64],
    leader: int,
    lead_book: Book,
    lead_tick: float,
    follower: int,
    follow_book: Book,
    follow_tick: float,
    micro: bool,
    tau: float,
    out: NDArray[np.float64],
) -> None:
    """``out[e] = b_e - M(t_e^-)`` with ``b`` the leader/follower basis in bps and ``M(t^-)``
    the EWMA state built from the receipt cohorts strictly before ``t``.

    The state steps once per receipt time, when the next cohort begins, with the finished
    cohort's last finite basis (``pending``) and the clock gap between the two cohorts, so
    its sequence is exactly :func:`_ewma` applied to one value per receipt time. Stepping
    per event instead (dt == 0 inside a cohort, weight 0) would enter a cohort's *first*
    value at once and let the cohort's later events subtract part of a jump that the engine,
    deciding after the cohort's last event, is meant to see in full.
    """
    p_lead = np.nan
    p_follow = np.nan
    have = False
    m = 0.0
    t_last = 0
    pending = np.nan
    for e in range(ev_recv.shape[0]):
        if e > 0 and ev_recv[e] != ev_recv[e - 1] and pending == pending:
            # The previous cohort is complete: fold its last finite basis into the state.
            if have:
                m = _ewma_step(m, pending, ev_recv[e - 1] - t_last, tau)
            else:
                m = pending
                have = True
            t_last = ev_recv[e - 1]
            pending = np.nan
        if ev_kind[e] == BOOK_EVENT:
            if ev_venue[e] == leader:
                p_lead = _price_at(lead_book, ev_row[e], lead_tick, micro)
            elif ev_venue[e] == follower:
                p_follow = _price_at(follow_book, ev_row[e], follow_tick, micro)
        value = _rel_bps(p_lead, p_follow)
        out[e] = value - m if have else np.nan
        if value == value:
            pending = value


@njit()
def _return_kernel(
    ev_recv: NDArray[np.int64],
    ev_venue: NDArray[np.int8],
    ev_kind: NDArray[np.int8],
    ev_row: NDArray[np.int64],
    venue: int,
    book: Book,
    tick: float,
    micro: bool,
    window: int,
    out: NDArray[np.float64],
) -> None:
    """Return against the price observed as of receipt time ``t_e - window``: the state after
    the last event ``j`` with ``t_j <= t_e - window`` (``window > 0`` makes ``j < e``). A
    second cursor replays the stream up to ``j``, which only moves forward, so no per-event
    binary search and no price array are needed."""
    n = ev_recv.shape[0]
    now = np.nan
    then = np.nan
    j = -1
    for e in range(n):
        if ev_venue[e] == venue and ev_kind[e] == BOOK_EVENT:
            now = _price_at(book, ev_row[e], tick, micro)
        horizon = ev_recv[e] - window
        while j + 1 < n and ev_recv[j + 1] <= horizon:
            j += 1
            if ev_venue[j] == venue and ev_kind[j] == BOOK_EVENT:
                then = _price_at(book, ev_row[j], tick, micro)
        out[e] = _rel_bps(now, then)


@njit()
def _lead_gap_kernel(
    ev_recv: NDArray[np.int64],
    ev_venue: NDArray[np.int8],
    ev_kind: NDArray[np.int8],
    ev_row: NDArray[np.int64],
    leader: int,
    lead_book: Book,
    lead_tick: float,
    follower: int,
    follow_book: Book,
    follow_tick: float,
    micro: bool,
    window: int,
    out: NDArray[np.float64],
) -> None:
    """Leader minus follower trailing return (bps), both as in :func:`_return_kernel`."""
    n = ev_recv.shape[0]
    lead_now = np.nan
    lead_then = np.nan
    follow_now = np.nan
    follow_then = np.nan
    j = -1
    for e in range(n):
        if ev_kind[e] == BOOK_EVENT:
            if ev_venue[e] == leader:
                lead_now = _price_at(lead_book, ev_row[e], lead_tick, micro)
            elif ev_venue[e] == follower:
                follow_now = _price_at(follow_book, ev_row[e], follow_tick, micro)
        horizon = ev_recv[e] - window
        while j + 1 < n and ev_recv[j + 1] <= horizon:
            j += 1
            if ev_kind[j] == BOOK_EVENT:
                if ev_venue[j] == leader:
                    lead_then = _price_at(lead_book, ev_row[j], lead_tick, micro)
                elif ev_venue[j] == follower:
                    follow_then = _price_at(follow_book, ev_row[j], follow_tick, micro)
        out[e] = _rel_bps(lead_now, lead_then) - _rel_bps(follow_now, follow_then)


@njit()
def _trade_flow_kernel(
    ev_recv: NDArray[np.int64],
    ev_venue: NDArray[np.int8],
    ev_kind: NDArray[np.int8],
    ev_row: NDArray[np.int64],
    venue: int,
    side: NDArray[np.int8],
    size: NDArray[np.int32],
    book: Book,
    normalize: bool,
    tau: float,
    out: NDArray[np.float64],
) -> None:
    """Decayed signed stored quantity from ``venue``'s prints. The sum is held at the time of its
    last print and decayed to each event on output, so rounding compounds per print, not per
    stream event."""
    total = 0.0
    t_ref = 0
    depth = np.nan
    for e in range(ev_recv.shape[0]):
        t = ev_recv[e]
        if ev_venue[e] == venue:
            r = ev_row[e]
            if ev_kind[e] == TRADE_EVENT:
                x = float(side[r]) * float(size[r])
                if x != 0.0:
                    total = _decay(total, t - t_ref, tau) + x
                    t_ref = t
            elif normalize:
                depth = _depth_at(book, r, FLOW_DEPTH_LEVELS)
        value = _decay(total, t - t_ref, tau)
        out[e] = value / depth if normalize else value


@njit()
def _ofi_kernel(
    ev_recv: NDArray[np.int64],
    ev_venue: NDArray[np.int8],
    ev_kind: NDArray[np.int8],
    ev_row: NDArray[np.int64],
    venue: int,
    book: Book,
    tau: float,
    out: NDArray[np.float64],
) -> None:
    """Decayed sum of :func:`_ofi_at` increments over the observed level-1 depth."""
    total = 0.0
    t_ref = 0
    depth = np.nan
    for e in range(ev_recv.shape[0]):
        t = ev_recv[e]
        if ev_venue[e] == venue and ev_kind[e] == BOOK_EVENT:
            r = ev_row[e]
            x = _ofi_at(book, r)
            if x != 0.0:
                total = _decay(total, t - t_ref, tau) + x
                t_ref = t
            depth = _depth_at(book, r, 1)
        out[e] = _decay(total, t - t_ref, tau) / depth


@njit()
def _imbalance_kernel(
    ev_venue: NDArray[np.int8],
    ev_kind: NDArray[np.int8],
    ev_row: NDArray[np.int64],
    venue: int,
    book: Book,
    levels: int,
    out: NDArray[np.float64],
) -> None:
    value = np.nan
    for e in range(ev_venue.shape[0]):
        if ev_venue[e] == venue and ev_kind[e] == BOOK_EVENT:
            value = _imbalance_at(book, ev_row[e], levels)
        out[e] = value


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _frozen(arr: NDArray[Any]) -> NDArray[Any]:
    arr.setflags(write=False)
    return arr


def _positive_ns(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise InvalidConfigError(f"{name} must be an integer number of nanoseconds", context={"got": value})
    ns = int(value)
    if not 0 < ns <= MAX_DURATION_NS:
        raise InvalidConfigError(f"{name} must be in (0, 2**53] nanoseconds", context={"got": ns})
    return ns


def _price_flag(price: object) -> bool:
    """True for the microprice, False for the mid."""
    if not isinstance(price, str) or price not in PRICE_KINDS:
        raise InvalidConfigError("price must be 'mid' or 'micro'", context={"got": price})
    return price == "micro"


def _levels(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= int(value) <= DEPTH:
        raise InvalidConfigError(f"levels must be an integer in [1, {DEPTH}]", context={"got": value})
    return int(value)


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise InvalidConfigError(f"{name} must be a boolean", context={"got": value})
    return bool(value)


def _distinct(leader: int, follower: int) -> None:
    if leader == follower:
        raise InvalidConfigError("leader and follower must be different venues", context={"venue": leader})


# ---------------------------------------------------------------------------
# Evaluation context and features
# ---------------------------------------------------------------------------


class _Context:
    """One evaluation over one prepared market: type and stream checks plus venue lookup."""

    def __init__(self, prep: object) -> None:
        if not isinstance(prep, PreparedMarket):
            raise InvalidConfigError("signals need a PreparedMarket", context={"got": type(prep).__name__})
        self.prep = prep
        self.stream = prep.stream
        self.n = len(prep.stream)
        self.n_venues = len(prep.data)
        self._check_stream()
        self._keys = {spec.key: v for v, spec in enumerate(prep.data.instruments)}

    def _check_stream(self) -> None:
        """Signal kernels require complete tape coverage, even though replay preparation
        accepts genuine subsets. Keep a defensive bounds check before compiled indexing."""
        if _CHECKED.get(id(self.prep)) is self.prep:
            return
        stream, venues = self.prep.stream, self.prep.data.venues
        lengths = np.array([[len(t.book), len(t.trades)] for t in venues], np.int64)
        extent = np.zeros((self.n_venues, 2), np.int64)
        in_range = _stream_extent(stream.venue, stream.kind, stream.row, self.n_venues, extent)
        if not in_range or bool(np.any(extent > lengths)) or self.n != int(lengths.sum()):
            raise InvalidDataError(
                "event stream does not belong to this market data",
                context={"events": self.n, "rows": int(lengths.sum())},
            )
        _CHECKED[id(self.prep)] = self.prep

    def venue(self, ref: object) -> int:
        if isinstance(ref, str):
            v = self._keys.get(ref)
            if v is None:
                raise InvalidConfigError(
                    "instrument is not part of this market", context={"instrument": ref, "known": tuple(self._keys)}
                )
            return v
        if isinstance(ref, bool) or not isinstance(ref, (int, np.integer)) or not 0 <= int(ref) < self.n_venues:
            raise InvalidConfigError(
                "venue must be an instrument key or a venue index", context={"got": ref, "venues": self.n_venues}
            )
        return int(ref)

    def book(self, v: int) -> Book:
        p = self.prep
        return p.bpx[v], p.bsz[v], p.apx[v], p.asz[v], p.nb[v], p.na[v], p.valid[v]

    def tick(self, v: int) -> float:
        return float(self.prep.tick[v])

    def output(self) -> NDArray[np.float64]:
        return np.empty(self.n, np.float64)


def _basis_dev(ctx: _Context, leader: int, follower: int, tau_ns: int, micro: bool) -> NDArray[np.float64]:
    _distinct(leader, follower)
    s, out = ctx.stream, ctx.output()
    _basis_kernel(
        s.recv_ns,
        s.venue,
        s.kind,
        s.row,
        leader,
        ctx.book(leader),
        ctx.tick(leader),
        follower,
        ctx.book(follower),
        ctx.tick(follower),
        micro,
        float(tau_ns),
        out,
    )
    return out


def _trailing(ctx: _Context, v: int, window_ns: int, micro: bool) -> NDArray[np.float64]:
    s, out = ctx.stream, ctx.output()
    _return_kernel(s.recv_ns, s.venue, s.kind, s.row, v, ctx.book(v), ctx.tick(v), micro, window_ns, out)
    return out


def _lead_gap(ctx: _Context, leader: int, follower: int, window_ns: int, micro: bool) -> NDArray[np.float64]:
    _distinct(leader, follower)
    s, out = ctx.stream, ctx.output()
    _lead_gap_kernel(
        s.recv_ns,
        s.venue,
        s.kind,
        s.row,
        leader,
        ctx.book(leader),
        ctx.tick(leader),
        follower,
        ctx.book(follower),
        ctx.tick(follower),
        micro,
        window_ns,
        out,
    )
    return out


def _trade_flow(ctx: _Context, v: int, tau_ns: int, normalize: bool) -> NDArray[np.float64]:
    trades = ctx.prep.data.venues[v].trades
    s, out = ctx.stream, ctx.output()
    _trade_flow_kernel(
        s.recv_ns, s.venue, s.kind, s.row, v, trades.side_label, trades.sz, ctx.book(v), normalize, float(tau_ns), out
    )
    return out


def _ofi(ctx: _Context, v: int, tau_ns: int) -> NDArray[np.float64]:
    s, out = ctx.stream, ctx.output()
    _ofi_kernel(s.recv_ns, s.venue, s.kind, s.row, v, ctx.book(v), float(tau_ns), out)
    return out


def _imbalance(ctx: _Context, v: int, levels: int) -> NDArray[np.float64]:
    s, out = ctx.stream, ctx.output()
    _imbalance_kernel(s.venue, s.kind, s.row, v, ctx.book(v), levels, out)
    return out


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TopOfBook:
    """Observed level-1 state, each field ``(n_venues, n_events)`` float64 and read-only.

    Prices in quote currency, sizes in stored quantity units; NaN wherever the venue's observed book is missing or
    unusable.
    """

    bid: NDArray[np.float64]
    ask: NDArray[np.float64]
    bid_size: NDArray[np.float64]
    ask_size: NDArray[np.float64]
    mid: NDArray[np.float64]
    microprice: NDArray[np.float64]
    """``(bid * ask_size + ask * bid_size) / (bid_size + ask_size)``."""
    spread_bps: NDArray[np.float64]
    """``(ask - bid) / mid * 1e4``."""

    def __post_init__(self) -> None:
        shape = getattr(self.bid, "shape", None)
        for f in fields(self):
            arr = getattr(self, f.name)
            if not isinstance(arr, np.ndarray) or arr.dtype != np.float64 or arr.ndim != 2 or arr.shape != shape:
                raise InvalidDataError(
                    "top-of-book fields must be float64 arrays of one 2-D shape", context={"field": f.name}
                )
            arr.setflags(write=False)


_QUOTE_FIELDS: Final = tuple(f.name for f in fields(TopOfBook))


def observed_rows(prep: PreparedMarket) -> NDArray[np.int64]:
    """``(n_venues, n_events)``: last book row of each venue at or before each event (-1 before
    its first book row). Clear and invalid rows are observed too; consumers check validity."""
    ctx = _Context(prep)
    out = np.empty((ctx.n_venues, ctx.n), np.int64)
    s = ctx.stream
    for v in range(ctx.n_venues):
        _venue_rows(s.venue, s.kind, s.row, v, out[v])
    return _frozen(out)


def top_of_book(prep: PreparedMarket) -> TopOfBook:
    """Level-1 bid/ask, sizes, mid, microprice and spread of the observed book per event.

    Materialises seven ``(n_venues, n_events)`` arrays (about 1.8 GB for a two-venue day of
    16M events); the signal kernels never need it.
    """
    ctx = _Context(prep)
    out = np.empty((len(_QUOTE_FIELDS), ctx.n_venues, ctx.n), np.float64)
    s = ctx.stream
    for v in range(ctx.n_venues):
        _quotes_kernel(s.venue, s.kind, s.row, v, ctx.book(v), ctx.tick(v), out[:, v])
    return TopOfBook(*(out[f] for f in range(len(_QUOTE_FIELDS))))


def ewma_time(values: ArrayLike, t_ns: ArrayLike, tau_ns: int) -> NDArray[np.float64]:
    """Irregular-time EWMA: ``m_e = w m + (1 - w) x_e`` with ``w = exp(-(t_e - t_last) / tau)``.

    ``t_last`` is the time of the last *finite* input: NaN inputs are skipped and the state is
    held (and reported). The output is NaN until the first finite value, which initialises
    the state. A value arriving with ``dt == 0`` gets weight ``1 - exp(0) = 0``.
    """
    tau = _positive_ns("tau_ns", tau_ns)
    try:
        x = np.ascontiguousarray(values, dtype=np.float64)
        t = np.asarray(t_ns)
    except (TypeError, ValueError) as exc:
        raise InvalidDataError("values must be numeric and t_ns integer nanoseconds") from exc
    if t.dtype.kind not in "iu" or not np.can_cast(t.dtype, np.int64):
        raise InvalidDataError("t_ns must be integer nanoseconds", context={"dtype": str(t.dtype)})
    t = np.ascontiguousarray(t, dtype=np.int64)
    if x.ndim != 1 or t.shape != x.shape:
        raise InvalidDataError(
            "values and t_ns must be 1-D arrays of one length", context={"values": x.shape, "t_ns": t.shape}
        )
    if bool(np.isinf(x).any()):
        raise InvalidDataError("values must be finite or NaN")
    if x.shape[0] > 1 and bool(np.any(t[1:] < t[:-1])):
        raise InvalidDataError("t_ns must be non-decreasing")
    out = np.empty(x.shape[0], np.float64)
    _ewma(x, t, float(tau), out)
    return _frozen(out)


def basis_deviation_bps(
    prep: PreparedMarket, leader: VenueRef, follower: VenueRef, tau_ns: int, price: str = "mid"
) -> NDArray[np.float64]:
    """Deviation of the leader/follower basis from its own recent average, in bps.

    ``b_e = (P_leader / P_follower - 1) * 1e4`` and ``g_e = b_e - M(t_e^-)``, where ``M`` is
    the :func:`ewma_time` of one basis per receipt time (a cohort's last finite ``b``, at the
    cohort's clock) read *before* ``e``'s own cohort. The average therefore excludes ``b_e``
    and every value received together with it: the replay engine decides after a cohort's
    last event, and a jump that arrives with other events in its cohort (a print, the other
    venue's book) is reported in full there instead of being partly absorbed. (Up to
    :data:`SPEC_VERSION` 1 the average ran over earlier *events*, so a cohort's first value
    entered it immediately and a same-cohort event saw ``exp(-dt/tau)`` of the jump only.)
    Positive ``g``: the leader is rich against its recent relation, i.e. the follower is
    expected to rise. NaN while either price is unavailable and before the first basis.
    """
    ctx = _Context(prep)
    return _frozen(
        _basis_dev(ctx, ctx.venue(leader), ctx.venue(follower), _positive_ns("tau_ns", tau_ns), _price_flag(price))
    )


def trailing_return_bps(
    prep: PreparedMarket, venue: VenueRef, window_ns: int, price: str = "mid"
) -> NDArray[np.float64]:
    """``(P_e / P(as of t_e - window) - 1) * 1e4`` with ``P`` the venue's observed price.

    "As of" is the state after the last stream event with ``recv_ns <= t_e - window``
    (``searchsorted(..., side="right") - 1``); NaN if there is none or either price is
    unavailable.
    """
    ctx = _Context(prep)
    return _frozen(_trailing(ctx, ctx.venue(venue), _positive_ns("window_ns", window_ns), _price_flag(price)))


def lead_gap_bps(
    prep: PreparedMarket, leader: VenueRef, follower: VenueRef, window_ns: int, price: str = "mid"
) -> NDArray[np.float64]:
    """Trailing return of the leader minus that of the follower (bps, same window)."""
    ctx = _Context(prep)
    return _frozen(
        _lead_gap(ctx, ctx.venue(leader), ctx.venue(follower), _positive_ns("window_ns", window_ns), _price_flag(price))
    )


def trade_flow(prep: PreparedMarket, venue: VenueRef, tau_ns: int, normalize: bool = False) -> NDArray[np.float64]:
    """Time-decayed signed aggressor flow ``sum side_label * sz * exp(-(t_e - t_j) / tau)``.

    stored quantity units, decayed to every event; exactly 0 before the venue's first print. With
    ``normalize`` it is divided by the venue's observed top-5 depth (bid + ask stored quantity units) and is
    NaN while that book is unusable.
    """
    ctx = _Context(prep)
    return _frozen(
        _trade_flow(ctx, ctx.venue(venue), _positive_ns("tau_ns", tau_ns), _strict_bool("normalize", normalize))
    )


def ofi(prep: PreparedMarket, venue: VenueRef, tau_ns: int) -> NDArray[np.float64]:
    """Time-decayed level-1 order-flow imbalance over observed top-1 depth.

    Increments at the venue's book events (Cont, Kukanov and Stoikov 2014)::

        e_n = 1{b_n >= b_n-1} q^b_n - 1{b_n <= b_n-1} q^b_n-1
              - 1{a_n <= a_n-1} q^a_n + 1{a_n >= a_n-1} q^a_n-1

    are summed with weight ``exp(-(t_e - t_n) / tau)`` and divided by the observed level-1
    bid + ask size; pairs touching an unusable row contribute nothing.
    """
    ctx = _Context(prep)
    return _frozen(_ofi(ctx, ctx.venue(venue), _positive_ns("tau_ns", tau_ns)))


def imbalance(prep: PreparedMarket, venue: VenueRef, levels: int) -> NDArray[np.float64]:
    """``(bid size - ask size) / (bid size + ask size)`` over the top ``levels`` observed levels."""
    ctx = _Context(prep)
    return _frozen(_imbalance(ctx, ctx.venue(venue), _levels(levels)))


# ---------------------------------------------------------------------------
# Spec registry
# ---------------------------------------------------------------------------

_SPEC_FIELDS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "basis_dev": ("leader", "follower", "tau_ms", "price"),
        "lead_gap": ("leader", "follower", "window_ms"),
        "trade_flow": ("venue", "tau_ms", "normalize"),
        "ofi": ("venue", "tau_ms"),
        "imbalance": ("venue", "levels"),
        "return": ("venue", "window_ms"),
        "linear": ("terms",),
    }
)


def _ms_to_ns(name: str, ms: int | float) -> int:
    """Exact milliseconds -> nanoseconds; a float is read through its shortest repr, so
    ``0.1`` ms is exactly 100,000 ns. Sub-nanosecond durations are rejected, not rounded."""
    exact = Decimal(ms) if isinstance(ms, int) else Decimal(repr(float(ms)))
    ns = exact * NS_PER_MS
    if ns != ns.to_integral_value() or not 0 < ns <= MAX_DURATION_NS:
        raise InvalidConfigError(f"{name} must be a positive whole number of nanoseconds <= 2**53", context={"ms": ms})
    return int(ns)


def _norm_instrument(name: str, value: object) -> str:
    """An instrument key ``venue/symbol``; whitespace anywhere (including around the key) is
    rejected rather than normalised, so one instrument has one spelling and one spec_id."""
    if isinstance(value, str):
        venue, sep, symbol = value.partition("/")
        if venue and sep and symbol and not any(c.isspace() for c in value):
            return value
    raise InvalidConfigError(f"{name} must be an instrument key 'venue/symbol'", context={"got": value})


def _norm_duration_ms(name: str, value: object) -> int | float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise InvalidConfigError(f"{name} must be a number of milliseconds", context={"got": value})
    number: int | float
    if isinstance(value, (int, np.integer)):
        number = int(value)
    else:
        number = float(value)
        if not math.isfinite(number):
            raise InvalidConfigError(f"{name} must be finite", context={"got": value})
        if number.is_integer():
            number = int(number)  # 100.0 and 100 are one spec (and one spec_id)
    _ms_to_ns(name, number)
    return number


def _norm_terms(value: object, depth: int) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise InvalidConfigError("linear terms must be a non-empty list", context={"got": type(value).__name__})
    terms: list[dict[str, Any]] = []
    for i, term in enumerate(value):
        if not isinstance(term, Mapping) or set(term) != {"spec", "weight"}:
            raise InvalidConfigError("a linear term is a mapping with exactly 'spec' and 'weight'", context={"term": i})
        weight = term["weight"]
        if isinstance(weight, (bool, np.bool_)) or not isinstance(weight, (int, float, np.integer, np.floating)):
            raise InvalidConfigError("term weight must be a number", context={"term": i})
        try:
            w = float(weight)
        except OverflowError:
            w = math.inf
        if not math.isfinite(w):
            raise InvalidConfigError("term weight must be finite", context={"term": i})
        terms.append({"spec": _canonical(term["spec"], depth + 1), "weight": w})
    return terms


def _norm_field(name: str, value: object, depth: int) -> object:
    if name in ("leader", "follower", "venue"):
        return _norm_instrument(name, value)
    if name in ("tau_ms", "window_ms"):
        return _norm_duration_ms(name, value)
    if name == "price":
        _price_flag(value)
        return value
    if name == "normalize":
        return _strict_bool(name, value)
    if name == "levels":
        return _levels(value)
    return _norm_terms(value, depth)


def _canonical(spec: object, depth: int) -> dict[str, Any]:
    if depth > MAX_SPEC_NESTING:
        raise InvalidConfigError("signal spec nests too deeply", context={"max": MAX_SPEC_NESTING})
    if not isinstance(spec, Mapping):
        raise InvalidConfigError("a signal spec must be a mapping", context={"got": type(spec).__name__})
    kind = spec.get("kind")
    names = _SPEC_FIELDS.get(kind) if isinstance(kind, str) else None
    if names is None:
        raise InvalidConfigError("unknown signal kind", context={"kind": kind, "known": SIGNAL_KINDS})
    expected = {"kind", *names}
    keys = set(spec)
    if keys != expected:
        raise InvalidConfigError(
            "signal spec keys do not match its kind",
            context={
                "kind": kind,
                "missing": sorted(expected - keys),
                "unexpected": sorted(repr(k) for k in keys - expected),
            },
        )
    out: dict[str, Any] = {"kind": kind}
    for name in names:
        out[name] = _norm_field(name, spec[name], depth)
    if "leader" in out and out["leader"] == out["follower"]:
        raise InvalidConfigError(
            "leader and follower must be different instruments", context={"instrument": out["leader"]}
        )
    return out


def canonical_spec(spec: Mapping[str, object]) -> dict[str, Any]:
    """Validated, normalised copy of a signal spec: plain JSON types, integral milliseconds as
    ``int``, weights as ``float``, terms as a list. Raises :class:`InvalidConfigError` on an
    unknown kind, a missing or extra key, or an out-of-range value."""
    return _canonical(spec, 0)


def spec_id(spec: Mapping[str, object]) -> str:
    """First 16 hex digits of the SHA-256 of the canonical JSON of the validated spec.

    Equivalent spellings (key order, ``100`` vs ``100.0`` ms, tuple vs list) share an id; the
    order of linear terms is kept because it fixes the floating-point summation order. The
    hashed document also names :data:`SPEC_SCHEMA` and :data:`SPEC_VERSION`, so a change of
    kernel semantics (which bumps the version) retires every cached array.
    """
    return sha256_json({"schema": SPEC_SCHEMA, "spec": canonical_spec(spec), "version": SPEC_VERSION})[:16]


type Builder = Callable[[_Context, Mapping[str, Any]], NDArray[np.float64]]


def _build_basis(ctx: _Context, spec: Mapping[str, Any]) -> NDArray[np.float64]:
    return _basis_dev(
        ctx,
        ctx.venue(spec["leader"]),
        ctx.venue(spec["follower"]),
        _ms_to_ns("tau_ms", spec["tau_ms"]),
        spec["price"] == "micro",
    )


def _build_lead_gap(ctx: _Context, spec: Mapping[str, Any]) -> NDArray[np.float64]:
    return _lead_gap(
        ctx, ctx.venue(spec["leader"]), ctx.venue(spec["follower"]), _ms_to_ns("window_ms", spec["window_ms"]), False
    )


def _build_flow(ctx: _Context, spec: Mapping[str, Any]) -> NDArray[np.float64]:
    return _trade_flow(ctx, ctx.venue(spec["venue"]), _ms_to_ns("tau_ms", spec["tau_ms"]), spec["normalize"])


def _build_ofi(ctx: _Context, spec: Mapping[str, Any]) -> NDArray[np.float64]:
    return _ofi(ctx, ctx.venue(spec["venue"]), _ms_to_ns("tau_ms", spec["tau_ms"]))


def _build_imbalance(ctx: _Context, spec: Mapping[str, Any]) -> NDArray[np.float64]:
    return _imbalance(ctx, ctx.venue(spec["venue"]), spec["levels"])


def _build_return(ctx: _Context, spec: Mapping[str, Any]) -> NDArray[np.float64]:
    return _trailing(ctx, ctx.venue(spec["venue"]), _ms_to_ns("window_ms", spec["window_ms"]), False)


def _build_linear(ctx: _Context, spec: Mapping[str, Any]) -> NDArray[np.float64]:
    """Weighted sum in the listed order; NaN wherever any term is NaN (even at weight 0).
    Terms are scaled and accumulated in place, so at most two full arrays are alive."""
    first, *rest = spec["terms"]
    total = _evaluate(ctx, first["spec"])
    total *= float(first["weight"])
    for term in rest:
        part = _evaluate(ctx, term["spec"])
        part *= float(term["weight"])
        total += part
    return total


_BUILDERS: Final[Mapping[str, Builder]] = MappingProxyType(
    {
        "basis_dev": _build_basis,
        "lead_gap": _build_lead_gap,
        "trade_flow": _build_flow,
        "ofi": _build_ofi,
        "imbalance": _build_imbalance,
        "return": _build_return,
        "linear": _build_linear,
    }
)


def _evaluate(ctx: _Context, spec: Mapping[str, Any]) -> NDArray[np.float64]:
    return _BUILDERS[spec["kind"]](ctx, spec)


def build_signal(prep: PreparedMarket, spec: Mapping[str, object]) -> NDArray[np.float64]:
    """Evaluate a signal spec into one read-only float64 per stream event.

    Kinds and their (all required) keys::

        basis_dev   leader, follower, tau_ms, price ("mid" | "micro")
        lead_gap    leader, follower, window_ms        (mid prices)
        trade_flow  venue, tau_ms, normalize
        ofi         venue, tau_ms
        imbalance   venue, levels
        return      venue, window_ms                   (mid price)
        linear      terms: [{spec, weight}, ...]       (weighted sum, NaN if any term is)

    Venues are instrument keys such as ``"venue_a/ALPHA-USD"``. Unknown kinds, missing or extra
    keys and invalid values raise :class:`InvalidConfigError`.
    """
    canonical = canonical_spec(spec)
    return _frozen(_evaluate(_Context(prep), canonical))


__all__ = [
    "FLOW_DEPTH_LEVELS",
    "MAX_DURATION_NS",
    "MAX_SPEC_NESTING",
    "PRICE_KINDS",
    "SIGNAL_KINDS",
    "SPEC_SCHEMA",
    "SPEC_VERSION",
    "Book",
    "TopOfBook",
    "VenueRef",
    "basis_deviation_bps",
    "build_signal",
    "canonical_spec",
    "ewma_time",
    "imbalance",
    "lead_gap_bps",
    "observed_rows",
    "ofi",
    "spec_id",
    "top_of_book",
    "trade_flow",
    "trailing_return_bps",
]
