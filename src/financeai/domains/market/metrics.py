"""Ledger-derived paths and account metrics for replay results (numerical; no plotting).

Everything here is computed from the *complete*, ordered fill records of a
:class:`~financeai.domains.market.engine.ReplayResult` before any display reduction, so a
figure can only ever downsample numbers that already reconcile with the ledger.

Conventions:

* Paths are **post-event step functions**: element ``i`` is the value immediately after
  event ``i`` and stays in force until the next element; every path is flat/zero before
  its first element. Paths keep one element per fill record (the levels of one sweep share
  a timestamp), so their extremes are exactly the ones the ledger checks against the cap.
* True inventory and example cash change at fill **arrival** (the simulated exchange time
  on the local clock axis); the controller only learns them at **acknowledgement**.
* ``C`` is cumulative instrument-specific fill cash. It is not profit while inventory remains open.
* ``M`` adds marked inventory to cash: ``q * factor * mid`` for linear/spot instruments,
  or ``-q * factor / mid`` for inverse contracts. Prices come from the last received valid book.
  These marks omit liquidation costs and become NaN when a held instrument lacks a valid quote.
* Scalar aggregates require a common settlement currency. No automatic currency conversion occurs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from financeai.core.clock import MISSING_NS, NS_PER_S
from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.core.jit import njit
from financeai.domains.market.engine import PreparedMarket, ReplayResult
from financeai.domains.market.ledger import Fills
from financeai.domains.market.scenario import RiskLimits, Scenario

_NO_TIME: Final = np.iinfo(np.int64).max

RECONCILE_ABS_TOL: Final = 1e-12
"""Absolute tolerance when float paths are compared with the ledger's ``math.fsum`` totals."""
RECONCILE_REL_TOL: Final = 1e-12
"""Relative tolerance for the same comparisons (compensated sums differ by a few ulps)."""


def _frozen[S: np.generic](arr: NDArray[S]) -> NDArray[S]:
    arr.setflags(write=False)
    return arr


def _venue_count(result: ReplayResult) -> int:
    return len(result.settlement.final_position_units)


def _check_venue(result: ReplayResult, venue: int) -> int:
    n = _venue_count(result)
    if isinstance(venue, bool) or not isinstance(venue, int | np.integer) or not 0 <= venue < n:
        raise InvalidConfigError("venue index out of range", context={"venue": venue, "venues": n})
    return int(venue)


def _ordered_fills(result: ReplayResult) -> Fills:
    """The fill records, checked to be in true (arrival) order with causal clocks."""
    fills = result.fills
    if len(fills) > 1 and bool(np.any(np.diff(fills.arrival_ns) < 0)):
        raise InvalidDataError("fill records are not in exchange-arrival order")
    if len(fills) and (
        bool(np.any(fills.arrival_ns < fills.submit_ns)) or bool(np.any(fills.ack_ns < fills.arrival_ns))
    ):
        raise InvalidDataError("fill clocks violate submit <= arrival <= acknowledgement")
    n_venues = _venue_count(result)
    if len(fills) and (int(fills.venue.min()) < 0 or int(fills.venue.max()) >= n_venues):
        raise InvalidDataError("fill venue index out of range")
    return fills


@njit()
def _compensated_cumsum(values: NDArray[np.float64], out: NDArray[np.float64]) -> None:
    """Running sums with Neumaier compensation (error ~1 ulp of each partial sum)."""
    s = 0.0
    comp = 0.0
    for i in range(values.shape[0]):
        x = values[i]
        t = s + x
        if abs(s) >= abs(x):
            comp += (s - t) + x
        else:
            comp += (x - t) + s
        s = t
        out[i] = s + comp


def _cumulative_cash(values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.empty(values.shape[0], np.float64)
    _compensated_cumsum(np.ascontiguousarray(values, dtype=np.float64), out)
    return out


def _signed_qty(fills: Fills) -> NDArray[np.int64]:
    return fills.side.astype(np.int64) * fills.qty_units


def position_path(result: ReplayResult, venue: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """True inventory of ``venue`` (signed stored quantity units) as a post-event step at fill arrival."""
    v = _check_venue(result, venue)
    fills = _ordered_fills(result)
    mask = fills.venue == v
    times = np.ascontiguousarray(fills.arrival_ns[mask])
    return _frozen(times), _frozen(np.cumsum(_signed_qty(fills)[mask]))


def known_position_path(result: ReplayResult, venue: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Inventory as the controller knows it: the same fills, effective at acknowledgement.

    Records are stably sorted by ``ack_ns``; ties keep record (arrival) order.
    """
    v = _check_venue(result, venue)
    fills = _ordered_fills(result)
    mask = fills.venue == v
    acks = fills.ack_ns[mask]
    order = np.argsort(acks, kind="stable")
    times = np.ascontiguousarray(acks[order])
    return _frozen(times), _frozen(np.cumsum(_signed_qty(fills)[mask][order]))


def cash_path(result: ReplayResult, venue: int | None = None) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Cumulative example cash ``C`` in settlement currency at fill arrival (all venues when ``venue`` is None)."""
    fills = _ordered_fills(result)
    if venue is None:
        if len(set(result.settlement.currencies)) > 1:
            raise InvalidConfigError("aggregate cash paths require a common settlement currency")
        times = np.ascontiguousarray(fills.arrival_ns)
        cash = fills.cash
    else:
        mask = fills.venue == _check_venue(result, venue)
        times = np.ascontiguousarray(fills.arrival_ns[mask])
        cash = fills.cash[mask]
    return _frozen(times), _frozen(_cumulative_cash(cash))


# ---------------------------------------------------------------------------
# Observed-mid mark M
# ---------------------------------------------------------------------------


def _check_market(result: ReplayResult, prep: PreparedMarket) -> int:
    n = _venue_count(result)
    if len(prep.data) != n:
        raise InvalidConfigError(
            "prepared market does not match the replayed venues",
            context={"result_venues": n, "market_venues": len(prep.data)},
        )
    if len({spec.settlement_currency for spec in prep.data.instruments}) > 1:
        raise InvalidConfigError(
            "aggregate metrics require a common settlement currency; inspect each currency separately"
        )
    return n


def _clock_array(name: str, values: ArrayLike) -> NDArray[np.int64]:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.dtype.kind not in "iu" or not np.can_cast(arr.dtype, np.int64, "safe"):
        raise InvalidDataError(
            f"{name} must be a 1-D integer array of epoch nanoseconds",
            context={"dtype": str(arr.dtype), "shape": arr.shape},
        )
    return np.ascontiguousarray(arr, dtype=np.int64)


_Ticks = tuple[NDArray[np.int32], ...]
_Clocks = tuple[NDArray[np.int64], ...]
_Flags = tuple[NDArray[np.bool_], ...]


@njit()
def _mark_at(
    c: float,
    q: NDArray[np.int64],
    obs: NDArray[np.int64],
    bpx: _Ticks,
    apx: _Ticks,
    valid: _Flags,
    tick: NDArray[np.float64],
    inverse: NDArray[np.bool_],
    factor: NDArray[np.float64],
) -> float:
    """``C - sum_v q_v / mid_v`` over held venues; NaN when a held venue has no valid book."""
    inv = 0.0
    for v in range(q.shape[0]):
        if q[v] != 0:
            r = obs[v]
            if r < 0 or not valid[v][r]:
                return np.nan
            mid = 0.5 * (float(bpx[v][r, 0]) + float(apx[v][r, 0])) * tick[v]
            inv += q[v] * factor[v] / mid if inverse[v] else -q[v] * factor[v] * mid
    return float(c - inv)


@njit()
def _mark_path_kernel(
    f_arrival: NDArray[np.int64],
    f_venue: NDArray[np.int8],
    f_signed: NDArray[np.int64],
    cum_c: NDArray[np.float64],
    book_recv: _Clocks,
    bpx: _Ticks,
    apx: _Ticks,
    valid: _Flags,
    tick: NDArray[np.float64],
    inverse: NDArray[np.bool_],
    factor: NDArray[np.float64],
    out_t: NDArray[np.int64],
    out_m: NDArray[np.float64],
    write: bool,
) -> int:
    """Walk fills and book receipts in time order; emit M after each change point.

    A timestamp is a change point when a fill arrives or a held venue receives a book.
    While the account is flat M equals the (constant) cash, so the walk jumps straight to
    the next fill. Returns the number of change points (written only when ``write``).
    """
    nv = tick.shape[0]
    n_f = f_arrival.shape[0]
    q = np.zeros(nv, np.int64)
    pos = np.zeros(nv, np.int64)
    obs = np.full(nv, -1, np.int64)
    c = 0.0
    i = 0
    n_out = 0
    while True:
        t = _NO_TIME
        if i < n_f:
            t = f_arrival[i]
        for v in range(nv):
            if pos[v] < book_recv[v].shape[0] and book_recv[v][pos[v]] < t:
                t = book_recv[v][pos[v]]
        if t == _NO_TIME:
            break
        changed = False
        while i < n_f and f_arrival[i] == t:
            q[f_venue[i]] += f_signed[i]
            c = cum_c[i]
            i += 1
            changed = True
        held = False
        for v in range(nv):
            start = pos[v]
            while pos[v] < book_recv[v].shape[0] and book_recv[v][pos[v]] == t:
                pos[v] += 1
            if pos[v] > start:
                obs[v] = pos[v] - 1
                if q[v] != 0:
                    changed = True
            if q[v] != 0:
                held = True
        if changed:
            if write:
                out_t[n_out] = t
                out_m[n_out] = _mark_at(c, q, obs, bpx, apx, valid, tick, inverse, factor)
            n_out += 1
        if not held:
            if i >= n_f:
                break
            nxt = f_arrival[i]
            for v in range(nv):
                p = np.searchsorted(book_recv[v], nxt, side="left")
                if p > pos[v]:
                    obs[v] = p - 1
                    pos[v] = p
    return n_out


def marked_value(result: ReplayResult, prep: PreparedMarket, grid_ns: ArrayLike) -> NDArray[np.float64]:
    """``M(t) = C(t) - sum_v q_v(t) / mid_v(t)`` in settlement currency at each grid time (any order).

    ``C`` and ``q`` are as of fill arrival (``arrival_ns <= t``); ``mid_v`` is the mid of the
    last book *received* at or before ``t``. NaN where a held venue has no valid book.
    """
    _check_market(result, prep)
    fills = _ordered_fills(result)
    grid = _clock_array("grid_ns", grid_ns)
    n = len(fills)
    k = np.searchsorted(fills.arrival_ns, grid, side="right") - 1
    seen = k >= 0
    kk = np.maximum(k, 0)
    cash = np.zeros(grid.shape[0], np.float64)
    if n:
        cash = np.where(seen, _cumulative_cash(fills.cash)[kk], 0.0)
    signed = _signed_qty(fills)
    inv = np.zeros(grid.shape[0], np.float64)
    unmarked = np.zeros(grid.shape[0], np.bool_)
    for v in range(len(prep.data)):
        q = np.zeros(grid.shape[0], np.int64)
        if n:
            q = np.where(seen, np.cumsum(np.where(fills.venue == v, signed, 0))[kk], 0)
        recv = prep.book_recv[v]
        row = np.searchsorted(recv, grid, side="right") - 1
        rr = np.maximum(row, 0)
        if recv.shape[0]:
            ok = (row >= 0) & prep.valid[v][rr]
            mid = 0.5 * (prep.bpx[v][rr, 0].astype(np.float64) + prep.apx[v][rr, 0].astype(np.float64)) * prep.tick[v]
        else:
            ok = np.zeros(grid.shape[0], np.bool_)
            mid = np.ones(grid.shape[0], np.float64)
        held = q != 0
        priced = held & ok
        if prep.inverse[v]:
            inv = inv + np.divide(q * prep.cash_factor[v], mid, out=np.zeros(grid.shape[0], np.float64), where=priced)
        else:
            inv = inv - np.where(priced, q * prep.cash_factor[v] * mid, 0.0)
        unmarked |= held & ~ok
    out = cash - inv
    out[unmarked] = np.nan
    return _frozen(out)


def marked_value_path(result: ReplayResult, prep: PreparedMarket) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """M on its full event grid: one post-event value per change point (strictly increasing).

    Change points are fill arrivals and book receipts on venues held at that time; between
    them M is constant, so this path *is* the step function M(t) (zero before the first fill).
    """
    _check_market(result, prep)
    fills = _ordered_fills(result)
    args = (
        np.ascontiguousarray(fills.arrival_ns),
        np.ascontiguousarray(fills.venue),
        _signed_qty(fills),
        _cumulative_cash(fills.cash),
        prep.book_recv,
        prep.bpx,
        prep.apx,
        prep.valid,
        np.ascontiguousarray(prep.tick, dtype=np.float64),
        prep.inverse,
        prep.cash_factor,
    )
    n = int(_mark_path_kernel(*args, np.zeros(0, np.int64), np.zeros(0, np.float64), False))
    times = np.empty(n, np.int64)
    values = np.empty(n, np.float64)
    _mark_path_kernel(*args, times, values, True)  # same walk, now writing
    return _frozen(times), _frozen(values)


# ---------------------------------------------------------------------------
# Drawdown of M
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Drawdown:
    """Largest fall of M below its running peak, measured from the flat start (M = 0)."""

    max_pnl: float
    """Non-negative; ``0.0`` when M never fell below a previous peak."""
    peak_ns: int
    """Time the preceding peak was (last) set; ``MISSING_NS`` when it is the flat start."""
    trough_ns: int
    """Time of the deepest point; ``MISSING_NS`` without a drawdown."""
    unmarked_points: int
    """Path points skipped because a held venue had no valid mark (M is NaN there)."""


def _finite_path(times_ns: ArrayLike, marked_pnl: ArrayLike) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    times = _clock_array("times_ns", times_ns)
    m = np.asarray(marked_pnl, dtype=np.float64)
    if m.shape != times.shape:
        raise InvalidDataError(
            "times and values must have the same 1-D shape", context={"times": times.shape, "values": m.shape}
        )
    return times, m


def drawdown(times_ns: ArrayLike, marked_pnl: ArrayLike) -> Drawdown:
    """Maximum drawdown of a marked-value path (NaN instants are skipped and counted)."""
    times, m = _finite_path(times_ns, marked_pnl)
    finite = np.isfinite(m)
    unmarked = int(m.shape[0] - np.count_nonzero(finite))
    mf, tf = m[finite], times[finite]
    if mf.shape[0] == 0:
        return Drawdown(0.0, MISSING_NS, MISSING_NS, unmarked)
    peaks = np.maximum.accumulate(np.maximum(mf, 0.0))
    fall = peaks - mf
    k = int(np.argmax(fall))
    if not fall[k] > 0.0:
        return Drawdown(0.0, MISSING_NS, MISSING_NS, unmarked)
    hits = np.flatnonzero(mf[: k + 1] == peaks[k])
    peak_ns = int(tf[hits[-1]]) if hits.shape[0] else MISSING_NS
    return Drawdown(float(fall[k]), peak_ns, int(tf[k]), unmarked)


def drawdown_path(marked_pnl: ArrayLike) -> NDArray[np.float64]:
    """``M(t) - max(0, max_{s<=t} M(s))`` (<= 0) per point; NaN where M is NaN."""
    m = np.asarray(marked_pnl, dtype=np.float64)
    if m.ndim != 1:
        raise InvalidDataError("marked values must be 1-D", context={"shape": m.shape})
    out = np.full(m.shape[0], np.nan)
    finite = np.isfinite(m)
    mf = m[finite]
    out[finite] = mf - np.maximum.accumulate(np.maximum(mf, 0.0))
    return _frozen(out)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


@njit()
def _segment_sums(
    values: NDArray[np.float64], lo: NDArray[np.int64], hi: NDArray[np.int64], out: NDArray[np.float64]
) -> None:
    """Neumaier-compensated sum of ``values[lo[k]:hi[k]]`` into ``out[k]``."""
    for k in range(lo.shape[0]):
        s = 0.0
        comp = 0.0
        for i in range(lo[k], hi[k]):
            x = values[i]
            t = s + x
            if abs(s) >= abs(x):
                comp += (s - t) + x
            else:
                comp += (x - t) + s
            s = t
        out[k] = s + comp


def _sums(values: NDArray[np.float64], lo: NDArray[np.int64], hi: NDArray[np.int64]) -> NDArray[np.float64]:
    out = np.empty(lo.shape[0], np.float64)
    _segment_sums(np.ascontiguousarray(values), lo, hi, out)
    return out


@dataclass(frozen=True, slots=True)
class Episodes:
    """Per-venue exposure episodes (flat -> flat round trips), in chronological order.

    An episode starts with the fill that leaves flat and ends with the fill that returns to
    exactly zero; a sign flip without touching zero stays inside one episode. An episode
    still open at the end has ``closed=False``, ``exit_ns``/``holding_ns`` = ``MISSING_NS``
    and NaN gross/net: unresolved inventory has no payoff (it is not a zero). Its fees are
    already paid and stay finite. ``gross - fees = net`` (to rounding), all in settlement currency.
    """

    venue: NDArray[np.int8]
    direction: NDArray[np.int8]
    """Sign of the opening position (+1 long, -1 short)."""
    entry_ns: NDArray[np.int64]
    exit_ns: NDArray[np.int64]
    holding_ns: NDArray[np.int64]
    n_fills: NDArray[np.int64]
    max_abs_units: NDArray[np.int64]
    gross_pnl: NDArray[np.float64]
    """``sum(side * qty / price)``: the inverse-contract result before fees."""
    fees: NDArray[np.float64]
    net_pnl: NDArray[np.float64]
    """``sum(cash)``: the example-score contribution of the round trip."""
    closed: NDArray[np.bool_]

    def __post_init__(self) -> None:
        n = self.venue.shape[0]
        for f in fields(self):
            arr = getattr(self, f.name)
            if arr.ndim != 1 or arr.shape[0] != n:
                raise InvalidDataError(
                    "episode column has the wrong shape", context={"column": f.name, "shape": arr.shape}
                )
            arr.setflags(write=False)

    def __len__(self) -> int:
        return int(self.venue.shape[0])


def episodes(result: ReplayResult, prep: PreparedMarket) -> Episodes:
    """Split every venue's fills into flat -> flat round trips (``prep`` supplies tick sizes).

    Fees are the per-fill difference between the fee-free reciprocal flow
    ``side * qty / price`` and the ledger's ``cash``, so no fee schedule is re-read.
    """
    _check_market(result, prep)
    fills = _ordered_fills(result)
    signed = _signed_qty(fills)
    price = fills.px_ticks.astype(np.float64) * np.asarray(prep.tick, np.float64)[fills.venue]
    exposure = signed.astype(np.float64) * prep.cash_factor[fills.venue]
    gross_fill = np.where(prep.inverse[fills.venue], exposure / price, -exposure * price)
    fee_fill = gross_fill - fills.cash
    parts: list[dict[str, NDArray[Any]]] = []
    for v in range(len(prep.data)):
        idx = np.flatnonzero(fills.venue == v)
        if idx.shape[0] == 0:
            continue
        sv = signed[idx]
        q = np.cumsum(sv)
        starts = np.flatnonzero(q == sv)  # previous position was flat
        ends = np.flatnonzero(q == 0)
        n_ep, n_closed = starts.shape[0], ends.shape[0]
        hi = np.full(n_ep, idx.shape[0], np.int64)
        hi[:n_closed] = ends + 1
        closed = np.arange(n_ep) < n_closed
        entry = fills.arrival_ns[idx[starts]]
        exit_ns = np.full(n_ep, MISSING_NS, np.int64)
        exit_ns[:n_closed] = fills.arrival_ns[idx[ends]]
        holding = np.full(n_ep, MISSING_NS, np.int64)
        holding[:n_closed] = exit_ns[:n_closed] - entry[:n_closed]
        parts.append(
            {
                "first": idx[starts],
                "venue": np.full(n_ep, v, np.int8),
                "direction": np.sign(q[starts]).astype(np.int8),
                "entry_ns": entry,
                "exit_ns": exit_ns,
                "holding_ns": holding,
                "n_fills": hi - starts,
                "max_abs_units": np.maximum.reduceat(np.abs(q), starts),
                "gross_pnl": np.where(closed, _sums(gross_fill[idx], starts, hi), np.nan),
                "fees": _sums(fee_fill[idx], starts, hi),
                "net_pnl": np.where(closed, _sums(fills.cash[idx], starts, hi), np.nan),
                "closed": closed,
            }
        )
    order = (
        np.argsort(np.concatenate([part["first"] for part in parts]), kind="stable") if parts else np.zeros(0, np.int64)
    )

    def column(name: str, dtype: type[np.generic]) -> NDArray[Any]:
        if not parts:
            return np.zeros(0, dtype)
        return np.ascontiguousarray(np.concatenate([part[name] for part in parts])[order], dtype=dtype)

    return Episodes(
        venue=column("venue", np.int8),
        direction=column("direction", np.int8),
        entry_ns=column("entry_ns", np.int64),
        exit_ns=column("exit_ns", np.int64),
        holding_ns=column("holding_ns", np.int64),
        n_fills=column("n_fills", np.int64),
        max_abs_units=column("max_abs_units", np.int64),
        gross_pnl=column("gross_pnl", np.float64),
        fees=column("fees", np.float64),
        net_pnl=column("net_pnl", np.float64),
        closed=column("closed", np.bool_),
    )


# ---------------------------------------------------------------------------
# Account summary
# ---------------------------------------------------------------------------


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= RECONCILE_ABS_TOL + RECONCILE_REL_TOL * max(abs(a), abs(b))


def _opt_float(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def _opt_time(ns: int) -> int | None:
    return None if ns == MISSING_NS else int(ns)


def summary(
    result: ReplayResult, prep: PreparedMarket, scenario: Scenario | None = None, *, limits: RiskLimits | None = None
) -> dict[str, Any]:
    """Whole-account metrics from the full ordered records, reconciled with the settlement.

    Returns strict-JSON-ready values (``None`` marks an unavailable value, never ``0``).
    ``net_pnl`` is the settlement score and is ``None`` for an invalid account; the
    reconciliation block re-derives positions, cap extremes, turnover, cash and (with a
    ``scenario``) fees from the fill records and compares them with the ledger, and checks
    that the final example cash equals the net score when the account is valid. ``limits``
    (the replay's cap) lets the cap breach be re-derived too.
    """
    nv = _check_market(result, prep)
    fills = _ordered_fills(result)
    st = result.settlement
    instruments = prep.data.instruments
    cap = limits.max_abs_position_units if limits is not None else None

    final_pos: list[int] = []
    max_abs: list[int] = []
    cash_v: list[float] = []
    traded: list[int] = []
    counts: list[int] = []
    for v in range(nv):
        _, q = position_path(result, v)
        _, c = cash_path(result, v)
        final_pos.append(int(q[-1]) if q.shape[0] else 0)
        max_abs.append(int(np.abs(q).max()) if q.shape[0] else 0)
        cash_v.append(float(c[-1]) if c.shape[0] else 0.0)
        mask = fills.venue == v
        traded.append(int(fills.qty_units[mask].sum()))
        counts.append(int(np.count_nonzero(mask)))
    _, c_all = cash_path(result)
    final_cash = float(c_all[-1]) if c_all.shape[0] else 0.0

    fees_match: bool | None = None
    if scenario is not None:
        fees: list[float] = []
        for v, spec in enumerate(instruments):
            mask = fills.venue == v
            px = fills.px_ticks[mask].astype(np.float64) * float(spec.tick_size)
            fee = float(scenario.fee_for(spec))
            exposure = fills.qty_units[mask] * spec.cash_factor
            value = fee * exposure / px if spec.inverse else fee * exposure * px
            fixed = float(scenario.fee_per_unit_for(spec)) * float(spec.quantity_step) * fills.qty_units[mask]
            fees.append(math.fsum((value + fixed).tolist()))
        fees_match = all(_close(a, b) for a, b in zip(fees, st.fees, strict=True))
    else:
        fees = [float(x) for x in st.fees]

    not_flat = any(final_pos)
    violations: list[str] = []
    validity_match: bool | None
    if cap is not None:
        if any(m > cap for m in max_abs):
            violations.append("cap_exceeded")
        if not_flat:
            violations.append("not_flat")
        validity_match = st.valid == (not violations) and st.reason == (violations[0] if violations else None)
    else:
        if not_flat:
            violations.append("not_flat")
        validity_match = (
            None
            if st.reason == "cap_exceeded"
            else (st.valid == (not not_flat) and st.reason == ("not_flat" if not_flat else None))
        )

    ep = episodes(result, prep)
    closed = ep.closed
    n_closed = int(np.count_nonzero(closed))
    wins = int(np.count_nonzero(ep.net_pnl[closed] > 0.0))
    holding_s = ep.holding_ns[closed].astype(np.float64) / NS_PER_S
    mark_t, mark = marked_value_path(result, prep)
    dd = drawdown(mark_t, mark)
    final_mark = float(mark[-1]) if mark.shape[0] else 0.0

    net = st.net_pnl
    reconciliation: dict[str, bool | None] = {
        "positions_match": tuple(final_pos) == st.final_position_units,
        "max_abs_match": tuple(max_abs) == st.max_abs_position_units,
        "traded_match": tuple(traded) == st.traded_units,
        "fill_counts_match": tuple(counts) == st.n_fills,
        "cash_by_venue_match": all(_close(a, b) for a, b in zip(cash_v, st.cash, strict=True)),
        "fees_match": fees_match,
        "validity_match": validity_match,
        "final_cash_equals_net_pnl": None if net is None else _close(final_cash, net),
        "terminal_mark_equals_cash": None if not_flat else final_mark == final_cash,
        "episodes_sum_to_net": (None if net is None else _close(math.fsum(ep.net_pnl[closed].tolist()), net)),
    }
    orders = result.orders
    requested = int(orders.qty_units.sum())
    return {
        "scenario": scenario.name if scenario is not None else None,
        "timing_mode": scenario.timing.name if scenario is not None else None,
        "order_latency_ns": scenario.order_latency_ns if scenario is not None else None,
        "ack_latency_ns": scenario.ack_ns if scenario is not None else None,
        "venues": [spec.key for spec in instruments],
        "valid": bool(st.valid),
        "reason": st.reason,
        "violations": violations,
        "net_pnl": None if net is None else float(net),
        "settlement_currency": st.currency,
        "final_cash": final_cash,
        "cash_by_venue": cash_v,
        "fills": len(fills),
        "fills_by_venue": counts,
        "orders": len(orders),
        "orders_filled": int(np.count_nonzero(orders.filled_units > 0)),
        "fill_ratio": int(orders.filled_units.sum()) / requested if requested else None,
        "traded_units": sum(traded),
        "traded_units_by_venue": traded,
        "fees": math.fsum(fees),
        "fees_by_venue": fees,
        "round_trips": n_closed,
        "open_episodes": len(ep) - n_closed,
        "wins": wins,
        "win_rate": wins / n_closed if n_closed else None,
        "mean_holding_s": float(np.mean(holding_s)) if n_closed else None,
        "median_holding_s": float(np.median(holding_s)) if n_closed else None,
        "max_abs_position_units": max_abs,
        "final_position_units": final_pos,
        "cap_units": cap,
        "max_drawdown": dd.max_pnl,
        "drawdown_peak_ns": _opt_time(dd.peak_ns),
        "drawdown_trough_ns": _opt_time(dd.trough_ns),
        "mark_points": int(mark.shape[0]),
        "unmarked_points": dd.unmarked_points,
        "final_mark_pnl": _opt_float(final_mark),
        "decisions": int(result.decisions),
        "reconciliation": reconciliation,
        "reconciled": all(flag is not False for flag in reconciliation.values()),
    }


__all__ = [
    "RECONCILE_ABS_TOL",
    "RECONCILE_REL_TOL",
    "Drawdown",
    "Episodes",
    "cash_path",
    "drawdown",
    "drawdown_path",
    "episodes",
    "known_position_path",
    "marked_value",
    "marked_value_path",
    "position_path",
    "summary",
]
