"""Economically executable, side-specific labels with the replay engine's execution semantics.

A label answers one counterfactual per decision time ``t``: *if the policy sent a taker IOC for
``size_units`` on ``venue`` at ``t`` (``side`` +1 buys first, -1 sells first) and closed the position at
``t + horizon``, what would it net in settlement currency after both fees?* Models and diagnostics then train on what a
strategy can actually earn rather than on mid-price moves it can never capture.

Every mechanism is the engine's (:mod:`financeai.domains.market.engine`, ``_decide``) and reuses its
building blocks (:func:`~financeai.domains.market.execution.sweep`,
:func:`~financeai.domains.market.ledger.fill_cash`) instead of re-deriving them:

* **Entry.** Decided at ``t`` on the book *observed* (received) by ``t``: IOC limit = observed best
  opposite price +/- ``entry_slip_ticks``. The order arrives at ``t + L`` and sweeps the visible depth of
  the book in effect at arrival (``EXCHANGE_CLOCK``: last book with exchange time ``<= t + L``;
  ``RECEIPT_CLOCK``: last book received by then), haircut by the scenario's ``depth_fraction`` exactly as
  the engines do; the remainder lapses (one shot, no chasing). The policy learns the fill at the
  acknowledgement ``t + L + L_ack``.
* **Exit.** Decided at ``t_x = max(t + horizon, t + L + L_ack)`` (wall clock, never before the entry's
  fill report: the position is unknown until then and the engine keeps one order in flight per venue)
  on the book observed by ``t_x``, limit +/- ``exit_slip_ticks``, arriving at ``t_x + L``. An unfilled
  remainder is retried at that order's acknowledgement - the engine's next decision opportunity - until
  flat; liquidity we consumed stays unavailable for ``consumption_ttl_ns`` across those retries. A
  decision that observes no valid book (none received yet, a clear, a one-sided book) waits for the
  venue's next book, exactly as a blocked engine decision does.
* **Session.** No order may *arrive* after ``session_end_ns`` (the engine's execution cutoff ``T``;
  default: the last receipt time of the data). A position that cannot be closed by then, or within
  ``max_exit_retries`` retries, is CENSORED: its cash fields are NaN - never zero and never marked at a
  fabricated price - and its status says so.
* **Cash.** Each fill contributes the instrument-specific ``fill_cash`` and the sums are
  exact (bit-identical to ``math.fsum``), so ``net_pnl`` equals the engine settlement of the same
  one-shot episode bit for bit; ``fees`` likewise equals the settlement's per-venue fees.

Alignment with :func:`~financeai.domains.market.engine.run_replay`: a threshold policy entering on a
single cohort at ``t`` with ``max_hold_ns = horizon`` reproduces a label fill for fill whenever the
engine has a decision opportunity at ``t + horizon``. The engine decides only at receipt cohorts and
acknowledgements, so its hold expiry acts at the first cohort at or after ``t + horizon`` while the
label's exit clock is a timer; on dense data the two differ by the gap to the next event.

Labels are independent counterfactuals: each episode starts flat with an empty consumption table, so
overlapping decisions do not interact, and no L2 replay can model the market's reaction to our orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from financeai.core.errors import InvalidConfigError
from financeai.core.jit import njit
from financeai.domains.market.engine import PreparedMarket
from financeai.domains.market.execution import CONSUMPTION_SLOTS, sweep
from financeai.domains.market.ledger import Fills, fill_cash
from financeai.domains.market.scenario import RiskLimits, Scenario
from financeai.domains.market.stream import MarketData
from financeai.domains.market.tape import DEPTH


class LabelStatus(IntEnum):
    """Outcome class of one label (``status`` columns, int8)."""

    FILLED = 0
    """The full size was entered and closed: the cash fields are exact."""
    PARTIAL_ENTRY = 1
    """The IOC entry filled part of the size; that matched quantity was closed and is what is scored."""
    NO_FILL = 2
    """The entry was sent and filled nothing: an observed outcome worth exactly zero cash (not censoring)."""
    CENSORED = 3
    """Not flat by the cutoff or within the retry budget: the cash fields are NaN, never zero."""
    NO_ORDER = 4
    """Nothing could be sent (no valid observed book at ``t``, or arrival after the cutoff; for a
    liquidation a zero position): zero cash, but not an outcome of the action - exclude it when training."""


_FILLED: Final = int(LabelStatus.FILLED)
_PARTIAL: Final = int(LabelStatus.PARTIAL_ENTRY)
_NO_FILL: Final = int(LabelStatus.NO_FILL)
_CENSORED: Final = int(LabelStatus.CENSORED)
_NO_ORDER: Final = int(LabelStatus.NO_ORDER)
_ROUND_TRIP: Final = 0
_LIQUIDATE: Final = 1
_P_NET, _P_GROSS, _P_FEES, _P_ENTRY_INV, _P_EXIT_INV = range(5)
"""Rows of the per-episode exact-sum buffers (inverse sums are sum q/p for the average prices)."""

N_PARTIALS: Final = 2112
"""Capacity of one exact-sum buffer. Shewchuk partials are non-overlapping doubles, i.e. every partial owns
at least one of the 2098 bit positions a double can occupy (2^-1074 .. 2^1023), so at most 2098 are ever live
(partials can be a single bit wide: sign-mixed values across the exponent range keep about 90, not the
2098 / 53 < 40 a naive 53-bit argument gives). CPython's ``math.fsum`` grows its array instead; here the
bound is static because the kernel runs in nopython mode, where an overflow would be an unchecked write.
Five bounded partial-sum buffers avoid dynamic allocations inside the kernel."""
DEFAULT_MAX_EXIT_RETRIES: Final = 100_000
"""Maximum exit retries before a label is censored. Choose a budget that does not
bind your experiment; reaching it is a censored observation, never a zero payoff."""
_MAX_TIME_NS: Final = 2**62
_MAX_HORIZON_NS: Final = 2**61
_MAX_LATENCY_NS: Final = 2**60
"""Bound on ``order_latency_ns + ack_latency_ns``: with decisions <= 2**62 and horizons <= 2**61 the largest
clock the kernel forms, ``t + horizon + L + L_ack``, then stays below 2**63 (int64 would wrap silently)."""
_MAX_TTL_NS: Final = 2**61
"""Bound on ``consumption_ttl_ns``: an arrival (<= the cutoff <= 2**62) plus the TTL must fit int64."""
_MAX_SLIP_TICKS: Final = 1_000_000


# ---------------------------------------------------------------------------
# Kernels (Numba subset; identical pure-Python fallback)
# ---------------------------------------------------------------------------


@njit()
def _fsum_add(partials: NDArray[np.float64], n: int, x: float) -> int:
    """Add ``x`` to the exact partial sums (CPython ``math.fsum``, Shewchuk); returns the new count."""
    i = 0
    for j in range(n):
        y = partials[j]
        if abs(x) < abs(y):
            x, y = y, x
        hi = x + y
        lo = y - (hi - x)
        if lo != 0.0:
            partials[i] = lo
            i += 1
        x = hi
    if x != 0.0:
        partials[i] = x
        i += 1
    return i


@njit()
def _fsum_value(partials: NDArray[np.float64], n: int) -> float:
    """Correctly rounded value of the partial sums: CPython's final step, including its half-even
    fix-up across partials, so the result is bit-identical to ``math.fsum`` of the same values."""
    if n == 0:
        return 0.0
    n -= 1
    hi: float = partials[n]
    lo = 0.0
    while n > 0:
        x = hi
        n -= 1
        y = partials[n]
        hi = x + y
        lo = y - (hi - x)
        if lo != 0.0:
            break
    if n > 0 and ((lo < 0.0 and partials[n - 1] < 0.0) or (lo > 0.0 and partials[n - 1] > 0.0)):
        y = lo * 2.0
        x = hi + y
        if y == x - hi:
            hi = x
    return hi


@njit()
def _ioc(
    tau: int,
    side: int,
    qty: int,
    slip: int,
    obs: int,
    bpx: NDArray[np.int32],
    bsz: NDArray[np.int32],
    apx: NDArray[np.int32],
    asz: NDArray[np.int32],
    nb: NDArray[np.int8],
    na: NDArray[np.int8],
    valid: NDArray[np.bool_],
    exch_env: NDArray[np.int64],
    book_recv: NDArray[np.int64],
    lot: int,
    lat: int,
    timing: int,
    ttl: int,
    frac: float,
    c_px: NDArray[np.int32],
    c_amt: NDArray[np.int64],
    c_exp: NDArray[np.int64],
    out_px: NDArray[np.int32],
    out_qty: NDArray[np.int64],
) -> int:
    """The engine's order path (``_decide``) for one IOC decided at ``tau`` on observed row ``obs``.

    Limit from the observed best opposite price, arrival at ``tau + lat`` against the book in effect then
    (``timing`` 0 = exchange clock, 1 = receipt clock), visible-depth sweep haircut by the scenario's
    ``frac`` (``depth_fraction``) and net of our own consumption (``c_*[0]`` bids, ``c_*[1]`` asks). An
    arrival without a valid book fills nothing; the order still counts as sent. Returns the number of
    per-level fills written to ``out_px``/``out_qty``."""
    limit = apx[obs, 0] + slip if side > 0 else bpx[obs, 0] - slip
    arrival = tau + lat
    if timing == 0:
        row = np.searchsorted(exch_env, arrival, side="right") - 1
    else:
        row = np.searchsorted(book_recv, arrival, side="right") - 1
    if row < 0 or not valid[row]:
        return 0
    if side > 0:
        return sweep(
            side,
            qty,
            limit,
            apx[row],
            asz[row],
            na[row],
            lot,
            frac,
            c_px[1],
            c_amt[1],
            c_exp[1],
            arrival,
            ttl,
            out_px,
            out_qty,
        )
    return sweep(
        side,
        qty,
        limit,
        bpx[row],
        bsz[row],
        nb[row],
        lot,
        frac,
        c_px[0],
        c_amt[0],
        c_exp[0],
        arrival,
        ttl,
        out_px,
        out_qty,
    )


@njit()
def _add_fills(
    n_out: int,
    side: int,
    out_px: NDArray[np.int32],
    out_qty: NDArray[np.int64],
    tick: float,
    fee: float,
    inverse: bool,
    factor: float,
    fixed: float,
    inv_slot: int,
    parts: NDArray[np.float64],
    n_parts: NDArray[np.int64],
) -> int:
    """Fold per-level fills into the exact sums with the ledger's per-fill formulas (net = ``fill_cash``,
    gross = ``fill_cash`` at zero fee, fee = ``fee * q / p`` as in ``settle``) plus the inverse-price sum
    of row ``inv_slot``; returns the filled quantity."""
    filled = 0
    for j in range(n_out):
        qty = out_qty[j]
        price = out_px[j] * tick
        n_parts[_P_NET] = _fsum_add(
            parts[_P_NET], n_parts[_P_NET], fill_cash(side, qty, price, fee, inverse, factor, fixed)
        )
        n_parts[_P_GROSS] = _fsum_add(
            parts[_P_GROSS], n_parts[_P_GROSS], fill_cash(side, qty, price, 0.0, inverse, factor)
        )
        n_parts[_P_FEES] = _fsum_add(
            parts[_P_FEES],
            n_parts[_P_FEES],
            (fee * (qty * factor) / price if inverse else fee * (qty * factor) * price) + qty * fixed,
        )
        n_parts[inv_slot] = _fsum_add(parts[inv_slot], n_parts[inv_slot], qty / price if inverse else qty * price)
        filled += qty
    return filled


@njit()
def _record_fills(
    n_out: int,
    nf: int,
    oid: int,
    side: int,
    tau: int,
    lat: int,
    ack_lat: int,
    out_px: NDArray[np.int32],
    out_qty: NDArray[np.int64],
    tick: float,
    fee: float,
    inverse: bool,
    factor: float,
    fixed: float,
    f_order: NDArray[np.int64],
    f_side: NDArray[np.int8],
    f_px: NDArray[np.int32],
    f_qty: NDArray[np.int64],
    f_submit: NDArray[np.int64],
    f_arrival: NDArray[np.int64],
    f_ack: NDArray[np.int64],
    f_cash: NDArray[np.float64],
) -> int:
    """Append one order's fills as ledger rows (the engine's fill record); returns the new count."""
    for j in range(n_out):
        f_order[nf] = oid
        f_side[nf] = side
        f_px[nf] = out_px[j]
        f_qty[nf] = out_qty[j]
        f_submit[nf] = tau
        f_arrival[nf] = tau + lat
        f_ack[nf] = tau + lat + ack_lat
        f_cash[nf] = fill_cash(side, out_qty[j], out_px[j] * tick, fee, inverse, factor, fixed)
        nf += 1
    return nf


@njit()
def _label_kernel(
    mode: int,
    dec_t: NDArray[np.int64],
    sides: NDArray[np.int64],
    qtys: NDArray[np.int64],
    horizon: int,
    eslip: int,
    xslip: int,
    bpx: NDArray[np.int32],
    bsz: NDArray[np.int32],
    apx: NDArray[np.int32],
    asz: NDArray[np.int32],
    nb: NDArray[np.int8],
    na: NDArray[np.int8],
    valid: NDArray[np.bool_],
    exch_env: NDArray[np.int64],
    book_recv: NDArray[np.int64],
    lot: int,
    tick: float,
    fee: float,
    inverse: bool,
    factor: float,
    fixed: float,
    lat: int,
    ack_lat: int,
    timing: int,
    ttl: int,
    frac: float,
    cutoff: int,
    max_retries: int,
    o_entry_arrival: NDArray[np.int64],
    o_entry_qty: NDArray[np.int64],
    o_entry_avg: NDArray[np.float64],
    o_exit_first: NDArray[np.int64],
    o_exit_qty: NDArray[np.int64],
    o_exit_avg: NDArray[np.float64],
    o_exit_orders: NDArray[np.int64],
    o_net: NDArray[np.float64],
    o_gross: NDArray[np.float64],
    o_fees: NDArray[np.float64],
    o_status: NDArray[np.int8],
    o_known: NDArray[np.int64],
    record: bool,
    n_rec: NDArray[np.int64],
    f_order: NDArray[np.int64],
    f_side: NDArray[np.int8],
    f_px: NDArray[np.int32],
    f_qty: NDArray[np.int64],
    f_submit: NDArray[np.int64],
    f_arrival: NDArray[np.int64],
    f_ack: NDArray[np.int64],
    f_cash: NDArray[np.float64],
) -> None:
    """One independent episode per decision ``dec_t[i]`` on one venue.

    ``_ROUND_TRIP``: open ``qtys[i]`` units on side ``sides[i]`` at ``t`` and close the filled quantity
    from ``max(t + horizon, entry acknowledgement)``. ``_LIQUIDATE``: close an existing position of
    ``qtys[i]`` quantity units held on side ``sides[i]`` from ``t + horizon``. ``o_known`` is the time of the last
    decision the episode needed: the acknowledgement that completed it, the entry acknowledgement of a
    no-fill, or the instant at which censoring became certain (the decision refused by the cutoff or the
    retry budget; for a wait on a book that never comes, the last instant from which an order could still
    have arrived by the cutoff); it is never before an acknowledgement of the episode. Fills go to ``f_*``
    (count in ``n_rec[0]``) only when ``record`` is set."""
    n_rows = book_recv.shape[0]
    c_px = np.zeros((2, CONSUMPTION_SLOTS), np.int32)
    c_amt = np.zeros((2, CONSUMPTION_SLOTS), np.int64)
    c_exp = np.zeros((2, CONSUMPTION_SLOTS), np.int64)
    out_px = np.zeros(DEPTH, np.int32)
    out_qty = np.zeros(DEPTH, np.int64)
    parts = np.zeros((5, N_PARTIALS), np.float64)
    n_parts = np.zeros(5, np.int64)
    nf = 0
    for i in range(dec_t.shape[0]):
        t = dec_t[i]
        side = sides[i]
        qty = qtys[i]
        c_px[:] = 0  # every label is its own flat episode: no consumption carried between labels
        c_amt[:] = 0
        c_exp[:] = 0
        n_parts[:] = 0
        status = _FILLED
        oid = 0
        entry_arrival = -1
        entry_qty = 0
        exit_first = -1
        exit_qty = 0
        exit_orders = 0
        tau = t + horizon
        known = t
        remaining = qty
        if mode == _ROUND_TRIP:
            obs = np.searchsorted(book_recv, t, side="right") - 1
            if obs < 0 or not valid[obs] or t + lat > cutoff:
                status = _NO_ORDER
                remaining = 0
            else:
                n_out = _ioc(
                    t,
                    side,
                    qty,
                    eslip,
                    obs,
                    bpx,
                    bsz,
                    apx,
                    asz,
                    nb,
                    na,
                    valid,
                    exch_env,
                    book_recv,
                    lot,
                    lat,
                    timing,
                    ttl,
                    frac,
                    c_px,
                    c_amt,
                    c_exp,
                    out_px,
                    out_qty,
                )
                entry_qty = _add_fills(
                    n_out, side, out_px, out_qty, tick, fee, inverse, factor, fixed, _P_ENTRY_INV, parts, n_parts
                )
                if record:
                    nf = _record_fills(
                        n_out,
                        nf,
                        0,
                        side,
                        t,
                        lat,
                        ack_lat,
                        out_px,
                        out_qty,
                        tick,
                        fee,
                        inverse,
                        factor,
                        fixed,
                        f_order,
                        f_side,
                        f_px,
                        f_qty,
                        f_submit,
                        f_arrival,
                        f_ack,
                        f_cash,
                    )
                oid = 1
                entry_arrival = t + lat
                known = entry_arrival + ack_lat
                if known > tau:
                    tau = known  # the exit needs the entry's fill report (one order in flight per venue)
                remaining = entry_qty
                if entry_qty == 0:
                    status = _NO_FILL
                elif entry_qty < qty:
                    status = _PARTIAL
        elif qty == 0:
            status = _NO_ORDER
            known = tau
        close = -side
        while remaining > 0:
            obs = np.searchsorted(book_recv, tau, side="right") - 1
            while (obs < 0 or not valid[obs]) and obs + 1 < n_rows:
                tau = book_recv[obs + 1]  # blocked: only this venue's next book can unblock the decision
                obs = np.searchsorted(book_recv, tau, side="right") - 1
            if obs < 0 or not valid[obs] or tau + lat > cutoff or exit_orders > max_retries:
                status = _CENSORED
                known = tau
                if obs < 0 or not valid[obs]:
                    # the venue's rows ran out while waiting for a valid book: the policy only knows at
                    # cutoff - lat that no order can arrive in time (the retry budget is known at once)
                    known = max(tau, cutoff - lat)
                break
            n_out = _ioc(
                tau,
                close,
                remaining,
                xslip,
                obs,
                bpx,
                bsz,
                apx,
                asz,
                nb,
                na,
                valid,
                exch_env,
                book_recv,
                lot,
                lat,
                timing,
                ttl,
                frac,
                c_px,
                c_amt,
                c_exp,
                out_px,
                out_qty,
            )
            filled = _add_fills(
                n_out, close, out_px, out_qty, tick, fee, inverse, factor, fixed, _P_EXIT_INV, parts, n_parts
            )
            if record:
                nf = _record_fills(
                    n_out,
                    nf,
                    oid,
                    close,
                    tau,
                    lat,
                    ack_lat,
                    out_px,
                    out_qty,
                    tick,
                    fee,
                    inverse,
                    factor,
                    fixed,
                    f_order,
                    f_side,
                    f_px,
                    f_qty,
                    f_submit,
                    f_arrival,
                    f_ack,
                    f_cash,
                )
            if exit_first < 0:
                exit_first = tau
            oid += 1
            exit_orders += 1
            exit_qty += filled
            remaining -= filled
            tau += lat + ack_lat  # the next decision opportunity is this order's acknowledgement
            known = tau
            if filled == 0 and lat + ack_lat == 0 and remaining > 0:
                status = _CENSORED  # zero total latency: a retry at the same instant meets the same book
                break
        o_entry_arrival[i] = entry_arrival
        o_entry_qty[i] = entry_qty
        entry_sum = _fsum_value(parts[_P_ENTRY_INV], n_parts[_P_ENTRY_INV])
        o_entry_avg[i] = (entry_qty / entry_sum if inverse else entry_sum / entry_qty) if entry_qty > 0 else np.nan
        o_exit_first[i] = exit_first
        o_exit_qty[i] = exit_qty
        exit_sum = _fsum_value(parts[_P_EXIT_INV], n_parts[_P_EXIT_INV])
        o_exit_avg[i] = (exit_qty / exit_sum if inverse else exit_sum / exit_qty) if exit_qty > 0 else np.nan
        o_exit_orders[i] = exit_orders
        if status == _CENSORED:
            o_net[i] = np.nan
            o_gross[i] = np.nan
            o_fees[i] = np.nan
        else:
            o_net[i] = _fsum_value(parts[_P_NET], n_parts[_P_NET])
            o_gross[i] = _fsum_value(parts[_P_GROSS], n_parts[_P_GROSS])
            o_fees[i] = _fsum_value(parts[_P_FEES], n_parts[_P_FEES])
        o_status[i] = status
        o_known[i] = known
    n_rec[0] = nf


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


def _freeze(obj: object, columns: tuple[tuple[str, type[np.generic]], ...]) -> None:
    """Check column dtypes/shapes against the first column and make every column read-only."""
    n = getattr(obj, columns[0][0]).shape[0]
    for name, dtype in columns:
        arr = getattr(obj, name)
        if not isinstance(arr, np.ndarray) or arr.dtype != np.dtype(dtype) or arr.ndim != 1 or arr.shape[0] != n:
            raise InvalidConfigError("label column has wrong dtype/shape", context={"column": name})
        arr.setflags(write=False)


_LABEL_COLUMNS: Final = (
    ("decision_ns", np.int64),
    ("entry_arrival_ns", np.int64),
    ("entry_qty_units", np.int64),
    ("entry_avg_px", np.float64),
    ("exit_decision_ns", np.int64),
    ("exit_qty_units", np.int64),
    ("exit_avg_px", np.float64),
    ("exit_orders", np.int64),
    ("net_pnl", np.float64),
    ("gross_pnl", np.float64),
    ("fees", np.float64),
    ("status", np.int8),
    ("knowable_ns", np.int64),
)
_LIQUIDATION_COLUMNS: Final = (
    ("decision_ns", np.int64),
    ("position_units", np.int64),
    ("first_order_ns", np.int64),
    ("qty_units", np.int64),
    ("avg_px", np.float64),
    ("orders", np.int64),
    ("cash", np.float64),
    ("gross_pnl", np.float64),
    ("fees", np.float64),
    ("status", np.int8),
    ("knowable_ns", np.int64),
)
_HOLD_COLUMNS: Final = (
    ("decision_ns", np.int64),
    ("position_units", np.int64),
    ("value", np.float64),
    ("status", np.int8),
    ("knowable_ns", np.int64),
)


def _status_counts(status: NDArray[np.int8]) -> dict[str, int]:
    counts = np.bincount(status.astype(np.int64), minlength=len(LabelStatus))
    return {s.name: int(counts[s.value]) for s in LabelStatus}


@dataclass(frozen=True, slots=True)
class LabelSet:
    """Round-trip labels of one (side, venue, size, horizon), one row per decision (input order).

    Times are int64 epoch ns (-1 = no such event). Quantities use stored units; prices use quote currency.
    Average prices are harmonic for inverse contracts and arithmetic otherwise (NaN without fills).
    Cash fields contain the episode's settlement result.
    ``net_pnl`` is exactly ``fsum(fill_cash)``; ``gross_pnl`` and ``fees`` are separately exact sums, so
    ``net = gross - fees`` holds up to one rounding. Train only on rows whose ``knowable_ns`` precedes the
    training cutoff (purging), and mask CENSORED (NaN) and NO_ORDER rows explicitly."""

    side: int
    venue: int
    size_units: int
    horizon_ns: int
    decision_ns: NDArray[np.int64]
    entry_arrival_ns: NDArray[np.int64]
    """``t + L`` when the entry was sent, else -1."""
    entry_qty_units: NDArray[np.int64]
    entry_avg_px: NDArray[np.float64]
    exit_decision_ns: NDArray[np.int64]
    """Decision time of the first exit order (``max(t + horizon, entry ack)`` unless blocked), else -1."""
    exit_qty_units: NDArray[np.int64]
    exit_avg_px: NDArray[np.float64]
    exit_orders: NDArray[np.int64]
    """Exit orders sent: the first plus its retries; 0 when no exit was sent (NO_FILL, NO_ORDER, or
    censored before the first exit order)."""
    net_pnl: NDArray[np.float64]
    gross_pnl: NDArray[np.float64]
    fees: NDArray[np.float64]
    status: NDArray[np.int8]
    """:class:`LabelStatus` values."""
    knowable_ns: NDArray[np.int64]
    """When the outcome is known to the policy: the acknowledgement of the last exit order (of the entry
    for NO_FILL; ``t`` for NO_ORDER; for CENSORED the instant at which censoring became certain: the
    refused decision, or ``session_end - L`` when the venue's data ended while waiting for a book)."""

    def __post_init__(self) -> None:
        _freeze(self, _LABEL_COLUMNS)

    def __len__(self) -> int:
        return int(self.decision_ns.shape[0])

    def status_counts(self) -> dict[str, int]:
        return _status_counts(self.status)

    @property
    def trainable(self) -> NDArray[np.bool_]:
        """Rows that are outcomes of the action (FILLED, PARTIAL_ENTRY, NO_FILL): the mask to apply before
        fitting, besides purging on ``knowable_ns``. CENSORED (NaN) and NO_ORDER (0.0 by convention) are not."""
        return self.status <= _NO_FILL


@dataclass(frozen=True, slots=True)
class LiquidationSet:
    """Closing an existing position (one row per decision): decided from ``decision_ns``, retried at each
    acknowledgement until flat. ``cash`` is the exact ``fsum(fill_cash)`` of the closing fills (NaN when
    CENSORED); status is FILLED, CENSORED or NO_ORDER (zero position)."""

    decision_ns: NDArray[np.int64]
    position_units: NDArray[np.int64]
    first_order_ns: NDArray[np.int64]
    qty_units: NDArray[np.int64]
    avg_px: NDArray[np.float64]
    orders: NDArray[np.int64]
    cash: NDArray[np.float64]
    gross_pnl: NDArray[np.float64]
    fees: NDArray[np.float64]
    status: NDArray[np.int8]
    knowable_ns: NDArray[np.int64]

    def __post_init__(self) -> None:
        _freeze(self, _LIQUIDATION_COLUMNS)

    def __len__(self) -> int:
        return int(self.decision_ns.shape[0])


@dataclass(frozen=True, slots=True)
class HoldValueSet:
    """Incremental cash value of holding an existing position from ``t`` to ``t + horizon`` instead of
    liquidating at ``t``: ``later.cash - now.cash`` (no entry costs; both legs pay the exit fee).
    NaN when either liquidation is censored; 0.0 for a zero position (status NO_ORDER)."""

    venue: int
    horizon_ns: int
    decision_ns: NDArray[np.int64]
    position_units: NDArray[np.int64]
    now: LiquidationSet
    later: LiquidationSet
    value: NDArray[np.float64]
    status: NDArray[np.int8]
    knowable_ns: NDArray[np.int64]
    """``max`` of both legs' knowable times (the later leg's, unless it is degenerate)."""

    def __post_init__(self) -> None:
        _freeze(self, _HOLD_COLUMNS)

    def __len__(self) -> int:
        return int(self.decision_ns.shape[0])

    def status_counts(self) -> dict[str, int]:
        return _status_counts(self.status)

    @property
    def trainable(self) -> NDArray[np.bool_]:
        """Rows with a real value (both liquidations finished): not CENSORED (NaN) and not a zero position."""
        return np.asarray(self.status == _FILLED, dtype=np.bool_)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def _int(name: str, value: object, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or not minimum <= int(value) <= maximum:
        raise InvalidConfigError(f"{name} must be an integer in [{minimum}, {maximum}]", context={"got": value})
    return int(value)


def _times(values: ArrayLike, name: str) -> NDArray[np.int64]:
    arr = np.asarray(values)
    if arr.ndim != 1 or (arr.size and (not np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.uint64)):
        raise InvalidConfigError(
            f"{name} must be a 1-D integer array of epoch nanoseconds",
            context={"dtype": str(arr.dtype), "shape": arr.shape},
        )
    out = np.array(arr, dtype=np.int64)  # always a copy: results are frozen and must not alias the input
    if out.size and (int(out.min()) < 0 or int(out.max()) > _MAX_TIME_NS):
        raise InvalidConfigError(f"{name} must lie in [0, 2**62] ns")
    return out


@dataclass(frozen=True, slots=True)
class _Venue:
    """Validated per-venue execution context shared by every label kind."""

    prep: PreparedMarket
    venue: int
    lot: int
    tick: float
    fee: float
    inverse: bool
    factor: float
    fixed: float
    lat: int
    ack_lat: int
    timing: int
    ttl: int
    frac: float
    cutoff: int
    cap: int
    max_retries: int


def _venue(
    data: PreparedMarket | MarketData,
    venue: int,
    scenario: Scenario,
    session_end_ns: int | None,
    max_exit_retries: int,
    limits: RiskLimits | None,
) -> _Venue:
    """Validate the per-venue execution context; the (expensive) market preparation comes last."""
    if scenario.order_latency_ns + scenario.ack_ns > _MAX_LATENCY_NS:
        raise InvalidConfigError(
            "order_latency_ns + ack_latency_ns must not exceed 2**60 ns",
            context={"order_latency_ns": scenario.order_latency_ns, "ack_ns": scenario.ack_ns},
        )
    if scenario.consumption_ttl_ns > _MAX_TTL_NS:
        raise InvalidConfigError(
            "consumption_ttl_ns must not exceed 2**61 ns", context={"got": scenario.consumption_ttl_ns}
        )
    prep = data if isinstance(data, PreparedMarket) else PreparedMarket.build(data)
    v = _int("venue", venue, 0, len(prep.data) - 1)
    recv = prep.stream.recv_ns
    cutoff = (
        (int(recv[-1]) if recv.shape[0] else 0)
        if session_end_ns is None
        else _int("session_end_ns", session_end_ns, 0, _MAX_TIME_NS)
    )
    return _Venue(
        prep=prep,
        venue=v,
        lot=int(prep.lot[v]),
        tick=float(prep.tick[v]),
        fee=float(scenario.fee_for(prep.data.instruments[v])),
        inverse=bool(prep.inverse[v]),
        factor=float(prep.cash_factor[v]),
        fixed=float(scenario.fee_per_unit_for(prep.data.instruments[v]))
        * float(prep.data.instruments[v].quantity_step),
        lat=scenario.order_latency_ns,
        ack_lat=scenario.ack_ns,
        timing=int(scenario.timing),
        ttl=scenario.consumption_ttl_ns,
        frac=float(scenario.depth_fraction),
        cutoff=cutoff,
        cap=(limits or RiskLimits()).max_abs_position_units,
        max_retries=_int("max_exit_retries", max_exit_retries, 0, 2**31),
    )


def _side(side: object) -> None:
    if isinstance(side, bool) or side not in (1, -1):
        raise InvalidConfigError("side must be +1 (buy first) or -1 (sell first)", context={"got": side})


def _size(ctx: _Venue, size_units: object) -> int:
    size = _int("size_units", size_units, 1, ctx.cap)
    if size % ctx.lot:
        raise InvalidConfigError("size_units must be a multiple of the venue lot", context={"lot": ctx.lot})
    return size


@dataclass(frozen=True, slots=True)
class _Raw:
    entry_arrival: NDArray[np.int64]
    entry_qty: NDArray[np.int64]
    entry_avg: NDArray[np.float64]
    exit_first: NDArray[np.int64]
    exit_qty: NDArray[np.int64]
    exit_avg: NDArray[np.float64]
    exit_orders: NDArray[np.int64]
    net: NDArray[np.float64]
    gross: NDArray[np.float64]
    fees: NDArray[np.float64]
    status: NDArray[np.int8]
    known: NDArray[np.int64]
    fills: tuple[NDArray[Any], ...]


def _run(
    ctx: _Venue,
    mode: int,
    t: NDArray[np.int64],
    sides: NDArray[np.int64],
    qtys: NDArray[np.int64],
    horizon: int,
    eslip: int,
    xslip: int,
    capacity: int = 0,
) -> _Raw:
    n = t.shape[0]
    p, v = ctx.prep, ctx.venue
    if not p.data.instruments[v].shortable and np.any(sides < 0):
        raise InvalidConfigError("short labels require an instrument with borrowing/shorting enabled")
    i64 = [np.zeros(n, np.int64) for _ in range(5)]
    f64 = [np.zeros(n, np.float64) for _ in range(5)]
    status, known = np.zeros(n, np.int8), np.zeros(n, np.int64)
    n_rec = np.zeros(1, np.int64)
    f_order, f_qty, f_submit, f_arrival, f_ack = (np.zeros(capacity, np.int64) for _ in range(5))
    f_side, f_px, f_cash = np.zeros(capacity, np.int8), np.zeros(capacity, np.int32), np.zeros(capacity, np.float64)
    _label_kernel(
        mode,
        t,
        sides,
        qtys,
        horizon,
        eslip,
        xslip,
        p.bpx[v],
        p.bsz[v],
        p.apx[v],
        p.asz[v],
        p.nb[v],
        p.na[v],
        p.valid[v],
        p.exch_env[v],
        p.book_recv[v],
        ctx.lot,
        ctx.tick,
        ctx.fee,
        ctx.inverse,
        ctx.factor,
        ctx.fixed,
        ctx.lat,
        ctx.ack_lat,
        ctx.timing,
        ctx.ttl,
        ctx.frac,
        ctx.cutoff,
        ctx.max_retries,
        i64[0],
        i64[1],
        f64[0],
        i64[2],
        i64[3],
        f64[1],
        i64[4],
        f64[2],
        f64[3],
        f64[4],
        status,
        known,
        capacity > 0,
        n_rec,
        f_order,
        f_side,
        f_px,
        f_qty,
        f_submit,
        f_arrival,
        f_ack,
        f_cash,
    )
    used = int(n_rec[0])
    fills = (f_order, f_side, f_px, f_qty, f_submit, f_arrival, f_ack, f_cash)
    return _Raw(
        i64[0],
        i64[1],
        f64[0],
        i64[2],
        i64[3],
        f64[1],
        i64[4],
        f64[2],
        f64[3],
        f64[4],
        status,
        known,
        tuple(np.ascontiguousarray(a[:used]) for a in fills),
    )


def executable_labels(
    prep: PreparedMarket | MarketData,
    decision_t: ArrayLike,
    side: int,
    venue: int,
    size_units: int,
    horizon_ns: int,
    scenario: Scenario,
    *,
    entry_slip_ticks: int = 0,
    exit_slip_ticks: int = 20,
    session_end_ns: int | None = None,
    max_exit_retries: int = DEFAULT_MAX_EXIT_RETRIES,
    limits: RiskLimits | None = None,
) -> LabelSet:
    """Round-trip labels for decisions at receipt times ``decision_t`` (any order; one row each).

    ``side`` +1 buys first, -1 sells first: long and short labels are separate calls and never negatives
    of each other (both pay the spread and two fees). ``size_units`` must be a positive multiple of the venue
    lot within the position cap (``limits``, default 10,000 units).
    ``session_end_ns`` is the execution cutoff; no order may arrive after it. It defaults to the last receipt.
    Pass ``SessionPlan.exec_cutoff_ns``
    to mirror a replay session. See the module docstring for the exact semantics."""
    _side(side)
    horizon = _int("horizon_ns", horizon_ns, 0, _MAX_HORIZON_NS)
    eslip = _int("entry_slip_ticks", entry_slip_ticks, 0, _MAX_SLIP_TICKS)
    xslip = _int("exit_slip_ticks", exit_slip_ticks, 0, _MAX_SLIP_TICKS)
    t = _times(decision_t, "decision_t")
    ctx = _venue(prep, venue, scenario, session_end_ns, max_exit_retries, limits)
    size = _size(ctx, size_units)
    n = t.shape[0]
    raw = _run(ctx, _ROUND_TRIP, t, np.full(n, side, np.int64), np.full(n, size, np.int64), horizon, eslip, xslip)
    return LabelSet(
        side=int(side),
        venue=ctx.venue,
        size_units=size,
        horizon_ns=horizon,
        decision_ns=t,
        entry_arrival_ns=raw.entry_arrival,
        entry_qty_units=raw.entry_qty,
        entry_avg_px=raw.entry_avg,
        exit_decision_ns=raw.exit_first,
        exit_qty_units=raw.exit_qty,
        exit_avg_px=raw.exit_avg,
        exit_orders=raw.exit_orders,
        net_pnl=raw.net,
        gross_pnl=raw.gross,
        fees=raw.fees,
        status=raw.status,
        knowable_ns=raw.known,
    )


def label_fills(
    prep: PreparedMarket | MarketData,
    decision_ns: int,
    side: int,
    venue: int,
    size_units: int,
    horizon_ns: int,
    scenario: Scenario,
    *,
    entry_slip_ticks: int = 0,
    exit_slip_ticks: int = 20,
    session_end_ns: int | None = None,
    max_exit_retries: int = DEFAULT_MAX_EXIT_RETRIES,
    limits: RiskLimits | None = None,
) -> Fills:
    """The ledger fills behind one label (same arguments as :func:`executable_labels`, one decision).

    Order id 0 is the entry and 1, 2, ... the exit orders, as in a one-shot replay of the same episode, so
    the rows compare directly with ``run_replay(...).fills`` and settle with the ledger."""
    _side(side)
    horizon = _int("horizon_ns", horizon_ns, 0, _MAX_HORIZON_NS)
    eslip = _int("entry_slip_ticks", entry_slip_ticks, 0, _MAX_SLIP_TICKS)
    xslip = _int("exit_slip_ticks", exit_slip_ticks, 0, _MAX_SLIP_TICKS)
    t = _times(np.array([_int("decision_ns", decision_ns, 0, _MAX_TIME_NS)]), "decision_ns")
    ctx = _venue(prep, venue, scenario, session_end_ns, max_exit_retries, limits)
    size = _size(ctx, size_units)
    sides, sizes = np.array([side], np.int64), np.array([size], np.int64)
    # Two deterministic passes: count the episode's exit orders, then record into exactly sized buffers
    # (the entry plus each exit order fill at most DEPTH levels).
    exits = int(_run(ctx, _ROUND_TRIP, t, sides, sizes, horizon, eslip, xslip).exit_orders[0])
    raw = _run(ctx, _ROUND_TRIP, t, sides, sizes, horizon, eslip, xslip, DEPTH * (1 + exits))
    order, side_col, px, qty, submit, arrival, ack, cash = raw.fills
    return Fills(
        order_id=order,
        venue=np.full(order.shape[0], ctx.venue, np.int8),
        side=side_col,
        px_ticks=px,
        qty_units=qty,
        submit_ns=submit,
        arrival_ns=arrival,
        ack_ns=ack,
        cash=cash,
    )


def _liquidation(
    ctx: _Venue, t: NDArray[np.int64], position: NDArray[np.int64], horizon: int, xslip: int
) -> LiquidationSet:
    raw = _run(ctx, _LIQUIDATE, t, np.sign(position).astype(np.int64), np.abs(position), horizon, 0, xslip)
    return LiquidationSet(
        decision_ns=t + horizon,
        position_units=position,
        first_order_ns=raw.exit_first,
        qty_units=raw.exit_qty,
        avg_px=raw.exit_avg,
        orders=raw.exit_orders,
        cash=raw.net,
        gross_pnl=raw.gross,
        fees=raw.fees,
        status=raw.status,
        knowable_ns=raw.known,
    )


def hold_value_labels(
    prep: PreparedMarket | MarketData,
    t: ArrayLike,
    position_units: int | ArrayLike,
    venue: int,
    horizon_ns: int,
    scenario: Scenario,
    *,
    exit_slip_ticks: int = 20,
    session_end_ns: int | None = None,
    max_exit_retries: int = DEFAULT_MAX_EXIT_RETRIES,
    limits: RiskLimits | None = None,
) -> HoldValueSet:
    """Value of holding an existing signed ``position_units`` (scalar or one per decision) from ``t`` to
    ``t + horizon_ns`` versus liquidating at ``t``: the difference of the two liquidation outcomes, each
    decided on the observed book and retried at acknowledgements until flat (exit semantics of
    :func:`executable_labels`; no entry costs). Positive means holding was worth more."""
    horizon = _int("horizon_ns", horizon_ns, 0, _MAX_HORIZON_NS)
    xslip = _int("exit_slip_ticks", exit_slip_ticks, 0, _MAX_SLIP_TICKS)
    times = _times(t, "t")
    raw_pos = np.asarray(position_units)
    if raw_pos.size and (not np.issubdtype(raw_pos.dtype, np.integer) or raw_pos.dtype == np.uint64):
        raise InvalidConfigError("position_units must be integer quantity units", context={"dtype": str(raw_pos.dtype)})
    try:
        position = np.array(np.broadcast_to(raw_pos, times.shape), dtype=np.int64)
    except ValueError:
        raise InvalidConfigError(
            "position_units must be a scalar or one value per decision", context={"shape": raw_pos.shape}
        ) from None
    ctx = _venue(prep, venue, scenario, session_end_ns, max_exit_retries, limits)
    if position.size and (int(np.abs(position).max()) > ctx.cap or bool(np.any(position % ctx.lot))):
        raise InvalidConfigError(
            "positions must be lot multiples within the position cap", context={"lot": ctx.lot, "cap": ctx.cap}
        )
    now = _liquidation(ctx, times, position, 0, xslip)
    later = _liquidation(ctx, times, position, horizon, xslip)
    censored = (now.status == _CENSORED) | (later.status == _CENSORED)
    status = np.where(position == 0, _NO_ORDER, np.where(censored, _CENSORED, _FILLED)).astype(np.int8)
    return HoldValueSet(
        venue=ctx.venue,
        horizon_ns=horizon,
        decision_ns=times,
        position_units=position,
        now=now,
        later=later,
        value=later.cash - now.cash,
        status=status,
        knowable_ns=np.maximum(now.knowable_ns, later.knowable_ns),
    )


__all__ = [
    "DEFAULT_MAX_EXIT_RETRIES",
    "N_PARTIALS",
    "HoldValueSet",
    "LabelSet",
    "LabelStatus",
    "LiquidationSet",
    "executable_labels",
    "hold_value_labels",
    "label_fills",
]
