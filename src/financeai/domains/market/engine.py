"""Fast event-driven replay (Numba) with a built-in hysteretic threshold policy.

Semantics (shared exactly with :mod:`financeai.domains.market.reference`):

* Events are delivered in receipt order; a decision is taken after each receipt
  *cohort* and after each fill acknowledgement (``HISTORICAL_REPLAY`` timing: compute is
  instantaneous at the decision time ``t``).
* An order submitted at ``t`` arrives at ``t + L``. In ``EXCHANGE_CLOCK`` mode it meets
  the last book with exchange time ``<= t + L`` (hidden from the controller); in
  ``RECEIPT_CLOCK`` mode the last book received by ``t + L``. It is an IOC limit order
  sweeping the opposite side, minus liquidity we consumed earlier.
* Its fill report reaches the controller at ``t + L + L_ack``; until then the venue has a
  pending order and no other order is sent there. The known position therefore always
  equals the true position when a new order is sent, so ``|position| <= cap`` holds.
* Entries are one-shot per episode (no chasing); exits and the forced flatten retry
  until flat. Nothing is submitted that would arrive after the execution cutoff ``T``.
* A scenario's ``depth_fraction`` haircuts every displayed level our sweeps may take
  (execution only; decisions still observe the displayed book).

The policy maps a per-event signal ``s`` to a state in {-1, 0, +1}: enter long when
``s >= enter`` (short when ``s <= -enter`` and shorts are allowed); after ``min_hold``
leave a long when ``s <= exit`` (a short when ``s >= -exit``); always leave after
``max_hold`` or on a stop. Venue ``v`` targets ``state * direction[v] * size[v]`` quantity units.

Clock-based exits (absolute duration cap, one-shot early markout check) run on the
*episode clock*: its origin is the first decision of the episode at which some venue's
acknowledged position has the episode's sign, including a position inherited from the
previous episode. Every acknowledgement is a decision, so for an episode that starts flat
this is the acknowledgement time of its first entry fill, and an episode whose entries never
fill (and that inherits nothing) has no origin; a re-entry while the previous exit is still
pending (cooldown 0, or an exit IOC that missed) adopts that position at its first decision,
so its clock starts at the re-trigger, before any own entry is sent. Like the stop, and unlike
the signal exit and the reverse flip, the cap and the early check are not deferred by
``min_hold_ns``. There are no timer events: a deadline is enforced at the first decision
(event cohort or acknowledgement) at or after it, and the liquidation it starts completes
later still (latency, partial fills).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError, InvariantViolationError
from financeai.core.jit import njit
from financeai.domains.market.execution import CONSUMPTION_SLOTS, sweep
from financeai.domains.market.ledger import Fills, Settlement, fill_cash, settle
from financeai.domains.market.replay_validation import prepare_signal, validate_clock_headroom
from financeai.domains.market.scenario import RiskLimits, Scenario, SessionPlan, TimingMode
from financeai.domains.market.stream import (
    BOOK_EVENT,
    EventStream,
    MarketData,
    book_validity,
    exchange_envelope,
)
from financeai.domains.market.tape import DEPTH

REASON_ENTRY: Final = 1
REASON_EXIT: Final = 2
REASON_FLATTEN: Final = 3
_NO_TIME: Final = np.iinfo(np.int64).max
_FOREVER: Final = 2**62


@dataclass(frozen=True, slots=True)
class ThresholdParams:
    direction: tuple[int, ...]
    size_units: tuple[int, ...]
    enter: float
    exit: float
    allow_short: bool = True
    min_hold_ns: int = 0
    max_hold_ns: int = _FOREVER
    cooldown_ns: int = 0
    stop_bps: float = 0.0
    """Exit when the reference venue's mid moves this many bps against us (0 disables)."""
    entry_slip_ticks: int = 0
    """IOC limit for entries: observed best opposite price +/- this many ticks."""
    exit_slip_ticks: int = 20
    size_units_cap: int = 1
    """Signal-strength sizing: an entry takes ``min(cap, max(1, floor(|s| / enter)))`` units of
    ``size_units`` (1 disables)."""
    depth_ticks: int = -1
    """Depth-aware sizing: cap each venue's entry at the *observed* opposite-side displayed depth
    within this many ticks of the best price (-1 disables; 0 = best level only)."""
    max_clips: int = 1
    """Staged accumulation: while the trigger persists in the episode's direction, add one more
    entry clip (the episode's first size) every ``clip_spacing_ns``, up to this many clips."""
    clip_spacing_ns: int = 0
    refresh_hold: bool = False
    """Restart the hold timer whenever the trigger fires again in the episode's direction."""
    reverse: bool = False
    """While in a position, an opposite trigger (inside the entry window, after ``min_hold_ns``)
    flips it: the open position is closed first and the opposite entry follows its acknowledgement.
    A flip to short needs ``allow_short``. Like every exit rule below, a flip at the same decision
    takes precedence (it closes the position and starts a new episode with its own clock)."""
    abs_cap_ns: int = 0
    """Absolute position-duration cap (0 disables): leave the episode at the first decision at or
    after ``origin + abs_cap_ns`` on the episode clock (see the module notes), regardless of
    ``min_hold_ns``. Unlike ``max_hold_ns`` it is never refreshed: hold refreshes, accumulation clips
    and re-triggers do not move the origin. An episode with no origin (its entries never filled and it
    inherited no position) is never capped; one that adopts the previous episode's position still
    awaiting liquidation is timed from its first decision."""
    early_check_ns: int = 0
    """One-shot early non-confirmation check (0 disables). At the first decision at or after
    ``origin + early_check_ns`` (regardless of ``min_hold_ns``) the observed markout on the reference
    venue (the first venue with a non-zero direction), ``s * (m - pbar) / pbar * 1e4`` bp, is compared
    with ``early_markout_bps``; strictly below it the episode is left (equality confirms). ``s`` is the
    sign of the acknowledged position there, ``m`` the mid of the last received book,
    ``pbar = sum(q) / sum(q / p)`` the inverse-contract average price of the episode's own acknowledged
    entry fills there (position-increasing orders submitted since the episode began; the close order
    of a reverse flip is not one). Runs once per episode; if the markout is not yet observable (no such
    fill, no valid book) the check waits for the first decision at which it is. The reference venue
    must therefore be traded (``size_units > 0`` there), or the check could never run."""
    early_markout_bps: float = 0.0
    """Threshold of the early check in bp (may be negative; ignored while ``early_check_ns`` is 0)."""

    def __post_init__(self) -> None:
        """Type, range and overflow checks. Both engines run these parameters unchanged, so the two
        bounds below keep the compiled kernel's int64 arithmetic equal to the Python twin's:

        * ``size_units[v] * size_units_cap * max_clips <= 2**62`` for every venue. The kernel forms
          ``units * size_units`` (``units <= size_units_cap``) and then ``clip_size * clips``
          (``clips <= max_clips``) in int64 *before* the target is clipped to the risk cap; a product
          past 2**63 wraps and can flip the target's sign (a long-only entry then sells), silently in
          compiled mode where Python ints would not. At or below 2**62 every intermediate and its
          negation stay inside int64.
        * every nanosecond field ``<= 2**62`` (``_FOREVER``, the default ``max_hold_ns``, about 146
          years). The one deadline the kernels add to a clock is ``t + cooldown_ns`` (holds, clip spacing
          and the episode-clock caps are compared with elapsed time). Replay admission separately
          checks that market clocks and the session drain leave room for every derived deadline.
        * every tick offset (``entry_slip_ticks``, ``exit_slip_ticks``, ``depth_ticks``) ``<= 2**62``. The
          kernels add them to int64 tick prices (``best +/- slip`` for the limit, ``best + depth`` for the
          displayed-depth probe); tick prices are far below 2**62, so the sum stays inside int64 where a
          larger offset would wrap in compiled mode while the Python twin keeps the exact int.
        """
        if isinstance(self.max_clips, bool) or not isinstance(self.max_clips, int) or self.max_clips < 1:
            raise InvalidConfigError("max_clips must be an integer >= 1")
        if isinstance(self.size_units_cap, bool) or not isinstance(self.size_units_cap, int) or self.size_units_cap < 1:
            raise InvalidConfigError("size_units_cap must be an integer >= 1")
        if isinstance(self.depth_ticks, bool) or not isinstance(self.depth_ticks, int) or self.depth_ticks < -1:
            raise InvalidConfigError("depth_ticks must be an integer >= -1")
        if len(self.direction) != len(self.size_units) or not self.direction:
            raise InvalidConfigError("direction and size_units need one entry per venue")
        if any(not isinstance(d, int) or d not in (-1, 0, 1) or isinstance(d, bool) for d in self.direction):
            raise InvalidConfigError("direction entries must be -1, 0 or +1")
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in self.size_units):
            raise InvalidConfigError("size_units entries must be non-negative integers")
        for name in ("enter", "exit", "stop_bps", "early_markout_bps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise InvalidConfigError(f"{name} must be a finite number")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise InvalidConfigError(f"{name} must be a finite number")
        if self.enter <= 0:
            raise InvalidConfigError("enter must be finite and positive; exit finite")
        for name in (
            "min_hold_ns",
            "max_hold_ns",
            "cooldown_ns",
            "entry_slip_ticks",
            "exit_slip_ticks",
            "clip_spacing_ns",
            "abs_cap_ns",
            "early_check_ns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InvalidConfigError(f"{name} must be a non-negative integer")
        if self.stop_bps < 0:
            raise InvalidConfigError("stop_bps must be finite and non-negative")
        if any(not isinstance(v, bool) for v in (self.allow_short, self.reverse, self.refresh_hold)):
            raise InvalidConfigError("allow_short, reverse and refresh_hold must be booleans")
        for name in ("min_hold_ns", "max_hold_ns", "cooldown_ns", "clip_spacing_ns", "abs_cap_ns", "early_check_ns"):
            value = getattr(self, name)
            if value > _FOREVER:
                raise InvalidConfigError(
                    f"{name} must be at most 2**62 ns", context={"field": name, "max": _FOREVER, "got": value}
                )
        for name in ("entry_slip_ticks", "exit_slip_ticks", "depth_ticks"):
            value = getattr(self, name)
            if value > _FOREVER:  # added to int64 tick prices in the kernels (see above)
                raise InvalidConfigError(
                    f"{name} must be at most 2**62 ticks", context={"field": name, "max": _FOREVER, "got": value}
                )
        max_target = 2**62  # per-venue bound of the int64 entry-size product (see above)
        for venue, size in enumerate(self.size_units):
            if size * self.size_units_cap * self.max_clips > max_target:
                raise InvalidConfigError(
                    "size_units * size_units_cap * max_clips must be at most 2**62 per venue",
                    context={
                        "field": "size_units",
                        "venue": venue,
                        "size_units": size,
                        "size_units_cap": self.size_units_cap,
                        "max_clips": self.max_clips,
                        "max": max_target,
                    },
                )
        traded = [v for v, d in enumerate(self.direction) if d != 0]
        if self.early_check_ns > 0 and traded and self.size_units[traded[0]] == 0:
            # the markout is priced from the episode's own entry fills on the reference venue: with no
            # size there the one-shot check would wait forever and the option would be silently inert
            raise InvalidConfigError(
                "early_check_ns needs size_units > 0 on the reference venue (the first with a non-zero direction)",
                context={"reference_venue": traded[0]},
            )


@dataclass(frozen=True, slots=True)
class Orders:
    order_id: NDArray[np.int64]
    venue: NDArray[np.int8]
    side: NDArray[np.int8]
    qty_units: NDArray[np.int64]
    limit_ticks: NDArray[np.int64]
    submit_ns: NDArray[np.int64]
    arrival_ns: NDArray[np.int64]
    ack_ns: NDArray[np.int64]
    filled_units: NDArray[np.int64]
    reason: NDArray[np.int8]
    book_row: NDArray[np.int64]
    """Venue book row the order met (-1 when none existed)."""

    def __len__(self) -> int:
        return int(self.order_id.shape[0])


@dataclass(frozen=True, slots=True)
class ReplayResult:
    fills: Fills
    orders: Orders
    settlement: Settlement
    decisions: int


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

(
    _I_STATE,
    _I_ENTRY_T,
    _I_COOLDOWN,
    _I_NORD,
    _I_NFILL,
    _I_DECISIONS,
    _I_ERROR,
    _I_REFV,
    _I_SETTLED,
    _I_CLIPS,
    _I_LAST_CLIP,
    _I_EP_START,
    _I_ORIGIN,
    _I_EARLY_DONE,
    _I_EP_QTY,
) = range(15)
_N_STATE_I: Final = 15
"""``_I_SETTLED`` = 1 when the last full decision changed nothing (no state change, no order,
no venue blocked by a missing book): until an acknowledgement, a hold expiry, an exit-signal
crossing, a stop, a due episode-clock deadline or the flatten time, re-running the decision is
provably a no-op.

Episode clock: ``_I_EP_START`` is the decision time that started the episode (never refreshed),
``_I_ORIGIN`` the clock origin (``_NO_TIME`` until the acknowledged position first has the episode's
sign), ``_I_EARLY_DONE`` whether the one-shot early check ran. ``_I_EP_QTY`` and ``st_f[1]`` sum
``q`` and ``q / p`` (ticks) over the episode's acknowledged entry fills on the reference venue;
``st_f[0]`` is the reference mid at entry (stop)."""


@njit()
def _entry_size(
    v: int,
    side: int,
    units: int,
    p_size: Any,
    p_depth: int,
    obs_row: Any,
    bpx: Any,
    bsz: Any,
    apx: Any,
    asz: Any,
    nb: Any,
    na: Any,
    valid: Any,
    lot: Any,
) -> int:
    """Episode size for venue ``v``: ``units * size`` capped (optionally) by observed depth."""
    size = int(units * p_size[v])
    if p_depth >= 0:
        depth = 0
        r = obs_row[v]
        if r >= 0 and valid[v][r]:
            if side > 0:
                lim = apx[v][r, 0] + p_depth
                for k in range(na[v][r]):
                    if apx[v][r, k] > lim:
                        break
                    depth += int(asz[v][r, k])
            else:
                lim = bpx[v][r, 0] - p_depth
                for k in range(nb[v][r]):
                    if bpx[v][r, k] < lim:
                        break
                    depth += int(bsz[v][r, k])
        depth = int((depth // lot[v]) * lot[v])
        if depth < size:
            size = depth
    return size


@njit()
def _decide(
    t: Any,
    sig: Any,
    xsig: Any,
    nv: Any,
    st_i: Any,
    st_f: Any,
    p_dir: Any,
    p_size: Any,
    p_enter: Any,
    p_exit: Any,
    p_short: Any,
    p_min_hold: Any,
    p_max_hold: Any,
    p_cooldown: Any,
    p_stop: Any,
    p_eslip: Any,
    p_xslip: Any,
    p_units: Any,
    p_depth: Any,
    p_clips: Any,
    p_spacing: Any,
    p_refresh: Any,
    p_reverse: Any,
    p_abs_cap: Any,
    p_early: Any,
    p_early_bps: Any,
    bpx: Any,
    bsz: Any,
    apx: Any,
    asz: Any,
    nb: Any,
    na: Any,
    valid: Any,
    exch_env: Any,
    book_recv: Any,
    lot: Any,
    tick: Any,
    fee: Any,
    inverse: Any,
    factor: Any,
    shortable: Any,
    fixed: Any,
    lat: Any,
    ack_lat: Any,
    timing: Any,
    ttl: Any,
    frac: Any,
    cap: Any,
    s_start: Any,
    s_entry: Any,
    s_flat: Any,
    s_exec: Any,
    obs_row: Any,
    known: Any,
    true_pos: Any,
    pend: Any,
    pend_ack: Any,
    pend_fill: Any,
    pend_inc: Any,
    pend_sub: Any,
    pend_inv: Any,
    tried: Any,
    ep_size: Any,
    ep_clip: Any,
    c_px: Any,
    c_amt: Any,
    c_exp: Any,
    out_px: Any,
    out_qty: Any,
    f_order: Any,
    f_venue: Any,
    f_side: Any,
    f_px: Any,
    f_qty: Any,
    f_submit: Any,
    f_arrival: Any,
    f_ack: Any,
    f_cash: Any,
    o_venue: Any,
    o_side: Any,
    o_qty: Any,
    o_limit: Any,
    o_submit: Any,
    o_arrival: Any,
    o_ack: Any,
    o_filled: Any,
    o_reason: Any,
    o_row: Any,
) -> None:
    st_i[_I_DECISIONS] += 1
    state = st_i[_I_STATE]
    desired = state
    flatten = t >= s_flat
    have_sig = sig == sig
    have_x = xsig == xsig
    rv = st_i[_I_REFV]
    if flatten:
        desired = 0
    elif state == 0:
        if t >= s_start and t < s_entry and t >= st_i[_I_COOLDOWN] and have_sig:
            if sig >= p_enter:
                desired = 1
            elif p_short and sig <= -p_enter:
                desired = -1
    else:
        held = t - st_i[_I_ENTRY_T]
        signal_exit = (
            held >= p_min_hold and have_x and ((state == 1 and xsig <= p_exit) or (state == -1 and xsig >= -p_exit))
        )
        if held >= p_max_hold or signal_exit:
            desired = 0
        if desired != 0 and p_stop > 0.0 and st_f[0] > 0.0:
            r = obs_row[rv]
            if r >= 0 and valid[rv][r]:
                mid = 0.5 * (bpx[rv][r, 0] + apx[rv][r, 0])
                adverse = -state * p_dir[rv] * (mid - st_f[0]) / st_f[0] * 1e4
                if adverse >= p_stop:
                    desired = 0
        origin = st_i[_I_ORIGIN]
        if desired != 0 and origin != _NO_TIME:
            since = t - origin  # episode clock: never refreshed
            if p_abs_cap > 0 and since >= p_abs_cap:
                desired = 0
            elif p_early > 0 and st_i[_I_EARLY_DONE] == 0 and since >= p_early:
                r = obs_row[rv]
                k = known[rv]
                if st_i[_I_EP_QTY] > 0 and k != 0 and r >= 0 and valid[rv][r]:  # observable markout
                    st_i[_I_EARLY_DONE] = 1
                    pbar = st_i[_I_EP_QTY] / st_f[1] if inverse[rv] else st_f[1] / st_i[_I_EP_QTY]
                    mid = 0.5 * (bpx[rv][r, 0] + apx[rv][r, 0])
                    s = 1 if k > 0 else -1
                    if s * (mid - pbar) / pbar * 1e4 < p_early_bps:
                        desired = 0
        if (
            p_reverse
            and have_sig
            and held >= p_min_hold
            and t >= s_start
            and t < s_entry
            and ((state == 1 and p_short and sig <= -p_enter) or (state == -1 and sig >= p_enter))
        ):
            desired = -state
    changed = desired != state
    submitted = False
    blocked = False
    if (
        desired == state
        and state != 0
        and (p_clips > 1 or p_refresh)
        and have_sig
        and ((state == 1 and sig >= p_enter) or (state == -1 and sig <= -p_enter))
    ):
        if p_refresh:
            st_i[_I_ENTRY_T] = t
        if st_i[_I_CLIPS] < p_clips and t >= s_start and t < s_entry and t - st_i[_I_LAST_CLIP] >= p_spacing:
            st_i[_I_CLIPS] += 1
            st_i[_I_LAST_CLIP] = t
            for v in range(nv):
                ep_size[v] = ep_clip[v] * st_i[_I_CLIPS]
                tried[v] = 0
            changed = True
    if desired != state:
        if desired == 0:
            st_i[_I_COOLDOWN] = t + p_cooldown
        else:
            st_i[_I_ENTRY_T] = t
            st_f[0] = 0.0
            r = obs_row[rv]
            if r >= 0 and valid[rv][r]:
                st_f[0] = 0.5 * (bpx[rv][r, 0] + apx[rv][r, 0])
            units = 1
            if p_units > 1:
                units = min(p_units, max(1, int(abs(sig) / p_enter)))
            st_i[_I_CLIPS] = 1
            st_i[_I_LAST_CLIP] = t
            for v in range(nv):
                tried[v] = 0
                ep_size[v] = _entry_size(
                    v, desired * p_dir[v], units, p_size, p_depth, obs_row, bpx, bsz, apx, asz, nb, na, valid, lot
                )
                ep_clip[v] = ep_size[v]
            st_i[_I_EP_START] = t  # a new episode (also a reverse flip): fresh clock and entry prices
            st_i[_I_ORIGIN] = _NO_TIME
            st_i[_I_EARLY_DONE] = 0
            st_i[_I_EP_QTY] = 0
            st_f[1] = 0.0
        st_i[_I_STATE] = desired
    state = desired
    if state != 0 and st_i[_I_ORIGIN] == _NO_TIME:
        # the clock starts once the acknowledged position has the episode's sign (a flip's old,
        # opposite position does not count); acknowledgements are always full decisions
        for v in range(nv):
            if state * p_dir[v] * known[v] > 0:
                st_i[_I_ORIGIN] = t
                break

    for v in range(nv):
        if pend[v] != 0:
            continue
        r = obs_row[v]
        if r < 0 or not valid[v][r]:
            blocked = True
            continue
        tgt = state * p_dir[v] * ep_size[v]
        if tgt > cap:
            tgt = cap
        elif tgt < -cap:
            tgt = -cap
        if not shortable[v] and tgt < 0:
            tgt = 0
        tgt = (abs(tgt) // lot[v]) * lot[v] * (1 if tgt >= 0 else -1)
        k = known[v]
        if k != 0 and tgt != 0 and (k > 0) != (tgt > 0):
            tgt = 0  # reduce to flat before reversing
        increasing = abs(tgt) > abs(k)
        if increasing and tried[v] != 0:
            continue
        delta = tgt - k
        if delta == 0:
            continue
        if t + lat > s_exec:
            continue
        side = 1 if delta > 0 else -1
        qty = abs(delta)
        slip = p_eslip if increasing else p_xslip
        limit = apx[v][r, 0] + slip if side > 0 else bpx[v][r, 0] - slip
        oid = st_i[_I_NORD]
        if oid >= o_venue.shape[0]:
            st_i[_I_ERROR] = 1
            return
        arrival = t + lat
        if timing == 0:
            row = np.searchsorted(exch_env[v], arrival, side="right") - 1
        else:
            row = np.searchsorted(book_recv[v], arrival, side="right") - 1
        filled = 0
        inv = 0.0  # sum of qty / price (ticks) over this order's fills, in level order
        if row >= 0 and valid[v][row]:
            if side > 0:
                n_out = sweep(
                    side,
                    qty,
                    limit,
                    apx[v][row],
                    asz[v][row],
                    na[v][row],
                    lot[v],
                    frac,
                    c_px[v, 1],
                    c_amt[v, 1],
                    c_exp[v, 1],
                    arrival,
                    ttl,
                    out_px,
                    out_qty,
                )
            else:
                n_out = sweep(
                    side,
                    qty,
                    limit,
                    bpx[v][row],
                    bsz[v][row],
                    nb[v][row],
                    lot[v],
                    frac,
                    c_px[v, 0],
                    c_amt[v, 0],
                    c_exp[v, 0],
                    arrival,
                    ttl,
                    out_px,
                    out_qty,
                )
            for i in range(n_out):
                fi = st_i[_I_NFILL]
                if fi >= f_order.shape[0]:
                    st_i[_I_ERROR] = 1
                    return
                f_order[fi] = oid
                f_venue[fi] = v
                f_side[fi] = side
                f_px[fi] = out_px[i]
                f_qty[fi] = out_qty[i]
                f_submit[fi] = t
                f_arrival[fi] = arrival
                f_ack[fi] = arrival + ack_lat
                f_cash[fi] = fill_cash(side, out_qty[i], out_px[i] * tick[v], fee[v], inverse[v], factor[v], fixed[v])
                st_i[_I_NFILL] = fi + 1
                filled += out_qty[i]
                inv += out_qty[i] / out_px[i] if inverse[v] else out_qty[i] * float(out_px[i])
        else:
            row = -1 if row < 0 else row
        o_venue[oid] = v
        o_side[oid] = side
        o_qty[oid] = qty
        o_limit[oid] = limit
        o_submit[oid] = t
        o_arrival[oid] = arrival
        o_ack[oid] = arrival + ack_lat
        o_filled[oid] = filled
        o_reason[oid] = 3 if flatten else (1 if increasing else 2)
        o_row[oid] = row
        st_i[_I_NORD] = oid + 1
        true_pos[v] += side * filled
        pend[v] = 1
        pend_ack[v] = arrival + ack_lat
        pend_fill[v] = side * filled
        pend_inc[v] = 1 if increasing else 0
        pend_sub[v] = t
        pend_inv[v] = inv
        submitted = True
        if increasing:
            tried[v] = 1
    st_i[_I_SETTLED] = 0 if (changed or submitted or blocked) else 1


@njit()
def _next_ack(pend: NDArray[np.int64], pend_ack: NDArray[np.int64], nv: int) -> int:
    best = _NO_TIME
    for v in range(nv):
        if pend[v] != 0 and pend_ack[v] < best:
            best = int(pend_ack[v])
    return best


@njit()
def _apply_acks(
    t: int,
    pend: NDArray[np.int64],
    pend_ack: NDArray[np.int64],
    pend_fill: NDArray[np.int64],
    pend_inc: NDArray[np.int64],
    pend_sub: NDArray[np.int64],
    pend_inv: NDArray[np.float64],
    known: NDArray[np.int64],
    nv: int,
    rv: int,
    st_i: NDArray[np.int64],
    st_f: NDArray[np.float64],
) -> int:
    """Deliver the fill reports due by ``t``: the acknowledged positions and, for the early
    markout check, the entry fills on the reference venue of orders submitted since the current
    episode began (an older order's fills are not this episode's, even if acknowledged in it)."""
    applied = 0
    for v in range(nv):
        if pend[v] != 0 and pend_ack[v] <= t:
            known[v] += pend_fill[v]
            if v == rv and pend_inc[v] != 0 and pend_fill[v] != 0 and pend_sub[v] >= st_i[_I_EP_START]:
                st_i[_I_EP_QTY] += abs(pend_fill[v])
                st_f[1] += pend_inv[v]
            pend[v] = 0
            pend_fill[v] = 0
            applied += 1
    return applied


@njit()
def _run_kernel(
    ev_recv: Any,
    ev_venue: Any,
    ev_kind: Any,
    ev_row: Any,
    signal: Any,
    xsignal: Any,
    p_dir: Any,
    p_size: Any,
    p_enter: Any,
    p_exit: Any,
    p_short: Any,
    p_min_hold: Any,
    p_max_hold: Any,
    p_cooldown: Any,
    p_stop: Any,
    p_eslip: Any,
    p_xslip: Any,
    p_units: Any,
    p_depth: Any,
    p_clips: Any,
    p_spacing: Any,
    p_refresh: Any,
    p_reverse: Any,
    p_abs_cap: Any,
    p_early: Any,
    p_early_bps: Any,
    bpx: Any,
    bsz: Any,
    apx: Any,
    asz: Any,
    nb: Any,
    na: Any,
    valid: Any,
    exch_env: Any,
    book_recv: Any,
    lot: Any,
    tick: Any,
    fee: Any,
    inverse: Any,
    factor: Any,
    shortable: Any,
    fixed: Any,
    lat: Any,
    ack_lat: Any,
    timing: Any,
    ttl: Any,
    frac: Any,
    cap: Any,
    s_start: Any,
    s_entry: Any,
    s_flat: Any,
    s_exec: Any,
    s_end: Any,
    st_i: Any,
    st_f: Any,
    known: Any,
    true_pos: Any,
    f_order: Any,
    f_venue: Any,
    f_side: Any,
    f_px: Any,
    f_qty: Any,
    f_submit: Any,
    f_arrival: Any,
    f_ack: Any,
    f_cash: Any,
    o_venue: Any,
    o_side: Any,
    o_qty: Any,
    o_limit: Any,
    o_submit: Any,
    o_arrival: Any,
    o_ack: Any,
    o_filled: Any,
    o_reason: Any,
    o_row: Any,
) -> None:
    nv = p_dir.shape[0]
    obs_row = np.full(nv, -1, np.int64)
    pend = np.zeros(nv, np.int64)
    pend_ack = np.zeros(nv, np.int64)
    pend_fill = np.zeros(nv, np.int64)
    pend_inc = np.zeros(nv, np.int64)
    pend_sub = np.zeros(nv, np.int64)
    pend_inv = np.zeros(nv, np.float64)
    tried = np.zeros(nv, np.int64)
    ep_size = p_size.copy()
    ep_clip = p_size.copy()
    c_px = np.zeros((nv, 2, CONSUMPTION_SLOTS), np.int32)
    c_amt = np.zeros((nv, 2, CONSUMPTION_SLOTS), np.int64)
    c_exp = np.zeros((nv, 2, CONSUMPTION_SLOTS), np.int64)
    out_px = np.zeros(DEPTH, np.int32)
    out_qty = np.zeros(DEPTH, np.int64)
    rv = 0
    for v in range(nv):
        if p_dir[v] != 0:
            rv = v
            break
    st_i[_I_REFV] = rv
    st_i[_I_ORIGIN] = _NO_TIME
    n = ev_recv.shape[0]
    e = 0
    last_sig = np.nan
    last_xsig = np.nan
    flat_timer = s_flat  # one session-timer decision at the flatten time (event-density independent)
    while e < n:
        t = ev_recv[e]
        while True:
            ta = _next_ack(pend, pend_ack, nv)
            if flat_timer < ta:
                ta = flat_timer
            if ta >= t:
                break
            if ta == flat_timer:
                flat_timer = _NO_TIME
            if _apply_acks(ta, pend, pend_ack, pend_fill, pend_inc, pend_sub, pend_inv, known, nv, rv, st_i, st_f) > 0:
                st_i[_I_SETTLED] = 0
            _decide(
                ta,
                last_sig,
                last_xsig,
                nv,
                st_i,
                st_f,
                p_dir,
                p_size,
                p_enter,
                p_exit,
                p_short,
                p_min_hold,
                p_max_hold,
                p_cooldown,
                p_stop,
                p_eslip,
                p_xslip,
                p_units,
                p_depth,
                p_clips,
                p_spacing,
                p_refresh,
                p_reverse,
                p_abs_cap,
                p_early,
                p_early_bps,
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
                tick,
                fee,
                inverse,
                factor,
                shortable,
                fixed,
                lat,
                ack_lat,
                timing,
                ttl,
                frac,
                cap,
                s_start,
                s_entry,
                s_flat,
                s_exec,
                obs_row,
                known,
                true_pos,
                pend,
                pend_ack,
                pend_fill,
                pend_inc,
                pend_sub,
                pend_inv,
                tried,
                ep_size,
                ep_clip,
                c_px,
                c_amt,
                c_exp,
                out_px,
                out_qty,
                f_order,
                f_venue,
                f_side,
                f_px,
                f_qty,
                f_submit,
                f_arrival,
                f_ack,
                f_cash,
                o_venue,
                o_side,
                o_qty,
                o_limit,
                o_submit,
                o_arrival,
                o_ack,
                o_filled,
                o_reason,
                o_row,
            )
            if st_i[_I_ERROR] != 0:
                return
        while e < n and ev_recv[e] == t:
            if ev_kind[e] == BOOK_EVENT:
                obs_row[ev_venue[e]] = ev_row[e]
            last_sig = signal[e]
            last_xsig = xsignal[e]
            e += 1
        if _apply_acks(t, pend, pend_ack, pend_fill, pend_inc, pend_sub, pend_inv, known, nv, rv, st_i, st_f) > 0:
            st_i[_I_SETTLED] = 0
        if flat_timer == t:
            flat_timer = _NO_TIME  # this cohort's decision is the timer decision
        state_now = st_i[_I_STATE]
        # Fast path 1: flat, idle and no entry trigger -> the full decision provably does nothing.
        if state_now == 0 and not (last_sig >= p_enter or (p_short and last_sig <= -p_enter)):
            idle = True
            for v in range(nv):
                if pend[v] != 0 or known[v] != 0:
                    idle = False
                    break
            if idle:
                st_i[_I_DECISIONS] += 1
                continue
        # Fast path 2: in position and settled; nothing can change before an ack, the hold
        # expiry, an exit-signal crossing, a due episode-clock deadline (absolute cap, or an early
        # check still to run) or the flatten time (stops always take the full path).
        elif state_now != 0 and st_i[_I_SETTLED] == 1 and t < s_flat and p_stop == 0.0:
            held = t - st_i[_I_ENTRY_T]
            origin = st_i[_I_ORIGIN]
            clock_due = origin != _NO_TIME and (
                (p_abs_cap > 0 and t - origin >= p_abs_cap)
                or (p_early > 0 and st_i[_I_EARLY_DONE] == 0 and t - origin >= p_early)
            )
            if held < p_max_hold and not clock_due:
                exit_now = (
                    held >= p_min_hold
                    and last_xsig == last_xsig
                    and ((state_now == 1 and last_xsig <= p_exit) or (state_now == -1 and last_xsig >= -p_exit))
                )
                retrig = (
                    (p_clips > 1 or p_refresh)
                    and last_sig == last_sig
                    and ((state_now == 1 and last_sig >= p_enter) or (state_now == -1 and last_sig <= -p_enter))
                )
                if (
                    p_reverse
                    and last_sig == last_sig
                    and ((state_now == 1 and last_sig <= -p_enter) or (state_now == -1 and last_sig >= p_enter))
                ):
                    retrig = True
                if not exit_now and not retrig:
                    quiet = True
                    for v in range(nv):
                        if pend[v] != 0:
                            quiet = False
                            break
                    if quiet:
                        st_i[_I_DECISIONS] += 1
                        continue
        _decide(
            t,
            last_sig,
            last_xsig,
            nv,
            st_i,
            st_f,
            p_dir,
            p_size,
            p_enter,
            p_exit,
            p_short,
            p_min_hold,
            p_max_hold,
            p_cooldown,
            p_stop,
            p_eslip,
            p_xslip,
            p_units,
            p_depth,
            p_clips,
            p_spacing,
            p_refresh,
            p_reverse,
            p_abs_cap,
            p_early,
            p_early_bps,
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
            tick,
            fee,
            inverse,
            factor,
            shortable,
            fixed,
            lat,
            ack_lat,
            timing,
            ttl,
            frac,
            cap,
            s_start,
            s_entry,
            s_flat,
            s_exec,
            obs_row,
            known,
            true_pos,
            pend,
            pend_ack,
            pend_fill,
            pend_inc,
            pend_sub,
            pend_inv,
            tried,
            ep_size,
            ep_clip,
            c_px,
            c_amt,
            c_exp,
            out_px,
            out_qty,
            f_order,
            f_venue,
            f_side,
            f_px,
            f_qty,
            f_submit,
            f_arrival,
            f_ack,
            f_cash,
            o_venue,
            o_side,
            o_qty,
            o_limit,
            o_submit,
            o_arrival,
            o_ack,
            o_filled,
            o_reason,
            o_row,
        )
        if st_i[_I_ERROR] != 0:
            return
    while True:
        ta = _next_ack(pend, pend_ack, nv)
        if flat_timer < ta:
            ta = flat_timer
        if ta == _NO_TIME or ta > s_end:
            break
        if ta == flat_timer:
            flat_timer = _NO_TIME
        _apply_acks(ta, pend, pend_ack, pend_fill, pend_inc, pend_sub, pend_inv, known, nv, rv, st_i, st_f)
        _decide(
            ta,
            last_sig,
            last_xsig,
            nv,
            st_i,
            st_f,
            p_dir,
            p_size,
            p_enter,
            p_exit,
            p_short,
            p_min_hold,
            p_max_hold,
            p_cooldown,
            p_stop,
            p_eslip,
            p_xslip,
            p_units,
            p_depth,
            p_clips,
            p_spacing,
            p_refresh,
            p_reverse,
            p_abs_cap,
            p_early,
            p_early_bps,
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
            tick,
            fee,
            inverse,
            factor,
            shortable,
            fixed,
            lat,
            ack_lat,
            timing,
            ttl,
            frac,
            cap,
            s_start,
            s_entry,
            s_flat,
            s_exec,
            obs_row,
            known,
            true_pos,
            pend,
            pend_ack,
            pend_fill,
            pend_inc,
            pend_sub,
            pend_inv,
            tried,
            ep_size,
            ep_clip,
            c_px,
            c_amt,
            c_exp,
            out_px,
            out_qty,
            f_order,
            f_venue,
            f_side,
            f_px,
            f_qty,
            f_submit,
            f_arrival,
            f_ack,
            f_cash,
            o_venue,
            o_side,
            o_qty,
            o_limit,
            o_submit,
            o_arrival,
            o_ack,
            o_filled,
            o_reason,
            o_row,
        )
        if st_i[_I_ERROR] != 0:
            return


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedMarket:
    """Per-venue arrays arranged for the kernels (reusable across many replays)."""

    data: MarketData
    stream: EventStream
    bpx: tuple[NDArray[np.int32], ...]
    bsz: tuple[NDArray[np.int32], ...]
    apx: tuple[NDArray[np.int32], ...]
    asz: tuple[NDArray[np.int32], ...]
    nb: tuple[NDArray[np.int8], ...]
    na: tuple[NDArray[np.int8], ...]
    valid: tuple[NDArray[np.bool_], ...]
    exch_env: tuple[NDArray[np.int64], ...]
    book_recv: tuple[NDArray[np.int64], ...]
    lot: NDArray[np.int64]
    tick: NDArray[np.float64]
    inverse: NDArray[np.bool_]
    cash_factor: NDArray[np.float64]
    shortable: NDArray[np.bool_]

    @classmethod
    def build(cls, data: MarketData, stream: EventStream | None = None) -> PreparedMarket:
        if stream is not None:
            stream.validate_for(data)
        books = [t.book for t in data.venues]
        lot = np.array([t.instrument.lot_size for t in data.venues], np.int64)
        tick = np.array([float(t.instrument.tick_size) for t in data.venues], np.float64)
        return cls(
            data=data,
            stream=stream if stream is not None else EventStream.build(data),
            bpx=tuple(b.bid_px for b in books),
            bsz=tuple(b.bid_sz for b in books),
            apx=tuple(b.ask_px for b in books),
            asz=tuple(b.ask_sz for b in books),
            nb=tuple(b.n_bid for b in books),
            na=tuple(b.n_ask for b in books),
            valid=tuple(book_validity(t) for t in data.venues),
            exch_env=tuple(exchange_envelope(t) for t in data.venues),
            book_recv=tuple(b.recv_ns for b in books),
            lot=lot,
            tick=tick,
            inverse=np.array([t.instrument.inverse for t in data.venues], np.bool_),
            cash_factor=np.array([t.instrument.cash_factor for t in data.venues], np.float64),
            shortable=np.array([t.instrument.shortable for t in data.venues], np.bool_),
        )


def _fees(data: MarketData, scenario: Scenario) -> NDArray[np.float64]:
    return np.array([float(scenario.fee_for(t.instrument)) for t in data.venues], np.float64)


def run_replay(
    data: MarketData | PreparedMarket,
    signal: NDArray[np.float64],
    params: ThresholdParams,
    scenario: Scenario,
    session: SessionPlan,
    limits: RiskLimits,
    *,
    stream: EventStream | None = None,
    exit_signal: NDArray[np.float64] | None = None,
) -> ReplayResult:
    """Replay ``params`` driven by a per-event ``signal`` (aligned with the event stream).

    ``exit_signal`` (same alignment) drives the exit rule instead of ``signal`` when given,
    e.g. a slow gap-closure measure while a fast trigger drives entries. A ``PreparedMarket``
    already contains its stream; pass ``stream`` only when supplying raw ``MarketData``."""
    if isinstance(data, PreparedMarket) and stream is not None:
        raise InvalidConfigError("a prepared market already defines its event stream")
    prep = data if isinstance(data, PreparedMarket) else PreparedMarket.build(data, stream)
    nv = len(prep.data)
    if len(params.direction) != nv:
        raise InvalidConfigError("params need one direction/size per venue", context={"venues": nv})
    last = int(prep.stream.recv_ns[-1]) if len(prep.stream) else 0
    validate_clock_headroom(last, scenario, session, cooldown_ns=params.cooldown_ns)
    sig = prepare_signal(signal, len(prep.stream), name="signal")
    xsig = sig if exit_signal is None else prepare_signal(exit_signal, len(prep.stream), name="exit_signal")
    fee = _fees(prep.data, scenario)
    fixed = np.array(
        [float(scenario.fee_per_unit_for(t.instrument)) * float(t.instrument.quantity_step) for t in prep.data.venues],
        np.float64,
    )
    capacity_orders, capacity_fills = 1 << 14, 1 << 15
    while True:
        st_i = np.zeros(_N_STATE_I, np.int64)
        st_f = np.zeros(2, np.float64)
        known = np.zeros(nv, np.int64)
        true_pos = np.zeros(nv, np.int64)
        f = [
            np.zeros(capacity_fills, d)
            for d in (np.int64, np.int8, np.int8, np.int32, np.int64, np.int64, np.int64, np.int64, np.float64)
        ]
        o = [
            np.zeros(capacity_orders, d)
            for d in (np.int8, np.int8, np.int64, np.int64, np.int64, np.int64, np.int64, np.int64, np.int8, np.int64)
        ]
        _run_kernel(
            prep.stream.recv_ns,
            prep.stream.venue,
            prep.stream.kind,
            prep.stream.row,
            sig,
            xsig,
            np.array(params.direction, np.int64),
            np.array(params.size_units, np.int64),
            float(params.enter),
            float(params.exit),
            bool(params.allow_short),
            params.min_hold_ns,
            params.max_hold_ns,
            params.cooldown_ns,
            float(params.stop_bps),
            params.entry_slip_ticks,
            params.exit_slip_ticks,
            params.size_units_cap,
            params.depth_ticks,
            params.max_clips,
            params.clip_spacing_ns,
            bool(params.refresh_hold),
            bool(params.reverse),
            params.abs_cap_ns,
            params.early_check_ns,
            float(params.early_markout_bps),
            prep.bpx,
            prep.bsz,
            prep.apx,
            prep.asz,
            prep.nb,
            prep.na,
            prep.valid,
            prep.exch_env,
            prep.book_recv,
            prep.lot,
            prep.tick,
            fee,
            prep.inverse,
            prep.cash_factor,
            prep.shortable,
            fixed,
            scenario.order_latency_ns,
            scenario.ack_ns,
            int(scenario.timing),
            scenario.consumption_ttl_ns,
            float(scenario.depth_fraction),
            limits.max_abs_position_units,
            session.start_ns,
            session.entry_cutoff_ns,
            session.flatten_start_ns,
            session.exec_cutoff_ns,
            session.end_ns,
            st_i,
            st_f,
            known,
            true_pos,
            *f,
            *o,
        )
        if st_i[_I_ERROR] == 0:
            break
        capacity_orders *= 4
        capacity_fills *= 4
    n_orders, n_fills = int(st_i[_I_NORD]), int(st_i[_I_NFILL])
    fills = Fills(*(np.ascontiguousarray(a[:n_fills]) for a in f))
    orders = Orders(np.arange(n_orders, dtype=np.int64), *(np.ascontiguousarray(a[:n_orders]) for a in o))
    instruments = prep.data.instruments
    fees = tuple(scenario.fee_for(i) for i in instruments)
    settlement = settle(
        fills,
        instruments,
        fees,
        cap_units=limits.max_abs_position_units,
        fees_per_unit=tuple(scenario.fee_per_unit_for(i) for i in instruments),
    )
    if settlement.final_position_units != tuple(int(x) for x in true_pos):
        raise InvariantViolationError("ledger positions disagree with simulator inventory")
    return ReplayResult(fills=fills, orders=orders, settlement=settlement, decisions=int(st_i[_I_DECISIONS]))


__all__ = [
    "REASON_ENTRY",
    "REASON_EXIT",
    "REASON_FLATTEN",
    "Orders",
    "PreparedMarket",
    "ReplayResult",
    "ThresholdParams",
    "TimingMode",
    "run_replay",
]
