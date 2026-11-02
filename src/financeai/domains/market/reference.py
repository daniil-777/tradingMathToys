"""Reference replay engine: plain Python, generic policies, independent sweep.

This engine is the oracle for :mod:`financeai.domains.market.engine` and the extension
point for arbitrary (slow) Python policies. It deliberately re-implements the event
loop, acknowledgement handling, execution layer and depth sweep without calling the
compiled kernels, so exact parity is evidence that both are right.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError
from financeai.domains.market.engine import (
    REASON_ENTRY,
    REASON_EXIT,
    REASON_FLATTEN,
    Orders,
    ReplayResult,
    ThresholdParams,
)
from financeai.domains.market.ledger import Fills, settle
from financeai.domains.market.replay_validation import prepare_signal, validate_clock_headroom
from financeai.domains.market.scenario import RiskLimits, Scenario, SessionPlan, TimingMode
from financeai.domains.market.stream import (
    BOOK_EVENT,
    EventStream,
    MarketData,
    book_validity,
    exchange_envelope,
)


@dataclass(frozen=True, slots=True)
class FillReport:
    """One fill (one price level of an order) as the controller learns it: with the order's
    acknowledgement, never at the exchange arrival."""

    venue: int
    side: int
    px_ticks: int
    qty_units: int
    submit_ns: int
    ack_ns: int
    reason: int
    """The order's ``REASON_*``: entries are exactly the position-increasing orders."""


@dataclass(frozen=True)
class ControllerView:
    """What a policy may see at decision time ``t``: received books, acknowledged
    positions, pending flags, newly acknowledged fills and the latest signal. Never the
    simulator state."""

    t: int
    signal: float
    observed_row: tuple[int, ...]
    known_units: tuple[int, ...]
    pending: tuple[bool, ...]
    data: MarketData
    valid: tuple[NDArray[np.bool_], ...]
    session: SessionPlan
    exit_signal: float = float("nan")
    """Signal driving exits (defaults to NaN; the engine passes the entry signal if none given)."""
    acked_fills: tuple[FillReport, ...] = ()
    """Fills whose acknowledgement arrived since the previous decision (in venue order, then level
    order). Every acknowledgement is followed by a decision at its time, so a stateful policy sees
    each fill exactly once, at ``t == ack_ns``."""

    def has_book(self, venue: int) -> bool:
        r = self.observed_row[venue]
        return r >= 0 and bool(self.valid[venue][r])

    def best_bid(self, venue: int) -> int:
        return int(self.data.venues[venue].book.bid_px[self.observed_row[venue], 0])

    def best_ask(self, venue: int) -> int:
        return int(self.data.venues[venue].book.ask_px[self.observed_row[venue], 0])

    def mid_ticks(self, venue: int) -> float:
        return 0.5 * (self.best_bid(venue) + self.best_ask(venue))

    def depth_within(self, venue: int, side: int, ticks: int) -> int:
        """Displayed opposite-side quantity within ``ticks`` of the best price (asks for a buy)."""
        book = self.data.venues[venue].book
        r = self.observed_row[venue]
        total = 0
        if side > 0:
            limit = int(book.ask_px[r, 0]) + ticks
            for k in range(int(book.n_ask[r])):
                if int(book.ask_px[r, k]) > limit:
                    break
                total += int(book.ask_sz[r, k])
        else:
            limit = int(book.bid_px[r, 0]) - ticks
            for k in range(int(book.n_bid[r])):
                if int(book.bid_px[r, k]) < limit:
                    break
                total += int(book.bid_sz[r, k])
        return total


@dataclass(frozen=True)
class PolicyAction:
    targets_units: tuple[int, ...]
    new_episode: bool = False
    """Starting a new exposure episode re-enables one entry attempt per venue."""
    entry_slip_ticks: int = 0
    exit_slip_ticks: int = 20


class Policy(Protocol):
    def act(self, view: ControllerView) -> PolicyAction: ...


@dataclass
class ThresholdPolicy:
    """Python twin of the compiled hysteretic threshold policy.

    The episode clock (absolute cap, early markout check) is kept from what the controller
    sees: ``origin`` is set at the first decision of the episode at which the acknowledged
    position has the episode's sign (a position inherited from the previous episode counts, so
    a re-entry while that exit is still pending is timed from the re-trigger), and the
    entry-price sums only take acknowledged fills of entry orders submitted since the episode
    began (``episode_start``; a reverse flip's close order is an exit, so it is not priced).
    Like the stop, neither clock exit waits for ``min_hold_ns``."""

    params: ThresholdParams
    state: int = 0
    entry_t: int = 0
    cooldown_until: int = 0
    entry_mid: float = 0.0
    ref_venue: int = field(init=False)
    episode_size: list[int] = field(init=False)
    clip_size: list[int] = field(init=False)
    clips: int = 0
    last_clip_t: int = 0
    episode_start: int = 0
    origin: int | None = None
    early_done: bool = False
    entry_qty: int = 0
    """Sum of ``q`` over the episode's acknowledged entry fills on the reference venue."""
    entry_inv: float = 0.0
    """Sum of ``q / p`` (ticks) over the same fills, accumulated order by order in level order."""

    def __post_init__(self) -> None:
        nz = [v for v, d in enumerate(self.params.direction) if d != 0]
        self.ref_venue = nz[0] if nz else 0
        self.episode_size = list(self.params.size_units)
        self.clip_size = list(self.params.size_units)

    def act(self, view: ControllerView) -> PolicyAction:
        p, t, sig = self.params, view.t, view.signal
        s = view.session
        rv = self.ref_venue
        order_qty, order_inv = 0, 0.0  # at most one order per venue is acknowledged per decision
        for f in view.acked_fills:
            if f.venue == rv and f.reason == REASON_ENTRY and f.submit_ns >= self.episode_start:
                order_qty += f.qty_units
                order_inv += (
                    f.qty_units / f.px_ticks
                    if view.data.venues[rv].instrument.inverse
                    else f.qty_units * float(f.px_ticks)
                )
        if order_qty:
            self.entry_qty += order_qty
            self.entry_inv += order_inv
        have_sig = sig == sig
        desired = self.state
        flatten = t >= s.flatten_start_ns
        if flatten:
            desired = 0
        elif self.state == 0:
            if s.start_ns <= t < s.entry_cutoff_ns and t >= self.cooldown_until and have_sig:
                if sig >= p.enter:
                    desired = 1
                elif p.allow_short and sig <= -p.enter:
                    desired = -1
        else:
            held = t - self.entry_t
            if held >= p.max_hold_ns:
                desired = 0
            elif held >= p.min_hold_ns and view.exit_signal == view.exit_signal:
                xs = view.exit_signal
                if (self.state == 1 and xs <= p.exit) or (self.state == -1 and xs >= -p.exit):
                    desired = 0
            if desired != 0 and p.stop_bps > 0.0 and self.entry_mid > 0.0 and view.has_book(self.ref_venue):
                mid = view.mid_ticks(self.ref_venue)
                d = p.direction[self.ref_venue]
                adverse = -self.state * d * (mid - self.entry_mid) / self.entry_mid * 1e4
                if adverse >= p.stop_bps:
                    desired = 0
            if desired != 0 and self.origin is not None:
                since = t - self.origin  # episode clock: never refreshed
                if p.abs_cap_ns > 0 and since >= p.abs_cap_ns:
                    desired = 0  # absolute duration cap
                elif p.early_check_ns > 0 and not self.early_done and since >= p.early_check_ns:
                    k = view.known_units[rv]
                    if self.entry_qty > 0 and k != 0 and view.has_book(rv):  # observable markout
                        self.early_done = True
                        pbar = (
                            self.entry_qty / self.entry_inv
                            if view.data.venues[rv].instrument.inverse
                            else self.entry_inv / self.entry_qty
                        )
                        sign = 1 if k > 0 else -1
                        if sign * (view.mid_ticks(rv) - pbar) / pbar * 1e4 < p.early_markout_bps:
                            desired = 0  # not confirmed early enough
            if (
                p.reverse
                and have_sig
                and held >= p.min_hold_ns
                and s.start_ns <= t < s.entry_cutoff_ns
                and ((self.state == 1 and p.allow_short and sig <= -p.enter) or (self.state == -1 and sig >= p.enter))
            ):
                desired = -self.state  # flip: the order layer closes first, the opposite entry follows
        new_episode = False
        if (
            desired == self.state
            and self.state != 0
            and (p.max_clips > 1 or p.refresh_hold)
            and have_sig
            and ((self.state == 1 and sig >= p.enter) or (self.state == -1 and sig <= -p.enter))
        ):
            if p.refresh_hold:
                self.entry_t = t
            if (
                self.clips < p.max_clips
                and s.start_ns <= t < s.entry_cutoff_ns
                and t - self.last_clip_t >= p.clip_spacing_ns
            ):
                self.clips += 1
                self.last_clip_t = t
                self.episode_size = [c * self.clips for c in self.clip_size]
                new_episode = True  # re-enables one increasing order per venue
        if desired != self.state:
            if desired == 0:
                self.cooldown_until = t + p.cooldown_ns
            else:
                self.entry_t = t
                self.entry_mid = view.mid_ticks(self.ref_venue) if view.has_book(self.ref_venue) else 0.0
                new_episode = True
                units = min(p.size_units_cap, max(1, int(abs(sig) / p.enter))) if p.size_units_cap > 1 else 1
                lots = [t_.instrument.lot_size for t_ in view.data.venues]
                for v, (d, size) in enumerate(zip(p.direction, p.size_units, strict=True)):
                    want = units * size
                    if p.depth_ticks >= 0:
                        depth = view.depth_within(v, desired * d, p.depth_ticks) if view.has_book(v) else 0
                        want = min(want, (depth // lots[v]) * lots[v])
                    self.episode_size[v] = want
                self.clip_size = list(self.episode_size)
                self.clips = 1
                self.last_clip_t = t
                self.episode_start = t  # also a reverse flip: a fresh clock and fresh entry prices
                self.origin = None
                self.early_done = False
                self.entry_qty = 0
                self.entry_inv = 0.0
            self.state = desired
        if (
            self.state != 0
            and self.origin is None
            and any(self.state * d * k > 0 for d, k in zip(p.direction, view.known_units, strict=True))
        ):
            self.origin = t  # the acknowledged position first has the episode's sign
        targets = tuple(self.state * d * size for d, size in zip(p.direction, self.episode_size, strict=True))
        return PolicyAction(targets, new_episode, p.entry_slip_ticks, p.exit_slip_ticks)


def _sweep_reference(
    side: int,
    qty: int,
    limit: int,
    px: Sequence[int],
    sz: Sequence[int],
    levels: int,
    lot: int,
    depth_fraction: float,
    consumed: dict[int, list[list[int]]],
    now: int,
    ttl: int,
) -> list[tuple[int, int]]:
    fills: list[tuple[int, int]] = []
    remaining = qty
    for k in range(levels):
        if remaining <= 0:
            break
        price = int(px[k])
        if (side > 0 and price > limit) or (side < 0 and price < limit):
            break
        shown = int(sz[k])
        if depth_fraction < 1.0:  # haircut of the displayed level, before netting our own takes
            shown = int(depth_fraction * shown) // lot * lot
        taken = sum(amount for amount, expiry in consumed.get(price, []) if expiry > now)
        available = ((shown - taken) // lot) * lot
        if available <= 0:
            continue
        take = min(remaining, available)
        fills.append((price, take))
        remaining -= take
        active = [entry for entry in consumed.get(price, []) if entry[1] > now]
        if active:
            active[0][0] += take
            active[0][1] = now + ttl
            consumed[price] = active
        else:
            consumed[price] = [[take, now + ttl]]
    return fills


def run_reference(
    data: MarketData,
    signal: NDArray[np.float64],
    policy: Policy,
    scenario: Scenario,
    session: SessionPlan,
    limits: RiskLimits,
    *,
    stream: EventStream | None = None,
    exit_signal: NDArray[np.float64] | None = None,
) -> ReplayResult:
    if stream is not None:
        stream.validate_for(data)
    ev = stream if stream is not None else EventStream.build(data)
    nv = len(data)
    if isinstance(policy, ThresholdPolicy) and len(policy.params.direction) != nv:
        raise InvalidConfigError("params need one direction/size per venue", context={"venues": nv})
    cooldown = policy.params.cooldown_ns if isinstance(policy, ThresholdPolicy) else 0
    validate_clock_headroom(int(ev.recv_ns[-1]) if len(ev) else 0, scenario, session, cooldown_ns=cooldown)
    signal = prepare_signal(signal, len(ev), name="signal")
    exit_signal = signal if exit_signal is None else prepare_signal(exit_signal, len(ev), name="exit_signal")
    valid = tuple(book_validity(t) for t in data.venues)
    env = tuple(exchange_envelope(t) for t in data.venues)
    lat, ack_lat = scenario.order_latency_ns, scenario.ack_ns
    fees = tuple(scenario.fee_for(i) for i in data.instruments)
    fee_f = [float(f) for f in fees]
    ticks = [float(i.tick_size) for i in data.instruments]
    lots = [i.lot_size for i in data.instruments]
    cap = limits.max_abs_position_units

    observed = [-1] * nv
    known = [0] * nv
    true_pos = [0] * nv
    pending: list[tuple[int, int] | None] = [None] * nv  # (ack time, signed fill)
    pending_reports: list[list[FillReport]] = [[] for _ in range(nv)]
    inbox: list[FillReport] = []  # acknowledged since the previous decision
    tried = [False] * nv
    consumed: list[dict[int, dict[int, list[list[int]]]]] = [{1: {}, -1: {}} for _ in range(nv)]
    fill_rows: list[tuple[int, int, int, int, int, int, int, int, float]] = []
    order_rows: list[tuple[int, int, int, int, int, int, int, int, int, int]] = []
    decisions = 0

    def decide(t: int, sig: float, xsig: float) -> None:
        nonlocal decisions
        decisions += 1
        view = ControllerView(
            t,
            sig,
            tuple(observed),
            tuple(known),
            tuple(p is not None for p in pending),
            data,
            valid,
            session,
            exit_signal=xsig,
            acked_fills=tuple(inbox),
        )
        inbox.clear()
        action = policy.act(view)
        if action.new_episode:
            for v in range(nv):
                tried[v] = False
        flatten = t >= session.flatten_start_ns
        for v in range(nv):
            if pending[v] is not None or not view.has_book(v):
                continue
            tgt = max(-cap if data.venues[v].instrument.shortable else 0, min(cap, int(action.targets_units[v])))
            tgt = (abs(tgt) // lots[v]) * lots[v] * (1 if tgt >= 0 else -1)
            k = known[v]
            if k != 0 and tgt != 0 and (k > 0) != (tgt > 0):
                tgt = 0
            increasing = abs(tgt) > abs(k)
            if increasing and tried[v]:
                continue
            delta = tgt - k
            if delta == 0 or t + lat > session.exec_cutoff_ns:
                continue
            side = 1 if delta > 0 else -1
            qty = abs(delta)
            slip = action.entry_slip_ticks if increasing else action.exit_slip_ticks
            limit = view.best_ask(v) + slip if side > 0 else view.best_bid(v) - slip
            arrival = t + lat
            book = data.venues[v].book
            clock = env[v] if scenario.timing == TimingMode.EXCHANGE_CLOCK else book.recv_ns
            row = int(np.searchsorted(clock, arrival, side="right")) - 1
            oid = len(order_rows)
            filled = 0
            reason = REASON_FLATTEN if flatten else (REASON_ENTRY if increasing else REASON_EXIT)
            reports: list[FillReport] = []
            if row >= 0 and valid[v][row]:
                if side > 0:
                    levels = _sweep_reference(
                        side,
                        qty,
                        limit,
                        book.ask_px[row].tolist(),
                        book.ask_sz[row].tolist(),
                        int(book.n_ask[row]),
                        lots[v],
                        scenario.depth_fraction,
                        consumed[v][1],
                        arrival,
                        scenario.consumption_ttl_ns,
                    )
                else:
                    levels = _sweep_reference(
                        side,
                        qty,
                        limit,
                        book.bid_px[row].tolist(),
                        book.bid_sz[row].tolist(),
                        int(book.n_bid[row]),
                        lots[v],
                        scenario.depth_fraction,
                        consumed[v][-1],
                        arrival,
                        scenario.consumption_ttl_ns,
                    )
                for price, take in levels:
                    spec = data.venues[v].instrument
                    exposure = take * spec.cash_factor
                    cash = (
                        (side - fee_f[v]) * exposure / (price * ticks[v])
                        if spec.inverse
                        else (-side - fee_f[v]) * exposure * (price * ticks[v])
                    )
                    cash -= take * (float(scenario.fee_per_unit_for(spec)) * float(spec.quantity_step))
                    fill_rows.append((oid, v, side, price, take, t, arrival, arrival + ack_lat, cash))
                    reports.append(FillReport(v, side, price, take, t, arrival + ack_lat, reason))
                    filled += take
            order_rows.append(
                (v, side, qty, limit, t, arrival, arrival + ack_lat, filled, reason, row if row >= 0 else -1)
            )
            true_pos[v] += side * filled
            pending[v] = (arrival + ack_lat, side * filled)
            pending_reports[v] = reports
            if increasing:
                tried[v] = True

    def next_ack() -> int | None:
        times = [p[0] for p in pending if p is not None]
        return min(times) if times else None

    def apply_acks(t: int) -> None:
        for v in range(nv):
            p = pending[v]
            if p is not None and p[0] <= t:
                known[v] += p[1]
                pending[v] = None
                inbox.extend(pending_reports[v])
                pending_reports[v] = []

    recv, venue, kind, rows = (ev.recv_ns.tolist(), ev.venue.tolist(), ev.kind.tolist(), ev.row.tolist())
    sig_list = np.asarray(signal, dtype=np.float64).tolist()
    xsig_list = exit_signal.tolist()
    n = len(recv)
    e = 0
    last_sig = float("nan")
    last_xsig = float("nan")
    flat_timer: int | None = session.flatten_start_ns

    def next_point() -> int | None:
        candidates = [x for x in (next_ack(), flat_timer) if x is not None]
        return min(candidates) if candidates else None

    while e < n:
        t = recv[e]
        while (ta := next_point()) is not None and ta < t:
            if ta == flat_timer:
                flat_timer = None
            apply_acks(ta)
            decide(ta, last_sig, last_xsig)
        while e < n and recv[e] == t:
            if kind[e] == BOOK_EVENT:
                observed[venue[e]] = rows[e]
            last_sig = sig_list[e]
            last_xsig = xsig_list[e]
            e += 1
        apply_acks(t)
        if flat_timer == t:
            flat_timer = None
        decide(t, last_sig, last_xsig)
    while (ta := next_point()) is not None and ta <= session.end_ns:
        if ta == flat_timer:
            flat_timer = None
        apply_acks(ta)
        decide(ta, last_sig, last_xsig)

    fills = Fills(
        order_id=np.array([r[0] for r in fill_rows], np.int64),
        venue=np.array([r[1] for r in fill_rows], np.int8),
        side=np.array([r[2] for r in fill_rows], np.int8),
        px_ticks=np.array([r[3] for r in fill_rows], np.int32),
        qty_units=np.array([r[4] for r in fill_rows], np.int64),
        submit_ns=np.array([r[5] for r in fill_rows], np.int64),
        arrival_ns=np.array([r[6] for r in fill_rows], np.int64),
        ack_ns=np.array([r[7] for r in fill_rows], np.int64),
        cash=np.array([r[8] for r in fill_rows], np.float64),
    )
    cols = list(zip(*order_rows, strict=True)) if order_rows else [()] * 10
    dtypes = (np.int8, np.int8, np.int64, np.int64, np.int64, np.int64, np.int64, np.int64, np.int8, np.int64)
    orders = Orders(
        np.arange(len(order_rows), dtype=np.int64), *(np.array(c, d) for c, d in zip(cols, dtypes, strict=True))
    )
    settlement = settle(
        fills,
        data.instruments,
        fees,
        cap_units=cap,
        fees_per_unit=tuple(scenario.fee_per_unit_for(i) for i in data.instruments),
    )
    return ReplayResult(fills=fills, orders=orders, settlement=settlement, decisions=decisions)
