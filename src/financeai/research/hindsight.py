"""Constructive hindsight benchmarks. Every optimizer here reads future prices.

``quote_oracle`` attains the top-quote relaxation in exact arithmetic. It grants
unlimited displayed size and instantaneous reversals. ``depth_oracle`` solves a
restricted, executable inventory graph: fixed receipt times, a USD inventory grid,
full IOC fills, and spacing sufficient for acknowledgement and liquidity expiry.
Its optimum is a LOWER bound on the full engine's hindsight optimum.

Neither API is a causal signal recipe or evidence of future trading performance.
See ``docs/hindsight.md`` for the state spaces, proofs and numerical qualifications.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError, InvariantViolationError
from financeai.core.jit import njit
from financeai.domains.market.engine import ReplayResult
from financeai.domains.market.reference import ControllerView, PolicyAction, run_reference
from financeai.domains.market.scenario import RiskLimits, Scenario, SessionPlan, TimingMode
from financeai.domains.market.stream import BOOK_EVENT, EventStream, MarketData, book_validity, exchange_envelope
from financeai.domains.market.tape import MarketTapes

_DEFAULT_LIMITS = RiskLimits()


@dataclass(frozen=True)
class QuoteOracle:
    """Nonzero transactions in tape order; no claim of executable displayed size."""

    rows: NDArray[np.int64]
    targets_units: NDArray[np.int64]
    delta_units: NDArray[np.int64]
    score_pnl: float
    numerical_dual_pnl: float
    valid_rows: int


@njit()
def _quote_path(
    bid: NDArray[np.int32], ask: NDArray[np.int32], valid: NDArray[np.bool_], fee: float, tick: float, cap: int
) -> tuple[NDArray[np.int64], float]:
    n = len(bid)
    parents = np.empty((n, 3), np.int8)
    values = np.array([-np.inf, 0.0, -np.inf])
    lo, hi, tv = 0.0, 0.0, 0.0
    started = False
    for i in range(n):
        parents[i] = np.arange(3, dtype=np.int8)
        if not valid[i]:
            continue
        buy = (1 - fee) / (ask[i] * tick)
        sell = (1 + fee) / (bid[i] * tick)
        if not started:
            lo, hi, started = buy, sell, True
        elif sell < lo:
            tv += lo - sell
            lo, hi = sell, sell
        elif buy > hi:
            tv += buy - hi
            lo, hi = buy, buy
        else:
            lo, hi = max(lo, buy), min(hi, sell)
        nxt = values.copy()  # strict improvement keeps a no-trade tie
        for target in range(3):
            for prev in range(3):
                delta = target - prev
                value = values[prev] + delta * (buy if delta > 0 else sell)
                if value > nxt[target]:
                    nxt[target] = value
                    parents[i, target] = prev
        values = nxt
    targets = np.empty(n, np.int64)
    state = 1  # terminal flat
    for i in range(n - 1, -1, -1):
        targets[i] = (state - 1) * cap
        state = parents[i, state]
    return targets, cap * tv


def _require_inverse(tapes: MarketTapes, scenario: Scenario | None = None) -> None:
    spec = tapes.instrument
    if not spec.inverse or spec.cash_factor != 1.0 or not spec.shortable:
        raise InvalidConfigError("this hindsight solver requires shortable inverse contracts with unit cash exposure")
    if scenario is not None and scenario.fee_per_unit_for(spec) != 0:
        raise InvalidConfigError("this hindsight solver supports proportional fees only")


def quote_oracle(tapes: MarketTapes, fee: Decimal, *, cap_units: int = 1_000_000) -> QuoteOracle:
    """Backtrack the optimal {-C, 0, C} inventory path in O(number of rows).

    The recurrence is exact for arbitrary lot-multiple inventories in [-C, C]
    in this relaxation; computation uses float64. ``numerical_dual_pnl`` is an
    independently computed TV value, not an outward-rounded certificate.
    """
    _require_inverse(tapes)
    if (
        not isinstance(fee, Decimal)
        or not fee.is_finite()
        or not 0 <= fee < 1
        or isinstance(cap_units, bool)
        or not isinstance(cap_units, int)
        or cap_units <= 0
        or cap_units % tapes.instrument.lot_size
    ):
        raise InvalidConfigError("need fee in [0,1) and a positive lot-multiple cap")
    b = tapes.book
    valid = book_validity(tapes)
    if np.any(valid & ((b.bid_px[:, 0] <= 0) | (b.ask_px[:, 0] <= 0))):
        raise InvalidConfigError("valid quotes must have positive prices")
    targets, dual = _quote_path(
        b.bid_px[:, 0], b.ask_px[:, 0], valid, float(fee), float(tapes.instrument.tick_size), cap_units
    )
    delta = np.diff(targets, prepend=np.int64(0))
    rows = np.flatnonzero(delta).astype(np.int64)
    delta = delta[rows]
    prices = np.where(delta > 0, b.ask_px[rows, 0], b.bid_px[rows, 0])
    cash = (np.sign(delta) - float(fee)) * np.abs(delta) / (prices * float(tapes.instrument.tick_size))
    score = math.fsum(cash.tolist())
    if not math.isclose(score, dual, rel_tol=1e-9, abs_tol=1e-8):
        raise InvariantViolationError("quote primal and TV dual disagree")
    return QuoteOracle(rows, targets[rows], delta, score, dual, int(valid.sum()))


@dataclass(frozen=True)
class DepthOracle:
    """A flat-to-flat schedule, optimal only for its specified restricted graph."""

    instrument_key: str
    submit_ns: NDArray[np.int64]
    observed_rows: NDArray[np.int64]
    targets_units: NDArray[np.int64]
    predicted_pnl: float
    opportunities: int
    step_units: int
    spacing_ns: int
    slip_ticks: int


@njit()
def _sample_times(clock: NDArray[np.int64], start: int, stop: int, spacing: int) -> NDArray[np.int64]:
    rows = np.empty(len(clock), np.int64)
    count, next_time = 0, start
    for i in range(len(clock)):
        t = clock[i]
        if t >= stop:
            break
        if t < next_time or (i + 1 < len(clock) and clock[i + 1] == t):
            continue  # the final row of a receipt cohort is observable
        rows[count] = i
        count += 1
        next_time = t + spacing
    return rows[:count]


@njit()
def _curve(
    px: NDArray[np.int32],
    sz: NDArray[np.int32],
    levels: int,
    side: int,
    limit: int,
    step: int,
    half: int,
    lot: int,
    fraction: float,
    fee: float,
    tick: float,
) -> NDArray[np.float64]:
    """IOC score for each completely filled grid quantity (unfillable -> -inf)."""
    out = np.full(half + 1, -np.inf)
    out[0] = 0.0
    level, used, cost, filled = 0, 0, 0.0, 0
    for k in range(1, half + 1):
        target = k * step
        while filled < target and level < levels:
            p = px[level]
            if (side > 0 and p > limit) or (side < 0 and p < limit):
                break
            shown = int(sz[level])
            if fraction < 1:
                shown = (int(fraction * shown) // lot) * lot
            available = (shown // lot) * lot - used
            take = min(target - filled, max(0, available))
            cost += (side - fee) * take / (p * tick)
            used += take
            filled += take
            if used >= (shown // lot) * lot:
                level += 1
                used = 0
        if filled != target:
            break
        out[k] = cost
    return out


@njit()
def _depth_path(
    times: NDArray[np.int64],
    observed: NDArray[np.int64],
    arrivals: NDArray[np.int64],
    valid: NDArray[np.bool_],
    bid: NDArray[np.int32],
    ask: NDArray[np.int32],
    bs: NDArray[np.int32],
    az: NDArray[np.int32],
    nb: NDArray[np.int8],
    na: NDArray[np.int8],
    step: int,
    half: int,
    lot: int,
    fraction: float,
    fee: float,
    tick: float,
    slip: int,
    entry_cutoff: int,
) -> tuple[NDArray[np.int64], float]:
    n, states = len(times), 2 * half + 1
    parents = np.empty((n, states), np.int16)
    values = np.full(states, -np.inf)
    values[half] = 0.0
    for i in range(n):
        nxt = values.copy()
        parents[i] = np.arange(states, dtype=np.int16)
        o, r = observed[i], arrivals[i]
        if r >= 0 and valid[o] and valid[r]:
            buys = _curve(ask[r], az[r], na[r], 1, int(ask[o, 0]) + slip, step, half, lot, fraction, fee, tick)
            sells = _curve(bid[r], bs[r], nb[r], -1, int(bid[o, 0]) - slip, step, half, lot, fraction, fee, tick)
            for target in range(states):
                q = target - half
                for prev in range(states):
                    p = prev - half
                    if p * q < 0 or (abs(q) > abs(p) and times[i] >= entry_cutoff):
                        continue  # replay closes through zero; no late increases
                    d = target - prev
                    cash = buys[d] if d >= 0 else sells[-d]
                    value = values[prev] + cash
                    if value > nxt[target]:
                        nxt[target] = value
                        parents[i, target] = prev
        values = nxt
    targets = np.empty(n, np.int64)
    state = half
    for i in range(n - 1, -1, -1):
        targets[i] = (state - half) * step
        state = parents[i, state]
    return targets, values[half]


def _depth_inputs(
    tapes: MarketTapes,
    scenario: Scenario,
    session: SessionPlan,
    limits: RiskLimits,
    step_units: int,
    spacings: tuple[int, ...],
    slip_ticks: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
    """Shared validation, receipt-cohort selection and actual arrival rows."""
    _require_inverse(tapes, scenario)
    cap, lot = limits.max_abs_position_units, tapes.instrument.lot_size
    if isinstance(cap, bool) or not isinstance(cap, int) or not 0 < cap <= 2**62:
        raise InvalidConfigError("cap must be an integer in [1, 2**62]")
    if (
        isinstance(step_units, bool)
        or not isinstance(step_units, int)
        or step_units <= 0
        or step_units % lot
        or cap % step_units
        or 2 * cap // step_units + 1 > 32767
    ):
        raise InvalidConfigError("step must divide cap, respect lots, and yield at most 32767 states")
    if not spacings or any(isinstance(s, bool) or not isinstance(s, int) or not 1 <= s <= 2**62 for s in spacings):
        raise InvalidConfigError("candidate spacings must be integer nanoseconds in [1, 2**62]")
    if isinstance(slip_ticks, bool) or not isinstance(slip_ticks, int) or not 0 <= slip_ticks <= 2**31:
        raise InvalidConfigError("slip_ticks must be an integer in [0, 2**31]")
    b = tapes.book
    if len(b.recv_ns):
        low, high = int(b.recv_ns[0]), int(b.recv_ns[-1])
        if low < 0 or high > 2**62:
            raise InvalidConfigError("depth oracle receipt clocks must be within [0, 2**62]")
        if high + max(*spacings, scenario.order_latency_ns + scenario.ack_ns) > np.iinfo(np.int64).max:
            raise InvalidConfigError("depth oracle derived clocks would overflow int64")
    if np.any(np.diff(b.recv_ns) < 0):
        raise InvalidConfigError("receipt clock must be monotone")
    stop = min(session.flatten_start_ns, session.exec_cutoff_ns - scenario.order_latency_ns + 1)
    observed = np.unique(np.concatenate([_sample_times(b.recv_ns, session.start_ns, stop, s) for s in spacings]))
    times = b.recv_ns[observed]
    clock = exchange_envelope(tapes) if scenario.timing == TimingMode.EXCHANGE_CLOCK else b.recv_ns
    arrivals = np.searchsorted(clock, times + scenario.order_latency_ns, side="right") - 1
    valid = book_validity(tapes)
    # Ordered, positive, unique levels are required by the independent IOC curve.
    for rows in (observed, arrivals[arrivals >= 0]):
        rows = rows[valid[rows]]
        for px, sz, counts, side in ((b.ask_px, b.ask_sz, b.n_ask, 1), (b.bid_px, b.bid_sz, b.n_bid, -1)):
            active = np.arange(px.shape[1])[None, :] < counts[rows, None]
            prices = px[rows]
            if (
                np.any(counts[rows] > px.shape[1])
                or np.any(active & ((prices <= 0) | (sz[rows] < 0)))
                or np.any(active[:, 1:] & (side * np.diff(prices, axis=1) <= 0))
            ):
                raise InvalidConfigError("depth oracle needs positive, strictly ordered book levels")
    return observed, times, arrivals, valid


def depth_oracle(
    tapes: MarketTapes,
    scenario: Scenario,
    session: SessionPlan,
    limits: RiskLimits = _DEFAULT_LIMITS,
    *,
    step_units: int = 1_000,
    spacing_ns: int = 1_000_000_000,
    slip_ticks: int = 200,
) -> DepthOracle:
    """Optimize a full-depth hindsight schedule with O(N K^2) work / O(N K) memory.

    N = sampled receipt cohorts, K = 2*cap/step + 1. Only fully filled grid
    transitions are admitted. Trading ends flat before forced liquidation.
    Spacing clears both pending acknowledgements and our liquidity-consumption
    memory, making inventory alone a sufficient state. Fees and latency come from
    the supplied scenario. No modification of the replay engine is needed.
    """
    minimum = max(1, scenario.order_latency_ns + scenario.ack_ns, scenario.consumption_ttl_ns)
    if isinstance(spacing_ns, bool) or not isinstance(spacing_ns, int) or spacing_ns < minimum:
        raise InvalidConfigError("spacing must cover the order round trip and consumption TTL")
    observed, times, arrivals, valid = _depth_inputs(
        tapes, scenario, session, limits, step_units, (spacing_ns,), slip_ticks
    )
    b, cap, lot = tapes.book, limits.max_abs_position_units, tapes.instrument.lot_size
    targets, score = _depth_path(
        times,
        observed,
        arrivals,
        valid,
        b.bid_px,
        b.ask_px,
        b.bid_sz,
        b.ask_sz,
        b.n_bid,
        b.n_ask,
        step_units,
        cap // step_units,
        lot,
        scenario.depth_fraction,
        float(scenario.fee_for(tapes.instrument)),
        float(tapes.instrument.tick_size),
        slip_ticks,
        session.entry_cutoff_ns,
    )
    take = np.flatnonzero(np.diff(targets, prepend=np.int64(0)))
    return DepthOracle(
        tapes.instrument.key,
        times[take],
        observed[take],
        targets[take],
        float(score),
        len(times),
        step_units,
        spacing_ns,
        slip_ticks,
    )


class HindsightPolicy:
    """Replay a fixed future-informed schedule through the ordinary Policy protocol.

    Off-schedule decisions keep the acknowledged position. Every scheduled order
    starts an episode so grid scaling is permitted. Single use, like other policies.
    """

    def __init__(self, plans: tuple[DepthOracle, ...]) -> None:
        self.plans = plans
        self.indices = [0] * len(plans)
        self.slip = max((p.slip_ticks for p in plans), default=0)
        if any(p.slip_ticks != self.slip for p in plans):
            raise InvalidConfigError("all venue schedules need the same slip limit")

    def act(self, view: ControllerView) -> PolicyAction:
        targets = list(view.known_units)
        changed = False
        for v, plan in enumerate(self.plans):
            i = self.indices[v]
            if i < len(plan.submit_ns) and view.t > plan.submit_ns[i]:
                raise InvariantViolationError("replay missed a scheduled receipt cohort")
            if i < len(plan.submit_ns) and view.t == plan.submit_ns[i]:
                if view.pending[v]:
                    raise InvariantViolationError("scheduled order still has a pending predecessor")
                targets[v] = int(plan.targets_units[i])
                if view.t >= view.session.entry_cutoff_ns and abs(targets[v]) > abs(view.known_units[v]):
                    raise InvariantViolationError("schedule increases exposure after the entry cutoff")
                self.indices[v] += 1
                changed = True
        if view.t < view.session.start_ns:
            targets = [0] * len(targets)
        if view.t >= view.session.flatten_start_ns:
            targets = [0] * len(targets)
        return PolicyAction(tuple(targets), changed, self.slip, self.slip)


def replay_depth_oracle(
    data: MarketData,
    plans: tuple[DepthOracle, ...],
    scenario: Scenario,
    session: SessionPlan,
    limits: RiskLimits = _DEFAULT_LIMITS,
    *,
    full_stream: bool = False,
) -> ReplayResult:
    """Verify a schedule with the existing independent Python reference engine.

    The compact stream retains actual book receipt cohorts at every order time.
    Other decisions cannot trade under this policy. Execution always queries the
    complete original tapes. ``full_stream=True`` verifies this reduction against
    every receipt/trade event; acknowledgements are processed in both modes.
    """
    if tuple(p.instrument_key for p in plans) != tuple(i.key for i in data.instruments):
        raise InvalidConfigError("plans must match market venue order")
    if full_stream:
        stream = EventStream.build(data)
    else:
        times = np.concatenate([p.submit_ns for p in plans])
        venues = np.concatenate([np.full(len(p.submit_ns), v, np.int8) for v, p in enumerate(plans)])
        rows = np.concatenate([p.observed_rows for p in plans])
        order = np.lexsort((rows, venues, times))
        stream = EventStream(times[order], venues[order], np.full(len(times), BOOK_EVENT, np.int8), rows[order])
    policy = HindsightPolicy(plans)
    result = run_reference(data, np.zeros(len(stream)), policy, scenario, session, limits, stream=stream)
    expected_orders = sum(len(p.submit_ns) for p in plans)
    for v, plan in enumerate(plans):
        actual = result.orders.venue == v
        signed = np.diff(plan.targets_units, prepend=np.int64(0))
        if not np.array_equal(result.orders.submit_ns[actual], plan.submit_ns) or not np.array_equal(
            result.orders.side[actual] * result.orders.filled_units[actual], signed
        ):
            raise InvariantViolationError("replay did not reproduce the planned position path")
    if (
        not result.settlement.valid
        or len(result.orders) != expected_orders
        or np.any(result.orders.filled_units != result.orders.qty_units)
        or not math.isclose(
            result.settlement.net_pnl or 0.0, sum(p.predicted_pnl for p in plans), rel_tol=1e-9, abs_tol=1e-8
        )
    ):
        raise InvariantViolationError("hindsight schedule failed replay feasibility or score agreement")
    return result
