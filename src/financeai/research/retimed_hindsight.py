"""Future-aware coordinate ascent for a fixed executable inventory path.

Buy and sell consumption memories are separate in the reference engine. Orders
therefore need a full TTL between successive orders of the SAME side, and only
an acknowledgement round trip between opposite sides. The returned schedule is
a lower-bound witness, never a global or causal optimum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError, InvariantViolationError
from financeai.core.jit import njit
from financeai.domains.market.scenario import RiskLimits, Scenario, SessionPlan
from financeai.domains.market.tape import MarketTapes
from financeai.research.hindsight import DepthOracle, _curve, _depth_inputs

_DEFAULT_LIMITS = RiskLimits()


@dataclass(frozen=True)
class RetimingResult:
    plan: DepthOracle
    passes: tuple[dict[str, int | float | str], ...]
    minimum_order_gap_ns: int | None
    minimum_same_side_gap_ns: int | None
    sub_ttl_opposite_side_pairs: int


@njit()
def _cash(
    i: int,
    delta: int,
    observed: NDArray[np.int64],
    arrivals: NDArray[np.int64],
    valid: NDArray[np.bool_],
    bid: NDArray[np.int32],
    ask: NDArray[np.int32],
    bs: NDArray[np.int32],
    az: NDArray[np.int32],
    nb: NDArray[np.int8],
    na: NDArray[np.int8],
    lot: int,
    fraction: float,
    fee: float,
    tick: float,
    slip: int,
) -> float:
    o, r = observed[i], arrivals[i]
    if r < 0 or not valid[o] or not valid[r]:
        return -np.inf
    if delta > 0:
        return float(
            _curve(
                ask[r],
                az[r],
                na[r],
                1,
                int(ask[o, 0]) + slip,
                delta,
                1,
                lot,
                fraction,
                fee,
                tick,
            )[1]
        )
    return float(
        _curve(
            bid[r],
            bs[r],
            nb[r],
            -1,
            int(bid[o, 0]) - slip,
            -delta,
            1,
            lot,
            fraction,
            fee,
            tick,
        )[1]
    )


@njit()
def _sweep(
    indices: NDArray[np.int64],
    deltas: NDArray[np.int64],
    increases: NDArray[np.bool_],
    previous: NDArray[np.int64],
    following: NDArray[np.int64],
    costs: NDArray[np.float64],
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
    lot: int,
    fraction: float,
    fee: float,
    tick: float,
    slip: int,
    rtt: int,
    ttl: int,
    start: int,
    stop: int,
    entry_cutoff: int,
    reverse: bool,
) -> tuple[int, int]:
    changed, scanned = 0, 0
    n = len(indices)
    for k in range(n):
        j = n - 1 - k if reverse else k
        lower, upper = start, stop - 1
        if j:
            lower = max(lower, times[indices[j - 1]] + rtt)
        if j + 1 < n:
            upper = min(upper, times[indices[j + 1]] - rtt)
        if previous[j] >= 0:
            lower = max(lower, times[indices[previous[j]]] + ttl)
        if following[j] >= 0:
            upper = min(upper, times[indices[following[j]]] - ttl)
        if increases[j]:
            upper = min(upper, entry_cutoff - 1)
        left, right = np.searchsorted(times, lower), np.searchsorted(times, upper, side="right")
        best, value = indices[j], costs[j]
        for i in range(left, right):
            scanned += 1
            candidate = _cash(
                i, deltas[j], observed, arrivals, valid, bid, ask, bs, az, nb, na, lot, fraction, fee, tick, slip
            )
            if candidate > value:  # stable no-move ties; arithmetic is float64
                best, value = i, candidate
        if best != indices[j]:
            indices[j], costs[j] = best, value
            changed += 1
    return changed, scanned


def retime_depth_oracle(
    tapes: MarketTapes,
    initial: DepthOracle,
    scenario: Scenario,
    session: SessionPlan,
    limits: RiskLimits = _DEFAULT_LIMITS,
    *,
    max_passes: int = 6,
) -> RetimingResult:
    """Improve order times at every book receipt, preserving every target quantity.

    Alternating forward/backward coordinate sweeps maximize the complete IOC cash
    of one order while other order times stay fixed. Same-side TTL separation
    clears all consumption at that order; RTT separation clears pending orders.
    Full reference replay remains the independent feasibility check. An unchanged
    sweep means coordinate-stationarity in float64, not exact/global optimality.
    """
    scenario = scenario.for_instrument(tapes.instrument)
    if isinstance(max_passes, bool) or not isinstance(max_passes, int) or max_passes <= 0:
        raise InvalidConfigError("max_passes must be a positive integer")
    if initial.instrument_key != tapes.instrument.key:
        raise InvalidConfigError("initial plan must match the instrument")
    for array in (initial.submit_ns, initial.observed_rows, initial.targets_units):
        if not isinstance(array, np.ndarray) or array.dtype != np.int64 or array.ndim != 1:
            raise InvalidConfigError("schedule columns must be one-dimensional int64 arrays")
    n = len(initial.submit_ns)
    if len(initial.observed_rows) != n or len(initial.targets_units) != n:
        raise InvalidConfigError("schedule columns must have equal lengths")
    cap, lot = limits.cap_for(tapes.instrument), tapes.instrument.lot_size
    targets = initial.targets_units
    if np.any(targets > cap) or np.any(targets < -cap) or np.any(targets % lot):
        raise InvalidConfigError("targets must respect the cap and lot")
    prev_targets = np.concatenate((np.zeros(1, np.int64), targets[:-1])) if n else targets
    delta = targets - prev_targets
    if (n and targets[-1] != 0) or np.any(delta == 0):
        raise InvalidConfigError("schedule must contain nonzero orders and finish flat")
    if np.any((targets > 0) & (prev_targets < 0)) or np.any((targets < 0) & (prev_targets > 0)):
        raise InvalidConfigError("schedule must close through zero")
    increases = np.abs(targets) > np.abs(prev_targets)
    rtt, ttl = scenario.order_latency_ns + scenario.ack_ns, scenario.consumption_ttl_ns
    if np.any(np.diff(initial.submit_ns) < rtt):
        raise InvalidConfigError("initial orders must be separated by the acknowledgement round trip")
    for side in (-1, 1):
        if np.any(np.diff(initial.submit_ns[np.sign(delta) == side]) < ttl):
            raise InvalidConfigError("initial orders of each side must be separated by the consumption TTL")
    observed, times, arrivals, valid = _depth_inputs(
        tapes, scenario, session, limits, initial.step_units, (1,), initial.slip_ticks
    )
    indices = np.searchsorted(times, initial.submit_ns)
    if (
        np.any(indices >= len(times))
        or not np.array_equal(times[indices], initial.submit_ns)
        or not np.array_equal(observed[indices], initial.observed_rows)
    ):
        raise InvalidConfigError("initial orders must use final receipt cohorts inside the session")
    if np.any(increases & (initial.submit_ns >= session.entry_cutoff_ns)):
        raise InvalidConfigError("initial orders must not increase exposure after the entry cutoff")
    previous, following = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    for side in (-1, 1):
        same = np.flatnonzero(np.sign(delta) == side)
        previous[same[1:]], following[same[:-1]] = same[:-1], same[1:]
    b = tapes.book
    common = (
        observed,
        arrivals,
        valid,
        b.bid_px,
        b.ask_px,
        b.bid_sz,
        b.ask_sz,
        b.n_bid,
        b.n_ask,
        lot,
        scenario.depth_fraction,
        float(scenario.fee_for(tapes.instrument)),
        float(tapes.instrument.tick_size),
        initial.slip_ticks,
    )
    costs = np.array([_cash(i, int(d), *common) for i, d in zip(indices, delta, strict=True)])
    if not np.all(np.isfinite(costs)):
        raise InvalidConfigError("every initial order must be fully fillable")
    if not math.isclose(math.fsum(costs), initial.predicted_pnl, rel_tol=1e-9, abs_tol=1e-8):
        raise InvariantViolationError("initial schedule cash disagrees with independent IOC curves")
    trace: list[dict[str, int | float | str]] = []
    stop = min(session.flatten_start_ns, session.exec_cutoff_ns - scenario.order_latency_ns + 1)
    for sweep in range(max_passes):
        before = math.fsum(costs)
        changed, scanned = _sweep(
            indices,
            delta,
            increases,
            previous,
            following,
            costs,
            times,
            *common,
            rtt,
            ttl,
            session.start_ns,
            stop,
            session.entry_cutoff_ns,
            sweep % 2 == 1,
        )
        after = math.fsum(costs)
        trace.append(
            {
                "pass": sweep + 1,
                "direction": "backward" if sweep % 2 else "forward",
                "changed_orders": changed,
                "candidate_evaluations": scanned,
                "before_pnl": before,
                "after_pnl": after,
                "gain_pnl": after - before,
            }
        )
        if not changed:
            break
    chosen = times[indices]
    gaps = np.diff(chosen)
    same_gaps = [np.diff(chosen[np.sign(delta) == s]) for s in (-1, 1)]
    all_same = np.concatenate(same_gaps)
    if np.any(gaps < rtt) or np.any(all_same < ttl):
        raise InvariantViolationError("retiming violated an execution-memory fence")
    plan = DepthOracle(
        initial.instrument_key,
        chosen,
        observed[indices],
        targets.copy(),
        math.fsum(costs),
        len(times),
        initial.step_units,
        rtt,
        initial.slip_ticks,
    )
    return RetimingResult(
        plan,
        tuple(trace),
        int(gaps.min()) if len(gaps) else None,
        int(all_same.min()) if len(all_same) else None,
        int(np.sum((gaps < ttl) & (np.sign(delta[:-1]) != np.sign(delta[1:])))),
    )
