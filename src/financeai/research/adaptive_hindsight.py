"""Future-informed optimization with flexible order timing and a liquidity cooldown.

The previous fixed opportunity grid is included explicitly. Cooldown constrains
selected nonzero orders, rather than every candidate opportunity. This enlarges
the executable hindsight graph without reusing consumed liquidity. It supplies
a lower bound on the full replay optimum, never a causal trading signal.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError
from financeai.core.jit import njit
from financeai.domains.market.scenario import RiskLimits, Scenario, SessionPlan
from financeai.domains.market.tape import MarketTapes
from financeai.research.hindsight import DepthOracle, _curve, _depth_inputs

_DEFAULT_LIMITS = RiskLimits()


@njit()
def _transition_quadratic(
    values: NDArray[np.float64],
    buys: NDArray[np.float64],
    sells: NDArray[np.float64],
    late: bool,
) -> tuple[NDArray[np.float64], NDArray[np.int16]]:
    """Deliberately simple reference for the accelerated transition maximization."""
    states, half = len(values), len(values) // 2
    out, parent = np.full(states, -np.inf), np.full(states, -1, np.int16)
    for q in range(states):
        for p in range(states):
            if (p - half) * (q - half) < 0 or (late and abs(q - half) > abs(p - half)):
                continue
            d = q - p
            cash = buys[d] if d >= 0 else sells[-d]
            candidate = values[p] + cash
            if candidate > out[q]:
                out[q], parent[q] = candidate, p
    return out, parent


@njit()
def _transition_monge(
    values: NDArray[np.float64],
    buys: NDArray[np.float64],
    sells: NDArray[np.float64],
    late: bool,
) -> tuple[NDArray[np.float64], NDArray[np.int16]]:
    """Divide-and-conquer maxima of a concave IOC convolution, split by sign.

    Adding arbitrary predecessor scores as column constants preserves the Monge
    inequality. Reachability need not be concave. An unreachable midpoint never
    narrows the search bounds for its children. Consistent leftmost ties are used.
    """
    states, half = len(values), len(values) // 2
    out, parent = np.full(states, -np.inf), np.full(states, -1, np.int16)
    buy_max, sell_max = 0, 0
    for k in range(1, len(buys)):
        if np.isfinite(buys[k]):
            buy_max = k
        if np.isfinite(sells[k]):
            sell_max = k
    # q=0 can be reached from either sign; handle it outside the sign blocks.
    for p in range(max(0, half - buy_max), min(states - 1, half + sell_max) + 1):
        d = half - p
        value = values[p] + (buys[d] if d >= 0 else sells[-d])
        if value > out[half]:
            out[half], parent[half] = value, p
    stack = np.empty((2 * states + 2, 4), np.int64)
    for positive in (False, True):
        low, high = (half + 1, states - 1) if positive else (0, half - 1)
        plow, phigh = (half, states - 1) if positive else (0, half)
        stack[0] = (low, high, plow, phigh)
        count = 1
        while count:
            count -= 1
            left, right, pmin, pmax = stack[count]
            if left > right:
                continue
            q = (left + right) // 2
            start, stop = max(pmin, q - buy_max), min(pmax, q + sell_max)
            if late:
                if positive:
                    start = max(start, q)
                else:
                    stop = min(stop, q)
            best, arg = -np.inf, -1
            for p in range(start, stop + 1):
                d = q - p
                value = values[p] + (buys[d] if d >= 0 else sells[-d])
                if value > best:
                    best, arg = value, p
            out[q], parent[q] = best, arg
            # With no reachable optimum the argmax is undefined: retain both bounds.
            upper_left = arg if arg >= 0 else pmax
            lower_right = arg if arg >= 0 else pmin
            stack[count] = (left, q - 1, pmin, upper_left)
            stack[count + 1] = (q + 1, right, lower_right, pmax)
            count += 2
    return out, parent


@njit()
def _advance_cooldown(
    times: NDArray[np.int64],
    observed: NDArray[np.int64],
    arrivals: NDArray[np.int64],
    predecessor: NDArray[np.int64],
    ring_size: int,
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
    accelerated: bool,
    start: int,
    stop: int,
    parents: NDArray[np.int16],
    ring: NDArray[np.float64],
) -> None:
    states = 2 * half + 1
    for i in range(start, stop):
        values = ring[i % ring_size].copy()  # wait: no new cooldown
        o, r = observed[i], arrivals[i]
        if r >= 0 and valid[o] and valid[r]:
            buys = _curve(ask[r], az[r], na[r], 1, int(ask[o, 0]) + slip, step, half, lot, fraction, fee, tick)
            sells = _curve(bid[r], bs[r], nb[r], -1, int(bid[o, 0]) - slip, step, half, lot, fraction, fee, tick)
            eligible = ring[(predecessor[i] + 1) % ring_size]
            if accelerated:
                scores, choices = _transition_monge(eligible, buys, sells, times[i] >= entry_cutoff)
            else:
                scores, choices = _transition_quadratic(eligible, buys, sells, times[i] >= entry_cutoff)
            for q in range(states):
                if choices[q] != q and scores[q] > values[q]:
                    values[q] = scores[q]
                    if len(parents):
                        parents[i - start, q] = choices[q]
        ring[(i + 1) % ring_size] = values


@njit()
def _cooldown_path(
    times: NDArray[np.int64],
    observed: NDArray[np.int64],
    arrivals: NDArray[np.int64],
    predecessor: NDArray[np.int64],
    ring_size: int,
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
    accelerated: bool,
) -> tuple[NDArray[np.int64], NDArray[np.int64], float]:
    n, states = len(times), 2 * half + 1
    parents = np.full((n, states), -1, np.int16)
    ring = np.full((ring_size, states), -np.inf)
    ring[0, half] = 0.0
    _advance_cooldown(
        times,
        observed,
        arrivals,
        predecessor,
        ring_size,
        valid,
        bid,
        ask,
        bs,
        az,
        nb,
        na,
        step,
        half,
        lot,
        fraction,
        fee,
        tick,
        slip,
        entry_cutoff,
        accelerated,
        0,
        n,
        parents,
        ring,
    )
    score = ring[n % ring_size, half]
    chosen, targets = np.empty(n, np.int64), np.empty(n, np.int64)
    i, q, count = n - 1, half, 0
    while i >= 0:
        p = parents[i, q]
        if p < 0:
            i -= 1
        else:
            chosen[count], targets[count] = i, (q - half) * step
            count += 1
            q, i = p, predecessor[i]
    return chosen[:count][::-1].copy(), targets[:count][::-1].copy(), score


def _checkpoint_path(
    times: NDArray[np.int64],
    observed: NDArray[np.int64],
    arrivals: NDArray[np.int64],
    predecessor: NDArray[np.int64],
    ring_size: int,
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
    accelerated: bool,
    checkpoint_rows: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], float]:
    """Save frontiers, then recompute each visited block with temporary parents.

    The exact same transition kernel and tie order are used in both passes.
    Files live in a private temporary directory and are removed on return/error.
    """
    n, states = len(times), 2 * half + 1
    ring = np.full((ring_size, states), -np.inf)
    ring[0, half] = 0.0
    kernel = (
        times,
        observed,
        arrivals,
        predecessor,
        ring_size,
        valid,
        bid,
        ask,
        bs,
        az,
        nb,
        na,
        step,
        half,
        lot,
        fraction,
        fee,
        tick,
        slip,
        entry_cutoff,
        accelerated,
    )
    selected, targets = [], []
    with tempfile.TemporaryDirectory(prefix="financeai-frontier-") as directory:
        root = Path(directory)
        for start in range(0, n, checkpoint_rows):
            np.save(root / f"{start}.npy", ring, allow_pickle=False)
            _advance_cooldown(*kernel, start, min(n, start + checkpoint_rows), np.empty((0, states), np.int16), ring)
        score = float(ring[n % ring_size, half])
        i, q = n - 1, half
        while i >= 0:
            start = (i // checkpoint_rows) * checkpoint_rows
            stop = min(n, start + checkpoint_rows)
            ring = np.load(root / f"{start}.npy", allow_pickle=False)
            parents = np.full((stop - start, states), -1, np.int16)
            _advance_cooldown(*kernel, start, stop, parents, ring)
            while i >= start:
                p = int(parents[i - start, q])
                if p < 0:
                    i -= 1
                else:
                    selected.append(i)
                    targets.append((q - half) * step)
                    q, i = p, int(predecessor[i])
    return np.array(selected[::-1], dtype=np.int64), np.array(targets[::-1], dtype=np.int64), score


def adaptive_depth_oracle(
    tapes: MarketTapes,
    scenario: Scenario,
    session: SessionPlan,
    limits: RiskLimits = _DEFAULT_LIMITS,
    *,
    step_units: int = 1_000,
    candidate_spacing_ns: int = 100_000_000,
    include_spacing_ns: tuple[int, ...] = (1_000_000_000,),
    cooldown_ns: int | None = None,
    slip_ticks: int = 200,
    accelerated: bool = True,
    checkpoint_rows: int | None = None,
) -> DepthOracle:
    """Choose profitable times dynamically while selected orders respect cooldown.

    ``include_spacing_ns`` explicitly unions earlier candidate grids. Including
    the previous fixed-grid opportunities and the same inventory step, with a
    cooldown no larger than the previous spacing, guarantees nondecrease of the
    exact-arithmetic optimum. Independently resampling at a
    shorter interval alone does not give that containment guarantee.

    F_i(q)=max(F_(i-1)(q), max_p F_j(i)(p)+IOC_i(q-p)), where j(i) is the last
    candidate at or before time_i-cooldown. Initial and terminal inventories are
    flat. Scores use a ring sized from actual predecessor-index lags; all parent
    choices are retained for backtracking. Monge acceleration uses concave IOC
    rewards on each inventory-sign block; ``accelerated=False`` is the quadratic
    reference. Floating arithmetic is checked by independent replay, not an exact
    rational optimality certificate.

    ``checkpoint_rows`` bounds parent storage to that many candidate rows.
    Temporary frontier snapshots permit deterministic block recomputation;
    this exchanges roughly one extra forward pass for much lower peak memory.
    """
    scenario = scenario.for_instrument(tapes.instrument)
    minimum = max(1, scenario.order_latency_ns + scenario.ack_ns, scenario.consumption_ttl_ns)
    cooldown = minimum if cooldown_ns is None else cooldown_ns
    if isinstance(cooldown, bool) or not isinstance(cooldown, int) or not minimum <= cooldown <= 2**62:
        raise InvalidConfigError("cooldown must cover acknowledgements and liquidity expiry, and be <=2**62")
    if not isinstance(accelerated, bool) or not isinstance(include_spacing_ns, tuple):
        raise InvalidConfigError("accelerated must be bool and include_spacing_ns must be a tuple")
    if checkpoint_rows is not None and (
        isinstance(checkpoint_rows, bool) or not isinstance(checkpoint_rows, int) or checkpoint_rows < 1
    ):
        raise InvalidConfigError("checkpoint_rows must be a positive integer or None")
    observed, times, arrivals, valid = _depth_inputs(
        tapes, scenario, session, limits, step_units, (candidate_spacing_ns, *include_spacing_ns), slip_ticks
    )
    predecessor = np.searchsorted(times, times - cooldown, side="right") - 1
    # Row k stores F after k opportunities. Preserve F_(j+1) until it is read.
    ring_size = int(np.max(np.arange(len(times)) - predecessor, initial=0)) + 2
    b = tapes.book
    kernel = (
        times,
        observed,
        arrivals,
        predecessor,
        ring_size,
        valid,
        b.bid_px,
        b.ask_px,
        b.bid_sz,
        b.ask_sz,
        b.n_bid,
        b.n_ask,
        step_units,
        limits.cap_for(tapes.instrument) // step_units,
        tapes.instrument.lot_size,
        scenario.depth_fraction,
        float(scenario.fee_for(tapes.instrument)),
        float(tapes.instrument.tick_size),
        slip_ticks,
        session.entry_cutoff_ns,
        accelerated,
    )
    if checkpoint_rows is None:
        chosen, targets, score = _cooldown_path(*kernel)
    else:
        chosen, targets, score = _checkpoint_path(*kernel, checkpoint_rows)
    return DepthOracle(
        tapes.instrument.key,
        times[chosen],
        observed[chosen],
        targets,
        float(score),
        len(times),
        step_units,
        cooldown,
        slip_ticks,
    )
