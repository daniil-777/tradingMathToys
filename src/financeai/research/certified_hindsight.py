"""Integer upper certificates for the explicitly restricted cooldown graph.

Every per-lot cash reward is rounded upward by exact integer division. A
score-only Bellman pass therefore bounds every path in the graph, including
paths missed by a floating optimizer. This is not a full-engine upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError
from financeai.core.jit import njit
from financeai.domains.market.scenario import RiskLimits, Scenario, SessionPlan
from financeai.domains.market.tape import MarketTapes
from financeai.research.hindsight import _depth_inputs

_DEFAULT_LIMITS = RiskLimits()
_UNREACHABLE = -(1 << 63)
_SAFE = (1 << 62) - 1


@dataclass(frozen=True)
class GraphCertificate:
    """Exact fixed-point numerator bounding the declared graph's optimum."""

    instrument_key: str
    upper_numerator: int
    scale: int
    upper_pnl: str
    opportunities: int
    step_units: int
    cap_units: int
    lot_units: int
    cooldown_ns: int
    slip_ticks: int
    candidate_spacing_ns: int
    include_spacing_ns: tuple[int, ...]
    max_orders: int
    max_path_rounding_error_numerator: int
    max_path_rounding_error_pnl: str
    integer_absolute_path_bound: int
    accelerated: bool
    uses_future_data: bool = True
    scope: str = "restricted cooldown graph only; not the full-engine optimum"


def _fixed_decimal(value: int, scale: int) -> str:
    """Exact decimal serialization; the public API requires a power-of-ten scale."""
    places = len(str(scale)) - 1
    if not places:
        return str(value)
    sign = "-" if value < 0 else ""
    digits = str(abs(value)).rjust(places + 1, "0")
    return sign + digits[:-places] + "." + digits[-places:]


@njit()
def _ceil_div(n: int, d: int) -> int:
    # Unlike (n+d-1)//d, this never forms a potentially overflowing numerator.
    return n // d + (1 if n % d else 0)


@njit()
def _integer_curve(
    px: NDArray[np.int32],
    sz: NDArray[np.int32],
    levels: int,
    side: int,
    limit: int,
    step: int,
    half: int,
    lot: int,
    fraction: float,
    numerator: int,
    denominator: int,
) -> NDArray[np.int64]:
    """Sum exact upward-rounded per-lot rewards for fully filled quantities."""
    out = np.full(half + 1, _UNREACHABLE, np.int64)
    out[0] = 0
    level, used, cash, filled = 0, 0, 0, 0
    for k in range(1, half + 1):
        target = k * step
        while filled < target and level < levels:
            price = int(px[level])
            if (side > 0 and price > limit) or (side < 0 and price < limit):
                break
            shown = int(sz[level])
            if fraction < 1:
                shown = (int(fraction * shown) // lot) * lot
            available = (shown // lot) * lot - used
            take = min(target - filled, max(0, available))
            rate = _ceil_div(numerator, denominator * price)
            cash += (take // lot) * rate
            used += take
            filled += take
            if used >= (shown // lot) * lot:
                level += 1
                used = 0
        if filled != target:
            break
        out[k] = cash
    return out


@njit()
def _integer_quadratic(
    values: NDArray[np.int64], buys: NDArray[np.int64], sells: NDArray[np.int64], late: bool
) -> NDArray[np.int64]:
    states, half = len(values), len(values) // 2
    out = np.full(states, _UNREACHABLE, np.int64)
    for q in range(states):
        for p in range(states):
            if values[p] == _UNREACHABLE or (p - half) * (q - half) < 0:
                continue
            if late and abs(q - half) > abs(p - half):
                continue
            d = q - p
            cash = buys[d] if d >= 0 else sells[-d]
            if cash != _UNREACHABLE:
                out[q] = max(out[q], values[p] + cash)
    return out


@njit()
def _integer_monge(
    values: NDArray[np.int64], buys: NDArray[np.int64], sells: NDArray[np.int64], late: bool
) -> NDArray[np.int64]:
    """Four triangular searches: inventory sign crossed with order direction.

    Rounded rewards need only be concave separately on each side of zero.
    Rounding may make buy(1)+sell(1)>0; no search ever straddles that kink.
    """
    states, half = len(values), len(values) // 2
    out = np.full(states, _UNREACHABLE, np.int64)
    buy_max, sell_max = 0, 0
    for k in range(1, len(buys)):
        if buys[k] != _UNREACHABLE:
            buy_max = k
        if sells[k] != _UNREACHABLE:
            sell_max = k
    # Flat can be reached from both inventory signs.
    for p in range(max(0, half - buy_max), min(states - 1, half + sell_max) + 1):
        d = half - p
        reward = buys[d] if d >= 0 else sells[-d]
        if values[p] != _UNREACHABLE and reward != _UNREACHABLE:
            out[half] = max(out[half], values[p] + reward)
    stack = np.empty((2 * states + 2, 4), np.int64)
    for positive in (False, True):
        for buying in (False, True):
            low, high = (half + 1, states - 1) if positive else (0, half - 1)
            plow, phigh = (half, states - 1) if positive else (0, half)
            if late and positive == buying:
                continue  # this order direction only increases exposure
            stack[0] = (low, high, plow, phigh)
            count = 1
            while count:
                count -= 1
                left, right, pmin, pmax = stack[count]
                if left > right:
                    continue
                q = (left + right) // 2
                if buying:
                    start, stop = max(pmin, q - buy_max), min(pmax, q)
                else:
                    start, stop = max(pmin, q), min(pmax, q + sell_max)
                best, arg = _UNREACHABLE, -1
                for p in range(start, stop + 1):
                    if values[p] == _UNREACHABLE:
                        continue
                    reward = buys[q - p] if buying else sells[p - q]
                    if reward == _UNREACHABLE:
                        continue
                    score = values[p] + reward
                    if score > best:
                        best, arg = score, p
                out[q] = max(out[q], best)
                # An unreachable midpoint gives no information about argmax order.
                upper_left = arg if arg >= 0 else pmax
                lower_right = arg if arg >= 0 else pmin
                stack[count] = (left, q - 1, pmin, upper_left)
                stack[count + 1] = (q + 1, right, lower_right, pmax)
                count += 2
    return out


@njit()
def _price_bounds(
    arrivals: NDArray[np.int64],
    valid: NDArray[np.bool_],
    bid: NDArray[np.int32],
    ask: NDArray[np.int32],
    nb: NDArray[np.int8],
    na: NDArray[np.int8],
) -> tuple[int, int]:
    low, high = (1 << 62), 0
    for row in arrivals:
        if row < 0 or not valid[row]:
            continue
        for k in range(nb[row]):
            low, high = min(low, int(bid[row, k])), max(high, int(bid[row, k]))
        for k in range(na[row]):
            low, high = min(low, int(ask[row, k])), max(high, int(ask[row, k]))
    return low, high


@njit()
def _integer_path(
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
    buy_num: int,
    buy_den: int,
    sell_num: int,
    sell_den: int,
    slip: int,
    entry_cutoff: int,
    accelerated: bool,
    absolute_bound: int,
) -> int:
    ring = np.full((ring_size, 2 * half + 1), _UNREACHABLE, np.int64)
    ring[0, half] = 0
    for i in range(len(times)):
        values = ring[i % ring_size].copy()
        o, r = observed[i], arrivals[i]
        if r >= 0 and valid[o] and valid[r]:
            buys = _integer_curve(
                ask[r], az[r], na[r], 1, int(ask[o, 0]) + slip, step, half, lot, fraction, buy_num, buy_den
            )
            sells = _integer_curve(
                bid[r], bs[r], nb[r], -1, int(bid[o, 0]) - slip, step, half, lot, fraction, sell_num, sell_den
            )
            eligible = ring[(predecessor[i] + 1) % ring_size]
            scores = (
                _integer_monge(eligible, buys, sells, times[i] >= entry_cutoff)
                if accelerated
                else _integer_quadratic(eligible, buys, sells, times[i] >= entry_cutoff)
            )
            for q in range(len(values)):
                if scores[q] != _UNREACHABLE:
                    if scores[q] < -absolute_bound or scores[q] > absolute_bound:
                        raise OverflowError("integer Bellman score exceeded its proven path bound")
                    values[q] = max(values[q], scores[q])
        ring[(i + 1) % ring_size] = values
    return int(ring[len(times) % ring_size, half])


def certify_adaptive_depth(
    tapes: MarketTapes,
    scenario: Scenario,
    session: SessionPlan,
    limits: RiskLimits = _DEFAULT_LIMITS,
    *,
    step_units: int = 1_000,
    candidate_spacing_ns: int = 1,
    include_spacing_ns: tuple[int, ...] = (1_000_000_000,),
    cooldown_ns: int | None = None,
    slip_ticks: int = 200,
    scale: int = 10**12,
    accelerated: bool = True,
) -> GraphCertificate:
    """Bound the exact-arithmetic optimum of the same adaptive cooldown graph.

    ``scale`` must be a positive power of ten. Exact fee/tick rationals and a
    conservative bound on every prefix are checked with Python big integers
    before any int64 kernel runs. Unsupported integer ranges are rejected.
    """
    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
        raise InvalidConfigError("scale must be a positive integer power of ten")
    reduced = scale
    while reduced > 1 and reduced % 10 == 0:
        reduced //= 10
    if reduced != 1:
        raise InvalidConfigError("scale must be a positive integer power of ten")
    minimum = max(1, scenario.order_latency_ns + scenario.ack_ns, scenario.consumption_ttl_ns)
    cooldown = minimum if cooldown_ns is None else cooldown_ns
    if isinstance(cooldown, bool) or not isinstance(cooldown, int) or not minimum <= cooldown <= 2**62:
        raise InvalidConfigError("cooldown must cover acknowledgements and liquidity expiry, and be <=2**62")
    if not isinstance(accelerated, bool) or not isinstance(include_spacing_ns, tuple):
        raise InvalidConfigError("accelerated must be bool and include_spacing_ns must be a tuple")
    observed, times, arrivals, valid = _depth_inputs(
        tapes, scenario, session, limits, step_units, (candidate_spacing_ns, *include_spacing_ns), slip_ticks
    )
    cap, lot = limits.max_abs_position_units, tapes.instrument.lot_size
    fee, tick = Fraction(scenario.fee_for(tapes.instrument)), Fraction(tapes.instrument.tick_size)
    buy = scale * lot * (1 - fee) / tick
    sell = scale * lot * (-1 - fee) / tick
    b = tapes.book
    min_price, max_price = _price_bounds(arrivals, valid, b.bid_px, b.ask_px, b.n_bid, b.n_ask)
    coefficients = (buy.numerator, buy.denominator, sell.numerator, sell.denominator)
    if any(abs(v) > _SAFE for v in coefficients) or max(buy.denominator, sell.denominator) * max_price > _SAFE:
        raise InvalidConfigError("exact rate coefficients exceed checked int64 headroom; reduce scale")
    max_orders = min(len(times), 1 + (int(times[-1]) - int(times[0])) // cooldown) if len(times) else 0
    max_rate = 0
    if max_price:
        max_rate = max(abs(-((-rate.numerator) // (rate.denominator * min_price))) for rate in (buy, sell))
    # No sign-crossing edge is admitted, so each order changes at most the position cap in stored units.
    absolute_bound = max_orders * (cap // lot) * max_rate
    if absolute_bound > _SAFE:
        raise InvalidConfigError("integer path bound exceeds checked int64 headroom; reduce scale")
    predecessor = np.searchsorted(times, times - cooldown, side="right") - 1
    ring_size = int(np.max(np.arange(len(times)) - predecessor, initial=0)) + 2
    upper = _integer_path(
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
        cap // step_units,
        lot,
        scenario.depth_fraction,
        buy.numerator,
        buy.denominator,
        sell.numerator,
        sell.denominator,
        slip_ticks,
        session.entry_cutoff_ns,
        accelerated,
        absolute_bound,
    )
    rounding = max_orders * (cap // lot)
    return GraphCertificate(
        tapes.instrument.key,
        upper,
        scale,
        _fixed_decimal(upper, scale),
        len(times),
        step_units,
        cap,
        lot,
        cooldown,
        slip_ticks,
        candidate_spacing_ns,
        include_spacing_ns,
        max_orders,
        rounding,
        _fixed_decimal(rounding, scale),
        absolute_bound,
        accelerated,
    )
