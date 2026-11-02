"""Exact integer shadow certificates for the local replay engine.

A single order is limited to the inventory cap because the generic engine closes
through zero. At most one side can execute in a latency-plus-ack bucket. The
optimizer uses the resulting maximum-of-side-residuals loss. It is an upper-bound
research tool, not a causal trading strategy. See docs/shadow_bound.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError
from financeai.core.jit import njit

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
_SAFE = 2**62 - 1
SAFE = _SAFE


@dataclass(frozen=True)
class ShadowBound:
    """An exact objective value and independently verifiable integer witness.

    Divide objective_units by the scale used to construct the input rates for
    an upper bound in settlement currency. Its validity for a market requires optimistic bucket
    inputs and the declared local-engine order constraints.
    The continuous max-side shadow optimum lies between objective_units minus
    continuous_objective_gap_units and objective_units. That interval is for
    the relaxation; its lower endpoint is not executable profit.
    """

    shadow_rates: IntArray
    tv_dual: IntArray
    objective_units: int
    maximum_absolute_intercept: int
    integer_product_bound: int
    distinct_breakpoint_rates: int
    continuous_objective_gap_units: int


@njit(cache=False)
def _add_checked(a: int, b: int) -> int:
    if (b > 0 and a > SAFE - b) or (b < 0 and a < -SAFE - b):
        raise OverflowError("exact shadow intercept exceeded checked integer range")
    return a + b


@njit(cache=False)
def _push(node: int, mass: IntArray, moment: IntArray, clear: BoolArray) -> None:
    if clear[node]:
        for child in (2 * node, 2 * node + 1):
            mass[child], moment[child], clear[child] = 0, 0, True
        clear[node] = False


@njit(cache=False)
def _insert(
    node: int,
    lo: int,
    hi: int,
    index: int,
    weight: int,
    rates: IntArray,
    mass: IntArray,
    moment: IntArray,
    clear: BoolArray,
) -> None:
    if lo == hi:
        mass[node] += weight
        moment[node] += weight * rates[index]
        clear[node] = False
        return
    _push(node, mass, moment, clear)
    mid = (lo + hi) // 2
    if index <= mid:
        _insert(2 * node, lo, mid, index, weight, rates, mass, moment, clear)
    else:
        _insert(2 * node + 1, mid + 1, hi, index, weight, rates, mass, moment, clear)
    mass[node] = mass[2 * node] + mass[2 * node + 1]
    moment[node] = moment[2 * node] + moment[2 * node + 1]


@njit(cache=False)
def _select(node: int, lo: int, hi: int, target: int, mass: IntArray, moment: IntArray, clear: BoolArray) -> int:
    """First breakpoint whose prefix mass reaches positive target."""
    while lo < hi:
        _push(node, mass, moment, clear)
        mid = (lo + hi) // 2
        if target <= mass[2 * node]:
            node, hi = 2 * node, mid
        else:
            target -= mass[2 * node]
            node, lo = 2 * node + 1, mid + 1
    return lo


@njit(cache=False)
def _remove(
    node: int,
    lo: int,
    hi: int,
    amount: int,
    left: bool,
    rates: IntArray,
    mass: IntArray,
    moment: IntArray,
    clear: BoolArray,
) -> int:
    """Delete an extreme amount of mass; return its exact first moment."""
    if amount == 0:
        return 0
    if amount == mass[node]:
        answer = moment[node]
        mass[node], moment[node], clear[node] = 0, 0, True
        return int(answer)
    if lo == hi:
        answer = amount * rates[lo]
        mass[node] -= amount
        moment[node] -= answer
        return int(answer)
    _push(node, mass, moment, clear)
    mid = (lo + hi) // 2
    if left:
        take = min(amount, mass[2 * node])
        answer = _remove(2 * node, lo, mid, take, left, rates, mass, moment, clear)
        answer += _remove(2 * node + 1, mid + 1, hi, amount - take, left, rates, mass, moment, clear)
    else:
        take = min(amount, mass[2 * node + 1])
        answer = _remove(2 * node + 1, mid + 1, hi, take, left, rates, mass, moment, clear)
        answer += _remove(2 * node, lo, mid, amount - take, left, rates, mass, moment, clear)
    mass[node] = mass[2 * node] + mass[2 * node + 1]
    moment[node] = moment[2 * node] + moment[2 * node + 1]
    return answer


@njit(cache=False)
def _prefix(
    node: int, lo: int, hi: int, index: int, mass: IntArray, moment: IntArray, clear: BoolArray
) -> tuple[int, int]:
    if index < lo:
        return 0, 0
    if hi <= index:
        return mass[node], moment[node]
    _push(node, mass, moment, clear)
    mid = (lo + hi) // 2
    a, b = _prefix(2 * node, lo, mid, index, mass, moment, clear)
    c, d = _prefix(2 * node + 1, mid + 1, hi, index, mass, moment, clear)
    return a + c, b + d


@njit(cache=False)
def _knots(b: IntArray, s: IntArray, u: IntArray, v: IntArray) -> tuple[IntArray, IntArray, IntArray, IntArray]:
    """Two-hinge representation of integer interpolation of max-side loss."""
    k1, k2 = b.copy(), s.copy()
    w1, w2 = u.copy(), v.copy()
    for i in range(len(b)):
        if b[i] == 0:
            w1[i], w2[i] = 0, 0
        elif b[i] > s[i] and u[i] + v[i] > 0:
            total = u[i] + v[i]
            numerator = u[i] * b[i] + v[i] * s[i]
            lo = numerator // total
            middle_slope = total * lo + v[i] - numerator
            k1[i], k2[i] = lo, lo + 1
            w1[i], w2[i] = middle_slope + u[i], v[i] - middle_slope
    return k1, k2, w1, w2


@njit(cache=False)
def _solve_max(
    b: IntArray,
    u: IntArray,
    cap: int,
    k1: IntArray,
    k2: IntArray,
    w1: IntArray,
    w2: IntArray,
    rates: IntArray,
    index1: IntArray,
    index2: IntArray,
) -> tuple[IntArray, int, int]:
    n, p = len(b), len(rates)
    mass, moment = np.zeros(4 * p + 4, np.int64), np.zeros(4 * p + 4, np.int64)
    clear = np.zeros(4 * p + 4, np.bool_)
    lower, upper = np.empty(n, np.int64), np.empty(n, np.int64)
    intercept, slope, maximum_intercept = 0, 0, 0
    for i in range(n):
        lower[i], upper[i] = rates[0], rates[-1]
        if b[i] > 0:
            intercept = _add_checked(intercept, u[i] * b[i])
            slope -= u[i]
            if w1[i]:
                _insert(1, 0, p - 1, index1[i], w1[i], rates, mass, moment, clear)
            if w2[i]:
                _insert(1, 0, p - 1, index2[i], w2[i], rates, mass, moment, clear)
            maximum_intercept = max(maximum_intercept, abs(intercept))
            if slope < -cap:
                amount = -cap - slope
                lower[i] = rates[_select(1, 0, p - 1, amount, mass, moment, clear)]
                removed = _remove(1, 0, p - 1, amount, True, rates, mass, moment, clear)
                intercept = _add_checked(intercept, -removed)
                slope += amount
            if slope + mass[1] > cap:
                amount = slope + mass[1] - cap
                upper[i] = rates[_select(1, 0, p - 1, mass[1] - amount + 1, mass, moment, clear)]
                _remove(1, 0, p - 1, amount, False, rates, mass, moment, clear)
            if lower[i] > upper[i] or not -cap <= slope <= 0 or not 0 <= slope + mass[1] <= cap:
                raise AssertionError("order-aware slope invariant failed")
    index = 0 if slope == 0 else _select(1, 0, p - 1, -slope, mass, moment, clear)
    x = rates[index]
    weight, first_moment = _prefix(1, 0, p - 1, index - 1, mass, moment, clear)
    objective = _add_checked(intercept, (slope + weight) * x - first_moment)
    shadow = np.empty(n, np.int64)
    if n:
        shadow[-1] = x
        for i in range(n - 2, -1, -1):
            shadow[i] = min(upper[i], max(lower[i], shadow[i + 1]))
    return shadow, objective, maximum_intercept


@njit(cache=False)
def _loss(x: int, b: int, s: int, u: int, v: int) -> int:
    return max(0, u * (b - x), v * (x - s)) if b else 0


@njit(cache=False)
def _gradient(x: int, b: int, s: int, u: int, v: int) -> tuple[int, int]:
    # Slopes of the linear interpolation of exact losses at integer x.
    # This formula is independent of the two-knot construction.
    middle = _loss(x, b, s, u, v)
    return middle - _loss(x - 1, b, s, u, v), _loss(x + 1, b, s, u, v) - middle


@njit(cache=False)
def _kkt(shadow: IntArray, b: IntArray, s: IntArray, u: IntArray, v: IntArray, cap: int) -> IntArray:
    n = len(shadow)
    low, high = np.zeros(n + 1, np.int64), np.zeros(n + 1, np.int64)
    for i in range(n):
        gl, gh = _gradient(shadow[i], b[i], s[i], u[i], v[i])
        al, ah = -cap, cap
        if i == n - 1:
            al, ah = 0, 0
        elif shadow[i + 1] > shadow[i]:
            al, ah = cap, cap
        elif shadow[i + 1] < shadow[i]:
            al, ah = -cap, -cap
        low[i + 1], high[i + 1] = max(al, low[i] + gl), min(ah, high[i] + gh)
        if low[i + 1] > high[i + 1]:
            raise AssertionError("order-aware KKT unreachable")
    dual = np.zeros(n + 1, np.int64)
    for i in range(n - 1, -1, -1):
        gl, gh = _gradient(shadow[i], b[i], s[i], u[i], v[i])
        lo, hi = max(low[i], dual[i + 1] - gh), min(high[i], dual[i + 1] - gl)
        if lo > hi:
            raise AssertionError("order-aware KKT backtrack failed")
        dual[i] = min(hi, max(lo, 0))
    _verify(shadow, dual, b, s, u, v, cap)
    return dual


@njit(cache=False)
def _verify(shadow: IntArray, dual: IntArray, b: IntArray, s: IntArray, u: IntArray, v: IntArray, cap: int) -> None:
    if dual[0] != 0 or dual[-1] != 0:
        raise AssertionError("wrong KKT endpoints")
    for i in range(len(shadow)):
        gl, gh = _gradient(shadow[i], b[i], s[i], u[i], v[i])
        if not gl <= dual[i + 1] - dual[i] <= gh or abs(dual[i]) > cap:
            raise AssertionError("wrong KKT subgradient")
        if i and ((shadow[i] > shadow[i - 1] and dual[i] != cap) or (shadow[i] < shadow[i - 1] and dual[i] != -cap)):
            raise AssertionError("wrong KKT TV sign")


@njit(cache=False)
def _objective(shadow: IntArray, b: IntArray, s: IntArray, u: IntArray, v: IntArray, cap: int) -> int:
    total = 0
    for i in range(len(shadow)):
        if i:
            total = _add_checked(total, cap * abs(shadow[i] - shadow[i - 1]))
        total = _add_checked(total, _loss(shadow[i], b[i], s[i], u[i], v[i]))
    return total


@njit(cache=False)
def _interpolation_gap(b: IntArray, s: IntArray, u: IntArray, v: IntArray) -> int:
    """Outward enclosure of the summed maximum interpolation excess."""
    error = 0
    for i in range(len(b)):
        total = int(u[i] + v[i])
        if b[i] > s[i] and total > 0:
            remainder = int((u[i] * b[i] + v[i] * s[i]) % total)
            if remainder:
                other = total - remainder
                if remainder <= SAFE // other:
                    product = remainder * other
                    addition = product // total + (1 if product % total else 0)
                else:
                    # The parabola r*(total-r)/total never exceeds total/4.
                    addition = total // 4 + (1 if total % 4 else 0)
                error = _add_checked(error, addition)
    return error


def _validate_inputs(
    buy: IntArray, sell: IntArray, buy_depth: IntArray, sell_depth: IntArray, cap: int
) -> tuple[int, int]:
    arrays = (buy, sell, buy_depth, sell_depth)
    if any(not isinstance(a, np.ndarray) or a.ndim != 1 or a.dtype != np.dtype(np.int64) for a in arrays):
        raise InvalidConfigError("shadow inputs must be one-dimensional int64 arrays")
    if len({len(a) for a in arrays}) != 1 or isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise InvalidConfigError("matching shadow input lengths and positive integer cap required")
    if any(np.any(a < 0) for a in arrays) or np.any((buy > 0) & (sell <= 0)):
        raise InvalidConfigError("nonnegative shadow inputs and paired active rates required")
    maximum = max(int(buy.max(initial=0)), int(sell.max(initial=0))) + 1
    bound = 8 * cap * maximum
    if bound > _SAFE:
        raise InvalidConfigError("shadow input products exceed conservative int64 headroom")
    return maximum, bound


def optimize_order_aware_shadow(
    buy_rates: IntArray, sell_rates: IntArray, buy_depth_units: IntArray, sell_depth_units: IntArray, cap_units: int
) -> ShadowBound:
    """Minimize an order-aware upper certificate over integer shadow rates.

    Rates must already be optimistic integers: buy credits rounded upward and
    sell costs downward in a caller-declared common scale. ``buy_rates==0``
    marks a missing bucket. Depths are clipped to ``cap_units``. The objective is
    C*TV(x)+sum(max(0,min(buy_depth,C)*(buy-x),min(sell_depth,C)*(x-sell))).

    Assumptions needed for a market upper bound: at most one same-venue order
    per bucket, its face quantity at most C, terminal-flat inventory in [-C,C],
    and optimistic rates/depths covering every possible arrival book. These
    are properties of the local reference engine and bucket construction;
    they do not follow from the original example's position cap alone.

    Exact convex interpolation on integer rate units supplies a Bellman
    optimizer and an independent KKT witness. No optimality over all continuous
    shadows, bucket phases, executable strategies, or future outcomes is claimed.
    """
    _, bound = _validate_inputs(buy_rates, sell_rates, buy_depth_units, sell_depth_units, cap_units)
    active = buy_rates > 0
    if not np.any(active):
        return ShadowBound(
            np.zeros(len(buy_rates), np.int64), np.zeros(len(buy_rates) + 1, np.int64), 0, 0, bound, 0, 0
        )
    u, v = np.minimum(buy_depth_units, cap_units), np.minimum(sell_depth_units, cap_units)
    k1, k2, w1, w2 = _knots(buy_rates, sell_rates, u, v)
    rates = np.unique(np.concatenate((k1[active], k2[active])))
    i1, i2 = np.searchsorted(rates, k1), np.searchsorted(rates, k2)
    shadow, value, maximum = _solve_max(buy_rates, u, cap_units, k1, k2, w1, w2, rates, i1, i2)
    del k1, k2, w1, w2, i1, i2
    if int(value) != int(_objective(shadow, buy_rates, sell_rates, u, v, cap_units)):
        raise AssertionError("shadow Bellman value differs from independent objective")
    dual = _kkt(shadow, buy_rates, sell_rates, u, v, cap_units)
    error = _interpolation_gap(buy_rates, sell_rates, u, v)
    return ShadowBound(shadow, dual, int(value), int(maximum), bound, len(rates), int(error))


def verify_order_aware_shadow(
    shadow_rates: IntArray,
    tv_dual: IntArray,
    buy_rates: IntArray,
    sell_rates: IntArray,
    buy_depth_units: IntArray,
    sell_depth_units: IntArray,
    cap_units: int,
) -> int:
    """Check a complete integer optimality witness and return its objective.

    Verification uses local subgradient conditions and direct objective
    evaluation; it does not run the Bellman optimizer or breakpoint tree.
    Input coverage, dtype, domain and arithmetic headroom are checked first.
    """
    maximum, _ = _validate_inputs(buy_rates, sell_rates, buy_depth_units, sell_depth_units, cap_units)
    if any(
        not isinstance(a, np.ndarray) or a.ndim != 1 or a.dtype != np.dtype(np.int64) for a in (shadow_rates, tv_dual)
    ):
        raise InvalidConfigError("shadow witnesses must be one-dimensional int64 arrays")
    if len(shadow_rates) != len(buy_rates) or len(tv_dual) != len(buy_rates) + 1:
        raise InvalidConfigError("shadow witness must cover every bucket")
    if np.any(shadow_rates < 0) or np.any(shadow_rates > maximum):
        raise InvalidConfigError("shadow witness lies outside its supported rate domain")
    if np.any(tv_dual < -cap_units) or np.any(tv_dual > cap_units):
        raise InvalidConfigError("shadow dual exceeds the inventory cap")
    u, v = np.minimum(buy_depth_units, cap_units), np.minimum(sell_depth_units, cap_units)
    _verify(shadow_rates, tv_dual, buy_rates, sell_rates, u, v, cap_units)
    return int(_objective(shadow_rates, buy_rates, sell_rates, u, v, cap_units))
