"""Exact fixed-shadow upper bounds using the displayed inverse-contract ladders.

The certificate assumes terminal-flat inventory bounded by ``cap_units``, orders
bounded by that same cap, and at most one arrival in each half-open bucket.
Those assumptions hold in the local replay engine when bucket width is no larger
than order latency plus acknowledgement latency. This is a historical upper
bound, not a trading policy or a proof that the upper bound can be attained.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from math import lcm

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.core.jit import njit
from financeai.domains.market.tape import BookTape

IntArray = NDArray[np.int64]
SmallArray = NDArray[np.int8]
LevelArray = NDArray[np.int32]
BoolArray = NDArray[np.bool_]
ExactNumber = Decimal | Fraction | str
_SAFE = 2**62 - 1
_INT64_MAX = 2**63 - 1


@dataclass(frozen=True)
class DepthShadowBound:
    """Exactly evaluated fixed shadow; divide monetary units by ``rate_scale``.

    ``bucket_residual_units`` contains maxima over complete arrival-book ladders,
    both order sides, and all whole-lot quantities up to the cap. The result makes
    no claim that the supplied shadow minimizes this stronger objective.
    """

    objective_units: int
    total_variation_units: int
    residual_units: int
    rate_scale: int
    bucket_residual_units: IntArray
    audited_valid_book_rows: int
    candidate_book_checks: int
    maximum_order_face_units: int
    integer_product_bound: int


@njit(cache=False)
def _audit_ladders(
    kind: SmallArray,
    nb: SmallArray,
    na: SmallArray,
    bp: LevelArray,
    bs: LevelArray,
    ap: LevelArray,
    az: LevelArray,
) -> tuple[BoolArray, int, int, int, int, int]:
    valid = np.zeros(len(kind), np.bool_)
    minimum, maximum, count = _INT64_MAX, 0, 0
    minimum_ask, maximum_bid = _INT64_MAX, 0
    depth = bp.shape[1]
    for row in range(len(kind)):
        # Match the generic engine's validity predicate, then reject malformed
        # executable ladders instead of silently omitting profitable fills.
        if kind[row] != 0 or nb[row] <= 0 or na[row] <= 0 or bp[row, 0] >= ap[row, 0]:
            continue
        if nb[row] > depth or na[row] > depth:
            raise ValueError("executable level count exceeds stored depth")
        for level in range(nb[row]):
            price = int(bp[row, level])
            if price <= 0 or bs[row, level] <= 0:
                raise ValueError("executable bid price and size must be positive")
            if level and price > bp[row, level - 1]:
                raise ValueError("executable bids must be nonincreasing")
            minimum, maximum = min(minimum, price), max(maximum, price)
        for level in range(na[row]):
            price = int(ap[row, level])
            if price <= 0 or az[row, level] <= 0:
                raise ValueError("executable ask price and size must be positive")
            if level and price < ap[row, level - 1]:
                raise ValueError("executable asks must be nondecreasing")
            minimum, maximum = min(minimum, price), max(maximum, price)
        valid[row] = True
        count += 1
        minimum_ask = min(minimum_ask, int(ap[row, 0]))
        maximum_bid = max(maximum_bid, int(bp[row, 0]))
    return valid, minimum, maximum, count, minimum_ask, maximum_bid


@njit(cache=False)
def _row_residual(
    row: int,
    shadow: int,
    nb: SmallArray,
    na: SmallArray,
    bp: LevelArray,
    bs: LevelArray,
    ap: LevelArray,
    az: LevelArray,
    buy_numerator: int,
    sell_numerator: int,
    denominator: int,
    cap: int,
    lot: int,
) -> int:
    remaining, buy = cap, 0
    for level in range(na[row]):
        divisor = denominator * int(ap[row, level])
        # Quotient plus remainder avoids overflow in numerator + divisor - 1.
        rate = buy_numerator // divisor + int(buy_numerator % divisor != 0)
        marginal = rate - shadow
        if marginal <= 0 or remaining == 0:
            break
        quantity = min(remaining, int(az[row, level]) // lot * lot)
        buy += quantity * marginal
        remaining -= quantity
    remaining, sell = cap, 0
    for level in range(nb[row]):
        divisor = denominator * int(bp[row, level])
        rate = sell_numerator // divisor
        marginal = shadow - rate
        if marginal <= 0 or remaining == 0:
            break
        quantity = min(remaining, int(bs[row, level]) // lot * lot)
        sell += quantity * marginal
        remaining -= quantity
    return max(buy, sell)


@njit(cache=False)
def _bucket_residuals(
    env: IntArray,
    valid: BoolArray,
    nb: SmallArray,
    na: SmallArray,
    bp: LevelArray,
    bs: LevelArray,
    ap: LevelArray,
    az: LevelArray,
    shadow: IntArray,
    start: int,
    end: int,
    period: int,
    buy_numerator: int,
    sell_numerator: int,
    denominator: int,
    cap: int,
    lot: int,
) -> tuple[IntArray, int]:
    residual = np.zeros(len(shadow), np.int64)
    previous, checks = -1, 0
    for k in range(len(shadow)):
        left = start + k * period
        right = min(left + period, end)
        while previous + 1 < len(env) and env[previous + 1] <= left:
            previous += 1
        row = previous
        while True:
            if row >= 0 and valid[row]:
                residual[k] = max(
                    residual[k],
                    _row_residual(
                        row,
                        int(shadow[k]),
                        nb,
                        na,
                        bp,
                        bs,
                        ap,
                        az,
                        buy_numerator,
                        sell_numerator,
                        denominator,
                        cap,
                        lot,
                    ),
                )
                checks += 1
            if row + 1 >= len(env) or env[row + 1] >= right:
                break
            row += 1
        previous = row
    return residual, checks


def _integer(name: str, value: int, *, positive: bool = True) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise InvalidConfigError(f"{name} must be an integer")
    value = int(value)
    if not -_INT64_MAX <= value <= _INT64_MAX or (positive and value <= 0):
        raise InvalidConfigError(f"{name} is outside its supported integer domain")
    return value


def _constants(tick_size: ExactNumber, taker_fee: ExactNumber, scale: int) -> tuple[int, int, int]:
    if not isinstance(tick_size, (str, Decimal, Fraction)) or not isinstance(taker_fee, (str, Decimal, Fraction)):
        raise InvalidConfigError("tick_size and taker_fee must be exact strings, Decimals, or Fractions")
    try:
        tick, fee = Fraction(tick_size), Fraction(taker_fee)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise InvalidConfigError("tick_size and taker_fee must be finite rational values") from exc
    if tick <= 0 or not 0 <= fee < 1:
        raise InvalidConfigError("require tick_size > 0 and 0 <= taker_fee < 1")
    buy, sell = (1 - fee) / tick, (1 + fee) / tick
    denominator = lcm(buy.denominator, sell.denominator)
    buy_num = buy.numerator * (denominator // buy.denominator) * scale
    sell_num = sell.numerator * (denominator // sell.denominator) * scale
    if max(buy_num, sell_num, denominator) > _INT64_MAX:
        raise InvalidConfigError("exact reciprocal-rate numerator lacks int64 headroom")
    return buy_num, sell_num, denominator


def certify_depth_shadow(
    book: BookTape,
    shadow_rates: IntArray,
    *,
    start_ns: int,
    end_ns: int,
    bucket_ns: int,
    minimum_arrival_gap_ns: int,
    cap_units: int,
    lot_units: int,
    tick_size: ExactNumber,
    taker_fee: ExactNumber,
    rate_scale: int = 10**14,
) -> DepthShadowBound:
    """Evaluate a rigorous actual-ladder bound for an arbitrary fixed shadow.

    The covered arrivals must lie in ``[start_ns, end_ns)``. This API derives the
    local engine's exchange-clock envelope from ``book.exch_ns``. Each bucket
    considers its left-edge effective book and every subsequent update before
    its right edge; including intermediate equal-clock updates is optimistic.

    Required trading assumptions: starting and terminal inventory are zero;
    absolute inventory and each order are bounded by ``cap_units``; order amounts
    and fills are whole ``lot_units`` multiples; displayed depth is never increased;
    and same-venue arrivals are separated by at least ``minimum_arrival_gap_ns``.
    Orders are single-side IOC sweeps of the supplied sorted ladders. The local
    generic engine provides these constraints by flattening before reversal and
    allowing one pending order until acknowledgement. Fees and ticks are exact.

    Prior consumption, IOC limits, receipt-time causality, session restrictions,
    and inter-bucket reachability are relaxed. Validate input hashes separately.
    Runtime is O(number_of_buckets + candidate_book_checks * displayed_depth);
    memory is O(number_of_rows + number_of_buckets). No floating arithmetic is
    used in the certificate; the final aggregation uses Python integers.
    """
    start = _integer("start_ns", start_ns, positive=False)
    end = _integer("end_ns", end_ns, positive=False)
    period = _integer("bucket_ns", bucket_ns)
    gap = _integer("minimum_arrival_gap_ns", minimum_arrival_gap_ns)
    cap = _integer("cap_units", cap_units)
    lot = _integer("lot_units", lot_units)
    scale = _integer("rate_scale", rate_scale)
    if end < start or period > gap:
        raise InvalidConfigError("require end_ns >= start_ns and bucket_ns <= minimum_arrival_gap_ns")
    count = (end - start + period - 1) // period
    if start + count * period > _INT64_MAX or count * period > _INT64_MAX:
        raise InvalidConfigError("bucket clock arithmetic lacks int64 headroom")
    if not isinstance(shadow_rates, np.ndarray) or shadow_rates.dtype != np.int64 or shadow_rates.ndim != 1:
        raise InvalidConfigError("shadow_rates must be a one-dimensional int64 array")
    if len(shadow_rates) != count:
        raise InvalidConfigError("shadow_rates must cover every bucket exactly")
    smallest = int(shadow_rates.min()) if count else 0
    largest = int(shadow_rates.max()) if count else 0
    if min(smallest, -largest) < -_SAFE:
        raise InvalidConfigError("shadow rate domain lacks checked integer headroom")
    buy_num, sell_num, denominator = _constants(tick_size, taker_fee, scale)
    try:
        valid, min_price, max_price, valid_rows, min_ask, max_bid = _audit_ladders(
            book.kind,
            book.n_bid,
            book.n_ask,
            book.bid_px,
            book.bid_sz,
            book.ask_px,
            book.ask_sz,
        )
    except ValueError as exc:
        raise InvalidDataError(str(exc)) from exc
    if denominator * int(max_price) > _INT64_MAX:
        raise InvalidConfigError("price denominator lacks int64 headroom")
    maximum_rate = (max(buy_num, sell_num) + int(min_price) - 1) // int(min_price) if valid_rows else 0
    rate_span = max(abs(smallest), abs(largest), abs(maximum_rate - smallest), abs(maximum_rate - largest))
    # Very low deep bids can create large negative sell marginals. They are
    # never multiplied: each side stops before its first nonpositive marginal.
    # Bound subtraction using all rates, but products using positive marginals.
    buy_divisor = denominator * int(min_ask) if valid_rows else 1
    max_buy_rate = (buy_num + buy_divisor - 1) // buy_divisor if valid_rows else 0
    min_sell_rate = sell_num // (denominator * int(max_bid)) if valid_rows else 0
    positive_span = max(0, max_buy_rate - smallest, largest - min_sell_rate) if valid_rows else 0
    maximum_order = cap // lot * lot
    product_bound = max(maximum_order * positive_span, cap * (largest - smallest))
    if rate_span > _SAFE or product_bound > _SAFE:
        raise InvalidConfigError("shadow residual or variation lacks checked integer headroom")
    env = np.maximum.accumulate(book.exch_ns)
    residual, checks = _bucket_residuals(
        env,
        valid,
        book.n_bid,
        book.n_ask,
        book.bid_px,
        book.bid_sz,
        book.ask_px,
        book.ask_sz,
        shadow_rates,
        start,
        end,
        period,
        buy_num,
        sell_num,
        denominator,
        maximum_order,
        lot,
    )
    # Each element/product is range-checked above. Python integers prevent the
    # many-bucket sum overflowing even when the overall objective exceeds int64.
    residual_total = sum(map(int, residual))
    variation_total = cap * sum(map(int, np.abs(np.diff(shadow_rates))))
    residual.setflags(write=False)
    return DepthShadowBound(
        objective_units=variation_total + residual_total,
        total_variation_units=variation_total,
        residual_units=residual_total,
        rate_scale=scale,
        bucket_residual_units=residual,
        audited_valid_book_rows=int(valid_rows),
        candidate_book_checks=int(checks),
        maximum_order_face_units=maximum_order,
        integer_product_bound=product_bound,
    )
