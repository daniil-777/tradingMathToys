"""Tighten a fixed-shadow depth certificate with per-price consumption memory.

For a half-open block no longer than the consumption TTL, protect against live
entry eviction before asserting that earlier fills at a price remain consumed.
Protected blocks have total price/side fills at most the largest fresh displayed
whole-lot size there. Other blocks retain the actual-ladder bucket residual sum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from financeai.core.errors import InvalidConfigError
from financeai.core.jit import njit
from financeai.domains.market.tape import BookTape
from financeai.research.depth_shadow_bound import (
    BoolArray,
    ExactNumber,
    IntArray,
    LevelArray,
    SmallArray,
    _constants,
    _integer,
    certify_depth_shadow,
)

_SAFE = 2**62 - 1


@dataclass(frozen=True)
class TTLShadowBound:
    """An exactly evaluated intersection of two valid block residual bounds.

    The stored capacity residual is saturated at the corresponding base block
    residual. It is the exact minimum, not the uncapped capacity sum. No claim
    of optimizing the supplied shadow or finding an attainable strategy is made.
    """

    objective_units: int
    total_variation_units: int
    base_depth_objective_units: int
    rate_scale: int
    block_first_bucket: IntArray
    block_stop_bucket: IntArray
    base_block_residual_units: IntArray
    capacity_capped_residual_units: IntArray
    consumption_ttl_ns: int
    block_buckets: int
    phase_buckets: int
    candidate_book_checks: int
    strictly_improved_blocks: int
    maximum_price_side_pairs_per_block: int
    consumption_slots_per_side: int | None
    unprotected_blocks_retained: int


@njit(cache=False)
def _capacity_minimum(
    buy_capacity: dict[int, int],
    sell_capacity: dict[int, int],
    minimum_shadow: int,
    maximum_shadow: int,
    buy_numerator: int,
    sell_numerator: int,
    denominator: int,
    ceiling: int,
) -> int:
    """Exactly min(ceiling, capacity sum), with no overflowing products."""
    total = 0
    if ceiling == 0:
        return total
    for price, quantity in buy_capacity.items():
        if price == 0:
            continue
        divisor = denominator * price
        rate = buy_numerator // divisor + int(buy_numerator % divisor != 0)
        marginal = rate - minimum_shadow
        if marginal > 0:
            remaining = ceiling - total
            if quantity > remaining // marginal:
                return ceiling
            total += quantity * marginal
            if total == ceiling:
                return ceiling
    for price, quantity in sell_capacity.items():
        if price == 0:
            continue
        rate = sell_numerator // (denominator * price)
        marginal = maximum_shadow - rate
        if marginal > 0:
            remaining = ceiling - total
            if quantity > remaining // marginal:
                return ceiling
            total += quantity * marginal
            if total == ceiling:
                return ceiling
    return total


@njit(cache=False)
def _ttl_blocks(
    env: IntArray,
    valid: BoolArray,
    nb: SmallArray,
    na: SmallArray,
    bp: LevelArray,
    bs: LevelArray,
    ap: LevelArray,
    az: LevelArray,
    shadow: IntArray,
    residual: IntArray,
    start: int,
    end: int,
    period: int,
    width: int,
    phase: int,
    buy_numerator: int,
    sell_numerator: int,
    denominator: int,
    lot: int,
    slots: int,
    maximum_distinct_fills_per_order: int,
) -> tuple[IntArray, IntArray, IntArray, IntArray, int, int, int]:
    n = len(shadow)
    count = (n + width - 1) // width if phase == 0 else 1 + (max(0, n - phase) + width - 1) // width
    if n == 0:
        count = 0
    first = np.empty(count, np.int64)
    stop = np.empty(count, np.int64)
    base = np.zeros(count, np.int64)
    tightened = np.zeros(count, np.int64)
    # A zero-price sentinel fixes integer key/value types for the optional JIT.
    # Valid executable prices are positive; dictionaries stay sparse even when
    # a tape contains extreme deep prices separated by billions of ticks.
    buy_capacity, sell_capacity = {0: 0}, {0: 0}
    previous, checks, maximum_pairs, unprotected = -1, 0, 0, 0
    left_bucket = 0
    for block in range(count):
        right_bucket = min(n, phase if block == 0 and phase else left_bucket + width)
        first[block], stop[block] = left_bucket, right_bucket
        minimum_shadow, maximum_shadow = int(shadow[left_bucket]), int(shadow[left_bucket])
        for k in range(left_bucket, right_bucket):
            minimum_shadow = min(minimum_shadow, int(shadow[k]))
            maximum_shadow = max(maximum_shadow, int(shadow[k]))
            value = int(residual[k])
            if base[block] > _SAFE - value:
                raise OverflowError("TTL block residual sum exceeds checked int64 range")
            base[block] += value
        left = start + left_bucket * period
        right = min(start + right_bucket * period, end)
        while previous + 1 < len(env) and env[previous + 1] <= left:
            previous += 1
        row = previous
        buy_capacity.clear()
        sell_capacity.clear()
        while True:
            if row >= 0 and valid[row]:
                checks += 1
                for level in range(na[row]):
                    price = int(ap[row, level])
                    quantity = int(az[row, level]) // lot * lot
                    buy_capacity[price] = max(buy_capacity.get(price, 0), quantity)
                for level in range(nb[row]):
                    price = int(bp[row, level])
                    quantity = int(bs[row, level]) // lot * lot
                    sell_capacity[price] = max(sell_capacity.get(price, 0), quantity)
            if row + 1 >= len(env) or env[row + 1] >= right:
                break
            row += 1
        previous = row
        maximum_pairs = max(maximum_pairs, len(buy_capacity) + len(sell_capacity))
        possible_fills = (right_bucket - left_bucket) * maximum_distinct_fills_per_order
        if slots > 0 and min(max(len(buy_capacity), len(sell_capacity)), possible_fills) > slots:
            # A bounded table may evict live consumption. Keep the original
            # residual unless every possible within-block price can be retained.
            tightened[block] = base[block]
            unprotected += 1
        else:
            tightened[block] = _capacity_minimum(
                buy_capacity,
                sell_capacity,
                minimum_shadow,
                maximum_shadow,
                buy_numerator,
                sell_numerator,
                denominator,
                int(base[block]),
            )
        left_bucket = right_bucket
    return first, stop, base, tightened, checks, maximum_pairs, unprotected


def certify_ttl_shadow(
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
    consumption_ttl_ns: int,
    block_buckets: int,
    phase_buckets: int = 0,
    rate_scale: int = 10**14,
    consumption_slots_per_side: int | None = 32,
) -> TTLShadowBound:
    """Intersect per-bucket depth bounds with consumption-memory capacities.

    Inherits every trading, tape, cash-flow and coverage assumption documented
    by ``certify_depth_shadow``. Additionally, takes at a given (price, side)
    either remain unavailable for ``consumption_ttl_ns``, or use a bounded table
    that evicts the soonest-expiring entry only when full and refreshes expiry
    only on an actual fill at that price. Replenishment does not erase consumption.
    The default 32 slots match the compiled engine. Pass ``None`` only for an
    unbounded table such as the reference engine's dictionary.

    All blocks align with shadow buckets. ``phase_buckets=0`` starts full blocks
    at bucket0. A positive phase creates a shorter first block, then full blocks.
    The last block may also be partial. Full width must not exceed the TTL.

    At each (price, side), use the maximum displayed whole-lot level size among
    every candidate arrival book in the block. Pair ask capacities with the
    block's smallest shadow and bid capacities with its largest shadow. These
    independent extrema are optimistic. Then retain the smaller of that capacity
    sum and the original depth residual sum. Both bound the same block residual;
    the unchanged inventory total-variation term completes the profit bound.

    The base depth certificate is recomputed and audited internally. Sparse
    dictionaries avoid allocating arrays across extreme deep-price tick ranges.
    Integer products are saturated by division before multiplication, so only
    values at most the checked base block sum are formed. Final totals use Python
    integers. Runtime is linear in candidate levels plus shadow buckets.
    """
    ttl = _integer("consumption_ttl_ns", consumption_ttl_ns)
    width = _integer("block_buckets", block_buckets)
    phase = _integer("phase_buckets", phase_buckets, positive=False)
    period = _integer("bucket_ns", bucket_ns)
    if not 0 <= phase < width or width * period > ttl:
        raise InvalidConfigError("require 0 <= phase_buckets < block_buckets and block duration <= consumption TTL")
    original = certify_depth_shadow(
        book,
        shadow_rates,
        start_ns=start_ns,
        end_ns=end_ns,
        bucket_ns=bucket_ns,
        minimum_arrival_gap_ns=minimum_arrival_gap_ns,
        cap_units=cap_units,
        lot_units=lot_units,
        tick_size=tick_size,
        taker_fee=taker_fee,
        rate_scale=rate_scale,
    )
    if int(end_ns) + ttl > 2**63 - 1:
        raise InvalidConfigError("arrival plus consumption TTL lacks int64 clock headroom")
    slots = (
        0
        if consumption_slots_per_side is None
        else _integer(
            "consumption_slots_per_side",
            consumption_slots_per_side,
        )
    )
    buy_num, sell_num, denominator = _constants(tick_size, taker_fee, original.rate_scale)
    valid = (book.kind == 0) & (book.n_bid > 0) & (book.n_ask > 0) & (book.bid_px[:, 0] < book.ask_px[:, 0])
    first, stop, base, tightened, checks, pairs, unprotected = _ttl_blocks(
        np.maximum.accumulate(book.exch_ns),
        valid,
        book.n_bid,
        book.n_ask,
        book.bid_px,
        book.bid_sz,
        book.ask_px,
        book.ask_sz,
        shadow_rates,
        original.bucket_residual_units,
        int(start_ns),
        int(end_ns),
        period,
        width,
        phase,
        buy_num,
        sell_num,
        denominator,
        int(lot_units),
        slots,
        min(book.bid_px.shape[1], original.maximum_order_face_units // int(lot_units)),
    )
    if sum(map(int, base)) != original.residual_units:
        raise AssertionError("TTL blocks did not partition every original residual")
    for array in (first, stop, base, tightened):
        array.setflags(write=False)
    return TTLShadowBound(
        objective_units=original.total_variation_units + sum(map(int, tightened)),
        total_variation_units=original.total_variation_units,
        base_depth_objective_units=original.objective_units,
        rate_scale=original.rate_scale,
        block_first_bucket=first,
        block_stop_bucket=stop,
        base_block_residual_units=base,
        capacity_capped_residual_units=tightened,
        consumption_ttl_ns=ttl,
        block_buckets=width,
        phase_buckets=phase,
        candidate_book_checks=int(checks),
        strictly_improved_blocks=int(np.count_nonzero(tightened < base)),
        maximum_price_side_pairs_per_block=int(pairs),
        consumption_slots_per_side=consumption_slots_per_side,
        unprotected_blocks_retained=int(unprotected),
    )
