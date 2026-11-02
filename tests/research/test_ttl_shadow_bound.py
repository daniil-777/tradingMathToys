"""Exhaustive small engine sweeps and boundary checks for TTL capacities."""

from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest

from financeai.core.errors import InvalidConfigError
from financeai.domains.market.execution import consumed_at, sweep
from financeai.domains.market.reference import _sweep_reference
from financeai.research.ttl_shadow_bound import _capacity_minimum, certify_ttl_shadow
from tests.research.test_depth_shadow_bound import book, rounded_rate


def bound(tape, shadows, **options):
    defaults = {
        "start_ns": 0,
        "end_ns": 10 * len(shadows),
        "bucket_ns": 10,
        "minimum_arrival_gap_ns": 10,
        "cap_units": 4,
        "lot_units": 1,
        "tick_size": "1",
        "taker_fee": "0.05",
        "rate_scale": 2520,
        "consumption_ttl_ns": 30,
        "block_buckets": 3,
        "phase_buckets": 0,
    }
    defaults.update(options)
    return certify_ttl_shadow(tape, np.asarray(shadows, np.int64), **defaults)


def test_repeated_quote_cannot_replenish_consumed_liquidity_inside_ttl():
    tape = book([([(9, 4)], [(10, 4)])] * 3, clocks=[0, 10, 20])
    result = bound(tape, [180] * 3)
    once = 4 * (rounded_rate(10, 1) - 180)
    assert result.base_depth_objective_units == 3 * once
    assert result.objective_units == once
    assert result.strictly_improved_blocks == 1


@pytest.mark.parametrize("seed", range(8))
def test_exhaustive_real_consumed_sweeps_are_bounded(seed):
    rng = np.random.default_rng(seed)
    rows = [([(9, int(rng.integers(1, 5))), (8, 3)], [(10, int(rng.integers(1, 5))), (11, 3)]) for _ in range(3)]
    shadows = rng.integers(180, 400, 3).tolist()
    result = bound(book(rows, clocks=[0, 10, 20]), shadows)
    choices = [(0, 0)] + [(side, quantity) for side in (-1, 1) for quantity in range(1, 5)]
    for commands in itertools.product(choices, repeat=3):
        consumed = {-1: {}, 1: {}}
        exact_residual, rounded_residual, position, profit = Fraction(0), 0, 0, Fraction(0)
        maximum_inventory = 0
        for k, (side, quantity) in enumerate(commands):
            if side == 0:
                continue
            levels = rows[k][0 if side < 0 else 1]
            fills = _sweep_reference(
                side,
                quantity,
                100 if side > 0 else 0,
                [p for p, _ in levels],
                [q for _, q in levels],
                len(levels),
                1,
                1.0,
                consumed[side],
                10 * k,
                30,
            )
            for price, face in fills:
                cash = Fraction(2520) * (side - Fraction(1, 20)) * face / price
                profit += cash
                exact_residual += cash - shadows[k] * side * face
                rounded_residual += side * face * (rounded_rate(price, side) - shadows[k])
                position += side * face
                maximum_inventory = max(maximum_inventory, abs(position))
        assert exact_residual <= rounded_residual <= sum(result.capacity_capped_residual_units)
        if position == 0 and maximum_inventory <= 4:
            assert profit <= result.objective_units


def test_capacity_is_maximum_size_not_sum_and_handles_duplicate_price_levels():
    tape = book([([(9, 4)], [(10, 3), (10, 5)]), ([(9, 4)], [(10, 7)])], clocks=[0, 10])
    result = bound(tape, [180, 180], cap_units=10, lot_units=2)
    assert result.objective_units == 6 * (rounded_rate(10, 1) - 180)


def test_ttl_boundary_allows_expired_capacity_in_next_half_open_block():
    tape = book([([(9, 4)], [(10, 4)])] * 4, clocks=[0, 10, 20, 30])
    result = bound(tape, [180] * 4)
    once = 4 * (rounded_rate(10, 1) - 180)
    assert result.block_first_bucket.tolist() == [0, 3]
    assert result.block_stop_bucket.tolist() == [3, 4]
    assert result.capacity_capped_residual_units.tolist() == [once, once]
    consumed = {}
    first = _sweep_reference(1, 4, 10, [10], [4], 1, 1, 1.0, consumed, 0, 30)
    inside = _sweep_reference(1, 4, 10, [10], [4], 1, 1, 1.0, consumed, 29, 30)
    boundary = _sweep_reference(1, 4, 10, [10], [4], 1, 1, 1.0, consumed, 30, 30)
    assert first == boundary == [(10, 4)]
    assert inside == []


def test_phase_partial_blocks_partition_all_buckets_and_exclude_right_boundary():
    rows = [([(9, 4)], [(10, 4)]), ([], []), ([(9, 8)], [(10, 8)])]
    tape = book(rows, clocks=[-1, 10, 40], kinds=[0, 1, 0])
    result = bound(tape, [180] * 5, phase_buckets=1)
    assert result.block_first_bucket.tolist() == [0, 1, 4]
    assert result.block_stop_bucket.tolist() == [1, 4, 5]
    assert result.capacity_capped_residual_units.tolist() == [240, 0, 240]
    assert not result.capacity_capped_residual_units.flags.writeable
    assert bound(tape, [], end_ns=0, phase_buckets=1).objective_units == 0
    tiny = bound(tape, [180], phase_buckets=2)
    assert tiny.block_first_bucket.tolist() == [0]
    assert tiny.block_stop_bucket.tolist() == [1]


def test_independent_integer_capacity_formula_matches_sparse_saturating_kernel():
    rng = np.random.default_rng(20261025)
    for _ in range(100):
        buy = {p: int(rng.integers(1, 10**9)) for p in range(1, 20)}
        sell = {p: int(rng.integers(1, 10**9)) for p in range(1, 20)}
        lo, hi = sorted(map(int, rng.integers(-(10**6), 10**6, 2)))
        buy_num, sell_num, den = 10**18, 10**18 + 1, 10
        ceiling = int(rng.integers(1, 2**62 - 1))
        exact = sum(q * max(0, (buy_num + den * p - 1) // (den * p) - lo) for p, q in buy.items())
        exact += sum(q * max(0, hi - sell_num // (den * p)) for p, q in sell.items())
        # Direct testing uses the Python implementation because ordinary Python
        # dictionaries are intentionally not a public JIT argument type.
        implementation = getattr(_capacity_minimum, "py_func", _capacity_minimum)
        assert implementation(buy, sell, lo, hi, buy_num, sell_num, den, ceiling) == min(exact, ceiling)


def test_compiled_33rd_price_eviction_requires_fallback_to_base_bound():
    prices = [100 + i for i in range(33)] + [100]
    rows = [([(price - 1, 1)], [(price, 1)]) for price in prices for _ in range(2)]
    tape = book(rows, clocks=list(range(0, 680, 10)))
    result = bound(tape, [0] * 68, cap_units=1, consumption_ttl_ns=1000, block_buckets=100)
    assert result.unprotected_blocks_retained == 1
    assert result.objective_units == result.base_depth_objective_units
    # Interleave sells so inventory returns to zero after each buy. The separate
    # buy table still evicts price100 before its revisit within the TTL.
    tables = {side: (np.zeros(32, np.int32), np.zeros(32, np.int64), np.zeros(32, np.int64)) for side in (-1, 1)}
    out_px, out_qty = np.zeros(10, np.int32), np.zeros(10, np.int64)
    visits = 0
    for i, price in enumerate(prices):
        for side, now, executed_price in ((1, i * 20, price), (-1, i * 20 + 10, price - 1)):
            count = sweep(
                side,
                1,
                executed_price,
                np.array([executed_price], np.int32),
                np.array([1], np.int32),
                1,
                1,
                1.0,
                *tables[side],
                now,
                1000,
                out_px,
                out_qty,
            )
            assert count == 1
            if side > 0 and price == 100:
                visits += int(out_qty[0])
    assert visits == 2
    unlimited = bound(
        tape, [0] * 68, cap_units=1, consumption_ttl_ns=1000, block_buckets=100, consumption_slots_per_side=None
    )
    assert unlimited.objective_units < result.objective_units


def test_outside_entry_expires_before_protected_inside_entries_even_at_left_edge():
    px, amount, expiry = np.zeros(2, np.int32), np.zeros(2, np.int64), np.zeros(2, np.int64)
    out_px, out_qty = np.zeros(10, np.int32), np.zeros(10, np.int64)
    for now, price in ((-1, 98), (0, 100), (0, 101)):
        assert (
            sweep(
                1,
                1,
                price,
                np.array([price], np.int32),
                np.array([1], np.int32),
                1,
                1,
                1.0,
                px,
                amount,
                expiry,
                now,
                30,
                out_px,
                out_qty,
            )
            == 1
        )
    assert consumed_at(px, amount, expiry, 98, 1) == 0
    assert consumed_at(px, amount, expiry, 100, 1) == 1
    assert consumed_at(px, amount, expiry, 101, 1) == 1


def test_arrival_count_protects_short_blocks_even_with_many_candidate_prices():
    rows = [([(90 + i, 1)], [(100 + i, 1)]) for i in range(40)]
    tape = book(rows, clocks=list(range(40)))
    result = bound(tape, [0, 0, 0, 0], cap_units=1, bucket_ns=10, consumption_ttl_ns=100, block_buckets=10)
    assert result.maximum_price_side_pairs_per_block == 80
    assert result.unprotected_blocks_retained == 0  # At most four one-lot fills.


def test_reject_expiry_clock_overflow_even_when_bucket_clocks_fit():
    with pytest.raises(InvalidConfigError, match=r"consumption TTL.*headroom"):
        bound(book([([(9, 4)], [(10, 4)])]), [180], start_ns=2**63 - 21, end_ns=2**63 - 11)


@pytest.mark.parametrize(
    "options",
    [
        {"block_buckets": 4},
        {"block_buckets": 0},
        {"phase_buckets": -1},
        {"phase_buckets": 3},
        {"consumption_ttl_ns": 0},
        {"phase_buckets": 0.5},
        {"consumption_slots_per_side": 0},
    ],
)
def test_reject_unsupported_ttl_partition(options):
    with pytest.raises(InvalidConfigError):
        bound(book([([(9, 4)], [(10, 4)])]), [180], **options)
