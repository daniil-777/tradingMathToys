"""Independent tiny allocation checks for the fixed-shadow depth certificate."""

from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.domains.market.reference import _sweep_reference
from financeai.domains.market.tape import BookTape
from financeai.research.depth_shadow_bound import certify_depth_shadow


def book(rows, clocks=None, kinds=None):
    n = len(rows)
    bid, ask, bid_size, ask_size = (np.zeros((n, 10), np.int32) for _ in range(4))
    nb, na = np.zeros(n, np.int8), np.zeros(n, np.int8)
    for i, (bids, asks) in enumerate(rows):
        nb[i], na[i] = len(bids), len(asks)
        for k, (price, size) in enumerate(bids):
            bid[i, k], bid_size[i, k] = price, size
        for k, (price, size) in enumerate(asks):
            ask[i, k], ask_size[i, k] = price, size
    clock = np.arange(n, dtype=np.int64) if clocks is None else np.array(clocks, np.int64)
    return BookTape(
        clock.copy(),
        clock,
        np.zeros(n, np.int8) if kinds is None else np.array(kinds, np.int8),
        nb,
        na,
        bid,
        bid_size,
        ask,
        ask_size,
    )


def bound(tape, shadows, **options):
    defaults = {
        "start_ns": 0,
        "end_ns": len(shadows) * 10,
        "bucket_ns": 10,
        "minimum_arrival_gap_ns": 10,
        "cap_units": 4,
        "lot_units": 1,
        "tick_size": "1",
        "taker_fee": "0.05",
        "rate_scale": 2520,
    }
    defaults.update(options)
    return certify_depth_shadow(tape, np.asarray(shadows, np.int64), **defaults)


def rounded_rate(price, side, scale=2520):
    exact = Fraction(scale) * (1 - side * Fraction(1, 20)) / price
    if side > 0:
        return (exact.numerator + exact.denominator - 1) // exact.denominator
    return exact.numerator // exact.denominator


def brute_allocations(levels, shadow, side, cap, lot):
    """Even allow arbitrary allocation among levels, including consumed prefixes."""
    best = 0
    for amounts in itertools.product(*(range(0, size // lot * lot + 1, lot) for _, size in levels)):
        if sum(amounts) <= cap:
            cash = sum(q * side * (rounded_rate(p, side) - shadow) for q, (p, _) in zip(amounts, levels, strict=True))
            best = max(best, cash)
    return best


@pytest.mark.parametrize("seed", range(20))
def test_ladder_maximum_equals_exhaustive_integer_allocations(seed):
    rng = np.random.default_rng(seed)
    bids = [(p, int(rng.integers(1, 5))) for p in (9, 8, 7)]
    asks = [(p, int(rng.integers(1, 5))) for p in (10, 11, 12)]
    shadow = int(rng.integers(180, 360))
    lot, cap = 1 + seed % 2, 3 + seed % 4
    result = bound(book([(bids, asks)]), [shadow], cap_units=cap, lot_units=lot)
    expected = max(brute_allocations(asks, shadow, 1, cap, lot), brute_allocations(bids, shadow, -1, cap, lot))
    assert result.objective_units == result.residual_units == expected
    assert result.total_variation_units == 0
    assert not result.bucket_residual_units.flags.writeable


def test_consumed_best_level_and_partial_lot_fill_are_still_dominated():
    bids, asks = [(9, 7), (8, 7), (7, 7)], [(10, 7), (11, 7), (12, 7)]
    tape = book([(bids, asks)])
    for side, levels, shadow in ((1, asks, 180), (-1, bids, 400)):
        result = bound(tape, [shadow], cap_units=9, lot_units=3)
        fills = _sweep_reference(
            side,
            9,
            100 if side > 0 else 0,
            [p for p, _ in levels],
            [q for _, q in levels],
            3,
            3,
            1.0,
            {levels[0][0]: [[6, 100]], levels[1][0]: [[3, 100]]},
            1,
            100,
        )
        assert fills[0][0] == levels[1][0]  # Actual fills skip the consumed top.
        actual = sum(Fraction(side * q * 2520) * (1 - side * Fraction(1, 20)) / p - side * q * shadow for p, q in fills)
        rounded = sum(side * q * (rounded_rate(p, side) - shadow) for p, q in fills)
        assert actual <= rounded <= result.residual_units


def test_coverage_carried_book_clear_boundary_ties_and_receding_exchange_clock():
    rows = [
        ([(9, 4)], [(10, 4)]),
        ([], []),
        ([(8, 4)], [(9, 4)]),
        ([(8, 4)], [(11, 4)]),
        ([(8, 4)], [(10, 4)]),
        ([(4, 4)], [(5, 4)]),
    ]
    tape = book(rows, clocks=[-1, 4, 10, 10, 9, 20], kinds=[0, 1, 0, 0, 0, 0])
    result = bound(tape, [180, 180], end_ns=20)
    # Bucket 0 includes the book effective at 0 before the clear at 4. Bucket 1
    # begins at 10: search-right selects the final row in the equal-clock cohort.
    assert result.bucket_residual_units.tolist() == [4 * (rounded_rate(10, 1) - 180)] * 2
    assert result.candidate_book_checks == 2
    # Moving the tied update inside a bucket may include intermediate rows. This
    # deliberate optimism agrees with the existing bucket relaxation's coverage.
    later = bound(tape, [180], start_ns=1, end_ns=20, bucket_ns=19, minimum_arrival_gap_ns=19)
    assert later.residual_units == 4 * (rounded_rate(9, 1) - 180)
    assert later.candidate_book_checks == 4


def test_actual_depth_strictly_improves_best_quote_times_unrelated_depth():
    tape = book([([(9, 4)], [(10, 1), (20, 100)]), ([(9, 4)], [(15, 100)])], clocks=[0, 5])
    result = bound(tape, [180])
    old_optimistic_top_residual = 4 * (rounded_rate(10, 1) - 180)
    assert result.objective_units == rounded_rate(10, 1) - 180
    assert result.objective_units < old_optimistic_top_residual


def test_outward_rounding_covers_every_exact_fractional_cash_flow():
    for price in range(1, 200):
        for side in (-1, 1):
            exact = Fraction(2520) * (1 - side * Fraction(1, 20)) / price
            rounded = rounded_rate(price, side)
            assert rounded >= exact if side > 0 else rounded <= exact


def test_total_variation_and_python_integer_sum_can_exceed_int64():
    # Each certified product is below 2**62; the sum deliberately exceeds int64.
    result = bound(book([]), [0, 10**12] * 10, cap_units=10**6)
    assert result.objective_units == result.total_variation_units == 19 * 10**18
    assert result.residual_units == 0
    assert bound(book([]), [], end_ns=0).objective_units == 0


def test_deep_bad_bid_has_no_positive_product_and_does_not_false_overflow():
    tape = book([([(50_000, 10**6), (1, 10**6)], [(50_001, 10**6)])])
    result = bound(tape, [4 * 10**9], cap_units=10**6, rate_scale=10**14, tick_size="0.5")
    assert result.objective_units == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"bucket_ns": 11},
        {"end_ns": 11},
        {"lot_units": 0},
        {"cap_units": True},
        {"tick_size": 0.5},
        {"tick_size": "0"},
        {"taker_fee": "1"},
        {"rate_scale": 10**30},
        {"cap_units": 10**18},
        {"start_ns": 2**63 - 10, "end_ns": 2**63 - 1},
    ],
)
def test_reject_invalid_or_overflowing_configurations(changes):
    with pytest.raises(InvalidConfigError):
        bound(book([([(9, 4)], [(10, 4)])]), [200], **changes)


@pytest.mark.parametrize(
    "rows",
    [
        [([(0, 4)], [(10, 4)])],
        [([(9, 0)], [(10, 4)])],
        [([(9, 4), (10, 4)], [(11, 4)])],
        [([(8, 4)], [(10, 4), (9, 4)])],
    ],
)
def test_reject_malformed_executable_ladders(rows):
    with pytest.raises(InvalidDataError):
        bound(book(rows), [200])
