"""Unit and exhaustive sweep tests for multi-scale sub-TTL shadow bounds."""

from __future__ import annotations

import itertools
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from financeai.domains.market.reference import _sweep_reference
from financeai.research.multiscale_ttl_shadow import (
    certify_multiscale_ttl_shadow,
    select_optimal_multiscale_partition,
    verify_multiscale_ttl_partition,
)
from financeai.research.ttl_shadow_bound import certify_ttl_shadow
from tests.research.test_depth_shadow_bound import book, rounded_rate


def _multiscale_bound(tape, shadows, scales=(1, 2, 4), **options):
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
        "consumption_ttl_ns": 40,
        "scales_buckets": scales,
        "consumption_slots_per_side": 32,
    }
    defaults.update(options)
    return certify_multiscale_ttl_shadow(tape, np.asarray(shadows, np.int64), **defaults)


def test_multiscale_strictly_beats_every_uniform_scale_on_mixed_regime() -> None:
    # 4 buckets (ttl = 40ns -> up to 4 buckets):
    # - Buckets 0, 1: shadow=180, static ask at (10, 4), only 1 ask price (<= 2 slots).
    # - Buckets 2, 3: shadow jumps from 180 to 400 and exposes 3 distinct ask prices (> 2 slots),
    #   so a 4-bucket block has > 2 ask prices (unprotected under consumption_slots_per_side=2!),
    #   whereas splitting [0, 4) into [0, 2) (scale 2, protected!) + [2, 3) + [3, 4) (scale 1)
    #   protects [0, 2) AND keeps [2, 4) exact!
    rows = [
        ([(9, 4)], [(10, 4)]),
        ([(9, 4)], [(10, 4)]),
        ([(9, 4)], [(10, 4), (11, 4), (12, 4)]),
        ([(9, 4)], [(10, 4), (11, 4), (12, 4)]),
    ]
    tape = book(rows, clocks=[0, 10, 20, 30])
    shadows = [180, 180, 180, 400]
    res = _multiscale_bound(
        tape,
        shadows,
        scales=(1, 2, 4),
        consumption_slots_per_side=2,
    )
    assert res.objective_units <= res.best_single_scale_objective_units
    assert res.objective_units < res.single_scale_objectives[1]
    assert res.objective_units < res.single_scale_objectives[4]
    assert res.additional_decrease_units > 0
    assert res.upper_pnl < res.base_depth_upper_pnl


@pytest.mark.parametrize("seed", range(6))
def test_exhaustive_consumed_sweeps_respect_multiscale_bound(seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = [
        (
            [(9, int(rng.integers(1, 5))), (8, 3)],
            [(10, int(rng.integers(1, 5))), (11, 3)],
        )
        for _ in range(4)
    ]
    shadows = rng.integers(180, 400, 4).tolist()
    res = _multiscale_bound(book(rows, clocks=[0, 10, 20, 30]), shadows, scales=(1, 2, 4))
    choices = [(0, 0)] + [(side, q) for side in (-1, 1) for q in range(1, 5)]
    for commands in itertools.product(choices, repeat=4):
        consumed = {-1: {}, 1: {}}
        exact_residual = Fraction(0)
        rounded_residual = 0
        position = 0
        profit = Fraction(0)
        max_inv = 0
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
                40,
            )
            for price, face in fills:
                cash = Fraction(2520) * (side - Fraction(1, 20)) * face / price
                profit += cash
                exact_residual += cash - shadows[k] * side * face
                rounded_residual += side * face * (rounded_rate(price, side) - shadows[k])
                position += side * face
                max_inv = max(max_inv, abs(position))
        assert exact_residual <= rounded_residual <= int(np.sum(res.capacity_capped_residual_units))
        if position == 0 and max_inv <= 4:
            assert profit <= res.objective_units


def test_multiscale_validation_and_tamper_detection() -> None:
    tape = book([([(9, 4)], [(10, 4)])] * 4, clocks=[0, 10, 20, 30])
    shadows = np.array([180, 180, 180, 180], dtype=np.int64)
    with pytest.raises(ValueError, match="non-empty"):
        _multiscale_bound(tape, shadows, scales=())
    with pytest.raises(ValueError, match="positive"):
        _multiscale_bound(tape, shadows, scales=(0, 2))
    with pytest.raises(ValueError, match="exceeds"):
        _multiscale_bound(tape, shadows, scales=(2, 5))

    scale_bounds = {
        s: certify_ttl_shadow(
            tape,
            shadows,
            start_ns=0,
            end_ns=40,
            bucket_ns=10,
            minimum_arrival_gap_ns=10,
            cap_units=4,
            lot_units=1,
            tick_size="1",
            taker_fee="0.05",
            rate_scale=2520,
            consumption_ttl_ns=40,
            block_buckets=s,
            phase_buckets=0,
        )
        for s in (1, 2, 4)
    }
    ms = select_optimal_multiscale_partition(scale_bounds)
    verify_multiscale_ttl_partition(ms, scale_bounds)

    # Tamper with objective_units
    bad_obj = replace(ms, objective_units=ms.objective_units - 1)
    with pytest.raises(ValueError, match="objective sum mismatch"):
        verify_multiscale_ttl_partition(bad_obj, scale_bounds)

    # Tamper with contiguity
    bad_first = replace(ms, block_first_bucket=ms.block_first_bucket + 1)
    with pytest.raises(ValueError, match="start at bucket 0"):
        verify_multiscale_ttl_partition(bad_first, scale_bounds)
