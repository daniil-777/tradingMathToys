"""Integer graph certificates checked against independent exact enumeration."""

from __future__ import annotations

import itertools
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from financeai.core.errors import InvalidConfigError
from financeai.domains.market.instruments import DEMO_INVERSE_A
from financeai.domains.market.scenario import RiskLimits, Scenario, SessionPlan
from financeai.domains.market.stream import MarketData
from financeai.research.adaptive_hindsight import adaptive_depth_oracle
from financeai.research.certified_hindsight import (
    _UNREACHABLE,
    _integer_curve,
    _integer_monge,
    _integer_quadratic,
    certify_adaptive_depth,
)
from financeai.research.hindsight import DepthOracle, replay_depth_oracle
from tests.market.synthetic import tapes_from_rows

SPEC = replace(DEMO_INVERSE_A, lot_size=1)


@pytest.mark.parametrize("late", [False, True])
def test_integer_monge_matches_quadratic_including_nonconcave_zero_kinks(late: bool) -> None:
    rng = np.random.default_rng(2087)
    for trial in range(500):
        half = int(rng.integers(1, 20))
        values = rng.integers(-500, 500, 2 * half + 1, dtype=np.int64)
        values[rng.random(len(values)) < 0.5] = _UNREACHABLE
        if trial % 13 == 0:
            values[:] = _UNREACHABLE
        if trial % 17 == 0:
            values[:] = 0
        buys = np.full(half + 1, _UNREACHABLE, np.int64)
        sells = buys.copy()
        buys[0] = sells[0] = 0
        for curve in (buys, sells):
            size = int(rng.integers(0, half + 1))
            # Each half is concave; the joint buy/sell kink deliberately need not be.
            curve[1 : size + 1] = np.cumsum(np.sort(rng.integers(-10, 11, size))[::-1])
        np.testing.assert_array_equal(
            _integer_monge(values, buys, sells, late), _integer_quadratic(values, buys, sells, late)
        )


@pytest.mark.parametrize("seed", range(12))
def test_certificate_encloses_fraction_exhaustive_optimum(seed: int) -> None:
    rng = np.random.default_rng(seed)
    microseconds = [0, 25_000, 50_000, 100_000, 150_000, 250_000]
    prices = rng.integers(90, 110, len(microseconds)).tolist()
    sizes = rng.integers(1, 4, len(microseconds)).tolist()
    tapes = tapes_from_rows(
        SPEC, [(t, t, [(p, z)], [(p + 1, z)]) for t, p, z in zip(microseconds, prices, sizes, strict=True)]
    )
    scenario = Scenario("test", {SPEC.key: Decimal("0.001")}, 1_000_000, consumption_ttl_ns=100_000_000)
    session = SessionPlan(0, 200_000_000, 300_000_000, 400_000_000, 500_000_000)
    best = Fraction(0)
    for middle in itertools.product(range(-2, 3), repeat=len(microseconds) - 1):
        path = (0, *middle, 0)
        value, last = Fraction(0), -(10**9)
        for i, t in enumerate(microseconds):
            p, q = path[i : i + 2]
            delta = q - p
            if not delta:
                continue
            if t - last < 100_000 or p * q < 0 or abs(delta) > sizes[i] or (t >= 200_000 and abs(q) > abs(p)):
                break
            last = t
            value += (
                Fraction(999 * delta, 500 * (prices[i] + 1)) if delta > 0 else Fraction(1001 * delta, 500 * prices[i])
            )
        else:
            best = max(best, value)
    certificates = [
        certify_adaptive_depth(
            tapes,
            scenario,
            session,
            RiskLimits(2),
            step_units=1,
            candidate_spacing_ns=1,
            scale=10**6,
            accelerated=accelerated,
        )
        for accelerated in (False, True)
    ]
    assert certificates[0].upper_numerator == certificates[1].upper_numerator
    result = certificates[0]
    upper = Fraction(result.upper_numerator, result.scale)
    assert best <= upper
    assert upper - best <= Fraction(result.max_path_rounding_error_numerator, result.scale)
    assert Fraction(result.upper_pnl) == upper


def test_multilevel_whole_lot_rewards_are_exact_upward_rounded() -> None:
    px, sz = np.array([101, 103], np.int32), np.array([5, 8], np.int32)
    curve = _integer_curve(px, sz, 2, 1, 103, 4, 3, 2, 1.0, 1998, 1)
    assert curve.tolist() == [0, 40, 80, 120]
    # A fractional depth haircut is applied by the same lot-floor semantics as replay.
    limited = _integer_curve(px, sz, 2, 1, 103, 4, 3, 2, 0.5, 1998, 1)
    assert limited.tolist() == [0, 40, _UNREACHABLE, _UNREACHABLE]


def test_low_scale_positive_zero_kink_still_certifies() -> None:
    tapes = tapes_from_rows(SPEC, [(t, t, [(100, 2)], [(101, 2)]) for t in (0, 150_000, 300_000)])
    scenario = Scenario("test", {SPEC.key: Decimal(0)}, 1_000_000, consumption_ttl_ns=100_000_000)
    session = SessionPlan(0, 250_000_000, 400_000_000, 500_000_000, 600_000_000)
    a = certify_adaptive_depth(tapes, scenario, session, RiskLimits(2), step_units=1, scale=1)
    b = certify_adaptive_depth(tapes, scenario, session, RiskLimits(2), step_units=1, scale=1, accelerated=False)
    assert a.upper_numerator == b.upper_numerator == 2
    # Exact constant-price best profit is zero; artificial rounded profit is only an upper bound.
    assert Fraction(a.upper_pnl) >= 0


def test_empty_window_and_rejected_integer_overflow() -> None:
    tapes = tapes_from_rows(SPEC, [(0, 0, [(99, 1)], [(100, 1)])])
    scenario = Scenario("test", {SPEC.key: Decimal(0)}, 5_000_000)
    session = SessionPlan(10**9, 2 * 10**9, 3 * 10**9, 4 * 10**9, 5 * 10**9)
    assert certify_adaptive_depth(tapes, scenario, session, RiskLimits(2), step_units=1).upper_numerator == 0
    with pytest.raises(InvalidConfigError, match="headroom"):
        certify_adaptive_depth(tapes, scenario, session, RiskLimits(2), step_units=1, scale=10**30)
    with pytest.raises(InvalidConfigError, match="power of ten"):
        certify_adaptive_depth(tapes, scenario, session, RiskLimits(2), step_units=1, scale=3)
    long_tapes = tapes_from_rows(SPEC, [(t, t, [(1, 2)], [(2, 2)]) for t in (0, 10**12)])
    long_session = SessionPlan(0, 2 * 10**15, 3 * 10**15, 4 * 10**15, 5 * 10**15)
    with pytest.raises(InvalidConfigError, match="path bound"):
        certify_adaptive_depth(long_tapes, scenario, long_session, RiskLimits(2), step_units=1, scale=10**18)


def test_full_engine_can_beat_the_certified_cooldown_graph() -> None:
    tapes = tapes_from_rows(
        SPEC,
        [
            (0, 0, [(100, 2)], [(101, 2)]),
            (20_000, 20_000, [(110, 2)], [(111, 2)]),
            (30_000, 30_000, [(100, 2)], [(101, 2)]),
            (250_000, 250_000, [(100, 2)], [(101, 2)]),
        ],
    )
    scenario = Scenario("counterexample", {SPEC.key: Decimal(0)}, 5_000_000)
    session = SessionPlan(0, 200_000_000, 300_000_000, 400_000_000, 500_000_000)
    graph = adaptive_depth_oracle(
        tapes, scenario, session, RiskLimits(2), step_units=1, candidate_spacing_ns=1, slip_ticks=0
    )
    bound = certify_adaptive_depth(tapes, scenario, session, RiskLimits(2), step_units=1, slip_ticks=0)
    assert graph.predicted_pnl == bound.upper_numerator == 0
    score = 2 / (101 * 0.5) - 2 / (110 * 0.5)
    plan = DepthOracle(
        SPEC.key,
        np.array([0, 20_000_000], np.int64),
        np.array([0, 1], np.int64),
        np.array([2, 0], np.int64),
        score,
        4,
        1,
        20_000_000,
        0,
    )
    result = replay_depth_oracle(MarketData((tapes,)), (plan,), scenario, session, RiskLimits(2), full_stream=True)
    assert result.settlement.final_position_units == (0,)
    assert result.settlement.net_pnl == pytest.approx(0.0032403240324032426)
