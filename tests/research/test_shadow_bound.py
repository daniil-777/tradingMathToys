"""Independent finite-grid optima and adversarial shadow-certificate checks."""

from __future__ import annotations

import itertools
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from financeai.core.errors import InvalidConfigError
from financeai.domains.market.instruments import DEMO_INVERSE_A
from financeai.domains.market.reference import PolicyAction, run_reference
from financeai.domains.market.scenario import RiskLimits, Scenario, SessionPlan
from financeai.domains.market.stream import EventStream, MarketData
from financeai.research.shadow_bound import optimize_order_aware_shadow, verify_order_aware_shadow
from tests.market.synthetic import tapes_from_rows


def objective(x, b, s, u, v, cap):
    value = sum(cap * abs(int(x[i]) - int(x[i - 1])) for i in range(1, len(x)))
    return value + sum(
        max(0, min(int(uu), cap) * (int(bb) - int(xx)), min(int(vv), cap) * (int(xx) - int(ss))) if bb else 0
        for xx, bb, ss, uu, vv in zip(x, b, s, u, v, strict=True)
    )


@pytest.mark.parametrize("seed", range(40))
def test_integer_optimum_matches_exhaustive_paths(seed):
    rng = np.random.default_rng(seed)
    n = 1 + seed % 5
    b, s = rng.integers(1, 5, (2, n), dtype=np.int64)
    u, v = rng.integers(0, 10, (2, n), dtype=np.int64)
    b[rng.random(n) < 0.2] = 0
    cap = 1 + seed % 4
    result = optimize_order_aware_shadow(b, s, u, v, cap)
    best = min(objective(x, b, s, u, v, cap) for x in itertools.product(range(1, 5), repeat=n))
    assert result.objective_units == best == objective(result.shadow_rates, b, s, u, v, cap)
    assert verify_order_aware_shadow(result.shadow_rates, result.tv_dual, b, s, u, v, cap) == best


@pytest.mark.parametrize("u,v", [(1, 1), (2, 3), (5, 7), (0, 7), (8, 0)])
def test_continuous_crossed_bucket_optimum_is_enclosed(u, v):
    b, s = np.array([9], np.int64), np.array([2], np.int64)
    buy, sell = np.array([u], np.int64), np.array([v], np.int64)
    result = optimize_order_aware_shadow(b, s, buy, sell, 10)
    exact = Fraction(u * v * 7, u + v) if u + v else Fraction(0)
    assert result.objective_units - result.continuous_objective_gap_units <= exact <= result.objective_units


def test_reject_truncated_tampered_and_overflowing_witnesses():
    b, s = np.array([4], np.int64), np.array([1], np.int64)
    depth = np.ones(1, np.int64)
    result = optimize_order_aware_shadow(b, s, depth, depth, 2)
    with pytest.raises(InvalidConfigError, match="every bucket"):
        verify_order_aware_shadow(result.shadow_rates[:0], result.tv_dual, b, s, depth, depth, 2)
    with pytest.raises(AssertionError, match="subgradient"):
        verify_order_aware_shadow(np.array([4], np.int64), result.tv_dual, b, s, depth, depth, 2)
    with pytest.raises(AssertionError, match="endpoints"):
        verify_order_aware_shadow(result.shadow_rates, np.array([1, 0], np.int64), b, s, depth, depth, 2)
    huge = np.array([np.iinfo(np.int64).max], np.int64)
    with pytest.raises(InvalidConfigError, match="headroom"):
        optimize_order_aware_shadow(huge, huge, depth, depth, 2)
    with pytest.raises(InvalidConfigError, match="domain"):
        verify_order_aware_shadow(huge, result.tv_dual, b, s, depth, depth, 2)


@pytest.mark.parametrize("bad_cap", [0, -1, True, 1.5])
def test_reject_invalid_caps(bad_cap):
    a = np.ones(1, np.int64)
    with pytest.raises(InvalidConfigError):
        optimize_order_aware_shadow(a, a, a, a, bad_cap)


def test_empty_and_missing_buckets_have_zero_certificate():
    for n in (0, 5):
        a = np.zeros(n, np.int64)
        result = optimize_order_aware_shadow(a, a, a, a, 10)
        assert (
            result.objective_units
            == verify_order_aware_shadow(result.shadow_rates, result.tv_dual, a, a, a, a, 10)
            == 0
        )


def test_arbitrary_policy_reversal_is_split_by_generic_engine():
    spec = replace(DEMO_INVERSE_A, lot_size=1)
    tape = tapes_from_rows(spec, [(t, t, [(99, 100)], [(100, 100)]) for t in range(0, 100_001, 1000)])
    data = MarketData((tape,))
    stream = EventStream.build(data)

    class ReverseImmediately:
        def act(self, view):
            target = -2 if view.t < 10_000_000 else 2 if view.t < 30_000_000 else 0
            return PolicyAction((target,), new_episode=True)

    scenario = Scenario("generic", {spec.key: Decimal(0)}, 5_000_000)
    session = SessionPlan(0, 50_000_000, 70_000_000, 90_000_000, 100_000_000)
    result = run_reference(data, np.zeros(len(stream)), ReverseImmediately(), scenario, session, RiskLimits(2))
    assert result.settlement.final_position_units == (0,)
    assert len(result.orders.qty_units) == 4
    assert np.max(result.orders.qty_units) == 2
    assert np.all(np.diff(result.orders.arrival_ns) >= 10_000_000)
