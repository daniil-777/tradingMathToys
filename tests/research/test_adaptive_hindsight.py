"""Exhaustive cooldown paths and adversarial Monge transition comparisons."""

from __future__ import annotations

import itertools
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from financeai.core.errors import InvalidConfigError
from financeai.domains.market.instruments import DEMO_INVERSE_A
from financeai.domains.market.scenario import RiskLimits, Scenario, SessionPlan
from financeai.domains.market.stream import MarketData
from financeai.research.adaptive_hindsight import (
    _transition_monge,
    _transition_quadratic,
    adaptive_depth_oracle,
)
from financeai.research.hindsight import depth_oracle, replay_depth_oracle
from tests.market.synthetic import tapes_from_rows

SPEC = replace(DEMO_INVERSE_A, lot_size=1)


@pytest.mark.parametrize("late", [False, True])
def test_monge_matches_exhaustive_for_arbitrary_values_depth_and_ties(late: bool) -> None:
    rng = np.random.default_rng(572)
    for trial in range(300):
        half = int(rng.integers(1, 24))
        values = rng.normal(0, 2, 2 * half + 1)
        values[rng.random(len(values)) < 0.6] = -np.inf
        if trial % 5 == 0:
            values[:] = -np.inf  # every midpoint infeasible
        if trial % 7 == 0:
            values[:] = 0  # deliberately tied predecessor scores
        buys, sells = np.full(half + 1, -np.inf), np.full(half + 1, -np.inf)
        buys[0] = sells[0] = 0
        buy_n, sell_n = rng.integers(0, half + 1, 2)
        buys[1 : buy_n + 1] = np.cumsum(np.sort(rng.uniform(0.1, 0.5, buy_n))[::-1])
        sells[1 : sell_n + 1] = -np.cumsum(np.sort(rng.uniform(0.6, 0.9, sell_n)))
        exact, _ = _transition_quadratic(values, buys, sells, late)
        fast, _ = _transition_monge(values, buys, sells, late)
        np.testing.assert_allclose(fast, exact, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("seed", range(16))
def test_cooldown_optimum_against_all_integer_paths_and_replay(seed: int) -> None:
    rng = np.random.default_rng(seed)
    times = [0, 25_000, 50_000, 100_000, 150_000, 250_000]  # microseconds
    prices = rng.integers(90, 110, len(times)).tolist()
    sizes = rng.integers(1, 4, len(times)).tolist()
    tapes = tapes_from_rows(
        SPEC, [(t, t, [(p, z)], [(p + 1, z)]) for t, p, z in zip(times, prices, sizes, strict=True)]
    )
    scenario = Scenario("test", {SPEC.key: Decimal("0.001")}, 1_000_000, consumption_ttl_ns=100_000_000)
    session = SessionPlan(0, 200_000_000, 300_000_000, 400_000_000, 500_000_000)
    best = 0.0
    for middle in itertools.product(range(-2, 3), repeat=len(times) - 1):
        path = (0, *middle, 0)
        value, last = 0.0, -(10**9)
        for i, t in enumerate(times):
            p, q = path[i : i + 2]
            d = q - p
            if d == 0:
                continue
            if t - last < 100_000 or p * q < 0 or abs(d) > sizes[i] or (t >= 200_000 and abs(q) > abs(p)):
                break
            last = t
            value += d * (0.999 / ((prices[i] + 1) * 0.5) if d > 0 else 1.001 / (prices[i] * 0.5))
        else:
            best = max(best, value)
    for accelerated in (False, True):
        plan = adaptive_depth_oracle(
            tapes,
            scenario,
            session,
            RiskLimits(2),
            step_units=1,
            candidate_spacing_ns=1,
            include_spacing_ns=(100_000_000,),
            slip_ticks=0,
            accelerated=accelerated,
        )
        assert plan.predicted_pnl == pytest.approx(best, abs=1e-12)
        assert np.all(np.diff(plan.submit_ns) >= scenario.consumption_ttl_ns)
        data = MarketData((tapes,))
        compact = replay_depth_oracle(data, (plan,), scenario, session, RiskLimits(2))
        full = replay_depth_oracle(data, (plan,), scenario, session, RiskLimits(2), full_stream=True)
        for name in full.fills.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(full.fills, name), getattr(compact.fills, name))


def test_adaptive_timing_strictly_improves_fixed_grid() -> None:
    prices = [100, 100, 90, 100, 100, 100, 110, 100, 100]
    tapes = tapes_from_rows(SPEC, [(i * 25_000, i * 25_000, [(p, 2)], [(p + 1, 2)]) for i, p in enumerate(prices)])
    scenario = Scenario("test", {SPEC.key: Decimal(0)}, 1_000_000, consumption_ttl_ns=100_000_000)
    session = SessionPlan(0, 200_000_000, 300_000_000, 400_000_000, 500_000_000)
    base = depth_oracle(tapes, scenario, session, RiskLimits(2), step_units=1, spacing_ns=100_000_000)
    improved = adaptive_depth_oracle(
        tapes,
        scenario,
        session,
        RiskLimits(2),
        step_units=1,
        candidate_spacing_ns=25_000_000,
        include_spacing_ns=(100_000_000,),
    )
    assert base.predicted_pnl == 0
    assert improved.predicted_pnl > 0.007


@pytest.mark.parametrize("block", [1, 2, 3, 7, 20])
@pytest.mark.parametrize("accelerated", [False, True])
def test_checkpoint_backtracking_matches_full_parent_table(block: int, accelerated: bool) -> None:
    times = [0, 10_000, 30_000, 50_000, 200_000, 210_000, 220_000, 600_000, 620_000]
    prices = [100, 95, 90, 110, 100, 120, 130, 90, 105]
    tapes = tapes_from_rows(SPEC, [(t, t, [(p, 3)], [(p + 1, 3)]) for t, p in zip(times, prices, strict=True)])
    scenario = Scenario("checkpoint", {SPEC.key: Decimal("0.001")}, 1_000_000, consumption_ttl_ns=20_000_000)
    session = SessionPlan(0, 610_000_000, 800_000_000, 900_000_000, 10**9)
    kwargs = {
        "step_units": 1,
        "candidate_spacing_ns": 1,
        "include_spacing_ns": (),
        "slip_ticks": 0,
        "accelerated": accelerated,
    }
    baseline = adaptive_depth_oracle(tapes, scenario, session, RiskLimits(3), **kwargs)
    checkpoint = adaptive_depth_oracle(tapes, scenario, session, RiskLimits(3), checkpoint_rows=block, **kwargs)
    assert checkpoint.predicted_pnl == baseline.predicted_pnl
    np.testing.assert_array_equal(checkpoint.submit_ns, baseline.submit_ns)
    np.testing.assert_array_equal(checkpoint.targets_units, baseline.targets_units)
    np.testing.assert_array_equal(checkpoint.observed_rows, baseline.observed_rows)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_reject_invalid_checkpoint_block(bad) -> None:
    tapes = tapes_from_rows(SPEC, [(0, 0, [(99, 1)], [(100, 1)])])
    scenario = Scenario("test", {SPEC.key: Decimal(0)}, 1_000_000)
    session = SessionPlan(0, 10**9, 2 * 10**9, 3 * 10**9, 4 * 10**9)
    with pytest.raises(InvalidConfigError, match="checkpoint_rows"):
        adaptive_depth_oracle(tapes, scenario, session, RiskLimits(2), step_units=1, checkpoint_rows=bad)


def test_empty_tape_window_and_rejected_cooldown() -> None:
    tapes = tapes_from_rows(SPEC, [(0, 0, [(99, 1)], [(100, 1)])])
    scenario = Scenario("test", {SPEC.key: Decimal(0)}, 5_000_000)
    session = SessionPlan(10**9, 2 * 10**9, 3 * 10**9, 4 * 10**9, 5 * 10**9)
    plan = adaptive_depth_oracle(tapes, scenario, session, RiskLimits(2), step_units=1)
    assert plan.predicted_pnl == 0 and not len(plan.submit_ns)
    with pytest.raises(InvalidConfigError, match="cooldown"):
        adaptive_depth_oracle(tapes, scenario, session, cooldown_ns=10_000_000)
    with pytest.raises(InvalidConfigError, match="candidate spacings"):
        adaptive_depth_oracle(tapes, scenario, session, candidate_spacing_ns=2**63)
