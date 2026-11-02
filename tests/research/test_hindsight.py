"""Independent exhaustive optima and actual-engine feasibility for hindsight solvers."""

from __future__ import annotations

import itertools
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from financeai.core.errors import InvalidConfigError
from financeai.domains.market.instruments import DEMO_INVERSE_A
from financeai.domains.market.scenario import RiskLimits, Scenario, SessionPlan, TimingMode
from financeai.domains.market.stream import MarketData
from financeai.research.hindsight import depth_oracle, quote_oracle, replay_depth_oracle
from tests.market.synthetic import tapes_from_rows

TEST_INSTRUMENT = replace(DEMO_INVERSE_A, lot_size=1)


def market(prices: list[int], sizes: list[int] | None = None) -> MarketData:
    sizes = sizes or [3] * len(prices)
    return MarketData(
        (
            tapes_from_rows(
                TEST_INSTRUMENT,
                [
                    (i * 100_000, i * 100_000, [(p, z)], [(p + 1, z)])
                    for i, (p, z) in enumerate(zip(prices, sizes, strict=True))
                ],
            ),
        )
    )


@pytest.mark.parametrize("seed", range(20))
def test_quote_primal_equals_exhaustive_all_integer_inventories(seed: int) -> None:
    rng = np.random.default_rng(seed)
    prices = rng.integers(90, 110, 5).tolist()
    data = market(prices)
    fee, cap = Decimal("0.001"), 2
    best = -np.inf
    for middle in itertools.product(range(-cap, cap + 1), repeat=4):
        path = (0, *middle, 0)
        value = 0.0
        for i, p in enumerate(prices):
            delta = path[i + 1] - path[i]
            value += delta * (0.999 / ((p + 1) * 0.5) if delta > 0 else 1.001 / (p * 0.5))
        best = max(best, value)
    plan = quote_oracle(data.venues[0], fee, cap_units=cap)
    assert plan.score_pnl == pytest.approx(best, abs=1e-12)
    assert plan.numerical_dual_pnl == pytest.approx(best, abs=1e-12)
    assert sum(plan.delta_units) == 0
    assert all(abs(q) <= cap for q in plan.targets_units)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_depth_optimum_equals_exhaustive_and_full_replay(seed: int, fraction: float) -> None:
    rng = np.random.default_rng(seed)
    prices = rng.integers(90, 110, 5).tolist()
    sizes = rng.integers(1, 5, 5).tolist()
    data = market(prices, sizes)
    scenario = Scenario(
        "test",
        {DEMO_INVERSE_A.key: Decimal("0.001")},
        1_000_000,
        consumption_ttl_ns=100_000_000,
        depth_fraction=fraction,
    )
    session = SessionPlan(0, 300_000_000, 500_000_000, 600_000_000, 700_000_000)
    cap = 2
    best = 0.0
    for middle in itertools.product(range(-cap, cap + 1), repeat=4):
        path = (0, *middle, 0)
        value = 0.0
        for i, p in enumerate(prices):
            prev, target = path[i : i + 2]
            delta = target - prev
            if prev * target < 0 or abs(delta) > int(sizes[i] * fraction) or (i >= 3 and abs(target) > abs(prev)):
                break
            value += delta * (0.999 / ((p + 1) * 0.5) if delta > 0 else 1.001 / (p * 0.5))
        else:
            best = max(best, value)
    plan = depth_oracle(
        data.venues[0], scenario, session, RiskLimits(cap), step_units=1, spacing_ns=100_000_000, slip_ticks=0
    )
    assert plan.predicted_pnl == pytest.approx(best, abs=1e-12)
    sparse = replay_depth_oracle(data, (plan,), scenario, session, RiskLimits(cap))
    full = replay_depth_oracle(data, (plan,), scenario, session, RiskLimits(cap), full_stream=True)
    for name in full.fills.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(sparse.fills, name), getattr(full.fills, name))


@pytest.mark.parametrize("timing", list(TimingMode))
def test_latency_depth_slip_duplicates_and_empty_plan(timing: TimingMode) -> None:
    # At the first receipt cohort the last row wins; latency changes the arrival book.
    tapes = tapes_from_rows(
        TEST_INSTRUMENT,
        [
            (0, 0, [(98, 2)], [(100, 2), (101, 4)]),
            (0, 0, [(99, 2)], [(100, 2), (101, 4)]),
            (5_000, 3_000, [(100, 2)], [(101, 2), (102, 4)]),
            (100_000, 98_000, [(120, 2), (119, 4)], [(121, 6)]),
            (200_000, 198_000, [(95, 6)], [(96, 6)]),
            (300_000, 298_000, [], []),
            (400_000, 398_000, [(110, 6)], [(111, 6)]),
        ],
    )
    scenario = Scenario(
        "test", {DEMO_INVERSE_A.key: Decimal("0.001")}, 5_000_000, timing=timing, consumption_ttl_ns=100_000_000
    )
    session = SessionPlan(0, 400_000_000, 500_000_000, 600_000_000, 700_000_000)
    data = MarketData((tapes,))
    for slip in (0, 2):
        plan = depth_oracle(
            tapes, scenario, session, RiskLimits(6), step_units=2, spacing_ns=100_000_000, slip_ticks=slip
        )
        full = replay_depth_oracle(data, (plan,), scenario, session, RiskLimits(6), full_stream=True)
        assert full.settlement.valid
        assert plan.predicted_pnl >= 0
    late = SessionPlan(500_000_000, 600_000_000, 700_000_000, 800_000_000, 900_000_000)
    plan = depth_oracle(tapes, scenario, late, RiskLimits(6), step_units=2, spacing_ns=100_000_000)
    assert replay_depth_oracle(data, (plan,), scenario, late, RiskLimits(6)).settlement.net_pnl == 0


def test_reject_spacing_that_leaves_hidden_state_and_invalid_grid() -> None:
    tapes = market([100, 110]).venues[0]
    scenario = Scenario("test", {DEMO_INVERSE_A.key: Decimal(0)}, 5_000_000)
    session = SessionPlan(0, 10**8, 2 * 10**8, 3 * 10**8, 4 * 10**8)
    with pytest.raises(InvalidConfigError, match="spacing"):
        depth_oracle(tapes, scenario, session, spacing_ns=10_000_000)
    with pytest.raises(InvalidConfigError, match="step"):
        depth_oracle(tapes, scenario, session, step_units=3)
