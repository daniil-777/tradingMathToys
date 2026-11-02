"""Side-specific TTL retiming versus the ordinary complete execution engine."""

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from financeai.core.errors import InvalidConfigError
from financeai.domains.market.instruments import DEMO_INVERSE_A
from financeai.domains.market.scenario import RiskLimits, Scenario, SessionPlan, TimingMode
from financeai.domains.market.stream import MarketData
from financeai.research.hindsight import DepthOracle, depth_oracle, replay_depth_oracle
from financeai.research.retimed_hindsight import retime_depth_oracle
from tests.market.synthetic import tapes_from_rows

SPEC = replace(DEMO_INVERSE_A, lot_size=1)
SESSION = SessionPlan(0, 3_100_000_000, 3_500_000_000, 4_000_000_000, 4_500_000_000)


def test_opposite_side_exit_uses_better_price_before_one_second():
    tapes = tapes_from_rows(
        SPEC,
        [
            (0, 0, [(99, 2)], [(100, 2)]),
            (200_000, 200_000, [(120, 2)], [(121, 2)]),
            (1_000_000, 1_000_000, [(110, 2)], [(111, 2)]),
        ],
    )
    scenario = Scenario("side-memory", {SPEC.key: Decimal(0)}, 5_000_000)
    initial = DepthOracle(
        SPEC.key,
        np.array([0, 10**9], np.int64),
        np.array([0, 2], np.int64),
        np.array([2, 0], np.int64),
        2 / 50 - 2 / 55,
        3,
        1,
        10**9,
        0,
    )
    result = retime_depth_oracle(tapes, initial, scenario, SESSION, RiskLimits(2))
    assert result.plan.predicted_pnl > initial.predicted_pnl
    np.testing.assert_array_equal(result.plan.submit_ns, [0, 200_000_000])
    assert result.sub_ttl_opposite_side_pairs == 1
    replay = replay_depth_oracle(
        MarketData((tapes,)), (result.plan,), scenario, SESSION, RiskLimits(2), full_stream=True
    )
    assert len(replay.orders) == 2 and replay.settlement.valid


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("timing", list(TimingMode))
def test_latency_depth_haircut_mixed_sides_and_same_side_fences(seed, timing):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(61):
        p = int(rng.integers(90, 121))
        z = int(rng.integers(1, 6))
        # Exchange updates precede receipt: arrival can see a different book.
        recv = i * 50_000
        rows.append((recv, max(0, recv - 20_000), [(p, z), (p - 1, 4)], [(p + 1, z), (p + 2, 4)]))
    tapes = tapes_from_rows(SPEC, rows)
    scenario = Scenario(
        "mixed",
        {SPEC.key: Decimal(".001")},
        5_000_000,
        timing=timing,
        consumption_ttl_ns=300_000_000,
        depth_fraction=0.5,
    )
    initial = depth_oracle(tapes, scenario, SESSION, RiskLimits(3), step_units=1, spacing_ns=300_000_000, slip_ticks=2)
    result = retime_depth_oracle(tapes, initial, scenario, SESSION, RiskLimits(3))
    assert result.plan.predicted_pnl >= initial.predicted_pnl - 1e-12
    np.testing.assert_array_equal(result.plan.targets_units, initial.targets_units)
    data = MarketData((tapes,))
    compact = replay_depth_oracle(data, (result.plan,), scenario, SESSION, RiskLimits(3))
    full = replay_depth_oracle(data, (result.plan,), scenario, SESSION, RiskLimits(3), full_stream=True)
    for ledger in ("orders", "fills"):
        a, b = getattr(compact, ledger), getattr(full, ledger)
        for field in a.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_reject_same_side_memory_and_partial_fill_initial_schedules():
    tapes = tapes_from_rows(SPEC, [(t, t, [(99, 2)], [(100, 2)]) for t in (0, 100_000, 200_000, 1_000_000, 2_000_000)])
    scenario = Scenario("reject", {SPEC.key: Decimal(0)}, 5_000_000)
    bad = DepthOracle(
        SPEC.key,
        np.array([0, 100_000_000, 2_000_000_000], np.int64),
        np.array([0, 1, 4], np.int64),
        np.array([1, 2, 0], np.int64),
        0.0,
        5,
        1,
        10_000_000,
        0,
    )
    with pytest.raises(InvalidConfigError, match="consumption TTL"):
        retime_depth_oracle(tapes, bad, scenario, SESSION, RiskLimits(3))
    partial = replace(
        bad,
        submit_ns=np.array([0, 2_000_000_000], np.int64),
        observed_rows=np.array([0, 4], np.int64),
        targets_units=np.array([3, 0], np.int64),
    )
    with pytest.raises(InvalidConfigError, match="fully fillable"):
        retime_depth_oracle(tapes, partial, scenario, SESSION, RiskLimits(3))
